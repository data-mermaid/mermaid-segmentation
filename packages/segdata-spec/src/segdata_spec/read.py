"""Open synced dataset and taxonomy versions from the local cache, without S3 access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from segdata_spec._s3 import MANIFEST_NAME, cache_path, dataset_prefix, taxonomy_prefix
from segdata_spec.errors import PinMismatch
from segdata_spec.manifest import Manifest, Pin, TaxonomyManifest, manifest_sha256


@dataclass(frozen=True)
class DatasetVersion:
    """A synced dataset version: its manifest, its tables and its cached image paths."""

    manifest: Manifest
    images: pa.Table
    points: pa.Table | None
    labels: pa.Table
    label_counts: pa.Table
    cache_dir: Path

    def image_path(self, row: Mapping[str, Any]) -> Path:
        """Return the cached image file for an ``images`` row (a mapping with ``key``)."""
        store = self.manifest.image_store
        return cache_path(self.cache_dir, store.bucket, store.prefix + row["key"])

    def mask_path(self, row: Mapping[str, Any]) -> Path:
        """Return the cached mask file for an ``images`` row (a mapping with ``mask_key``).

        Raises:
            ValueError: The row has no mask.
        """
        if row["mask_key"] is None:
            raise ValueError(f"image {row['image_id']} has no mask")
        store = self.manifest.image_store
        return cache_path(self.cache_dir, store.bucket, store.prefix + row["mask_key"])


@dataclass(frozen=True)
class Taxonomy:
    """A synced taxonomy version."""

    manifest: TaxonomyManifest
    benthic_attributes: pa.Table
    growth_forms: pa.Table


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not in the cache; run sync first")
    return path


def _read_pinned_manifest(cache_dir: Path, bucket: str, prefix: str, pin: Pin) -> bytes:
    path = _require(cache_path(cache_dir, bucket, prefix + MANIFEST_NAME))
    data = path.read_bytes()
    if manifest_sha256(data) != pin.manifest_sha256:
        raise PinMismatch(f"cached file {path} does not match the pin")
    return data


def _read_tables(
    cache_dir: Path, bucket: str, prefix: str, names: Mapping[str, Any]
) -> dict[str, pa.Table]:
    return {
        name: pq.read_table(_require(cache_path(cache_dir, bucket, prefix + name)))
        for name in names
    }


def open_version(cache_dir: Path, bucket: str, pin: Pin) -> DatasetVersion:
    """Open a dataset version that ``sync`` cached, re-verifying its manifest against ``pin``.

    Raises:
        FileNotFoundError: The manifest or a table it lists is not in the cache.
        PinMismatch: The cached manifest differs from ``pin``.
    """
    prefix = dataset_prefix(pin.name, pin.version)
    manifest = Manifest.model_validate_json(_read_pinned_manifest(cache_dir, bucket, prefix, pin))
    tables = _read_tables(cache_dir, bucket, prefix, manifest.files)
    return DatasetVersion(
        manifest=manifest,
        images=tables["images.parquet"],
        points=tables.get("points.parquet"),
        labels=tables["labels.parquet"],
        label_counts=tables["stats/label_counts.parquet"],
        cache_dir=cache_dir,
    )


def open_taxonomy(cache_dir: Path, bucket: str, pin: Pin) -> Taxonomy:
    """Open a taxonomy version that ``sync`` cached, re-verifying its manifest against ``pin``.

    Raises:
        FileNotFoundError: The manifest or a table it lists is not in the cache.
        PinMismatch: The cached manifest differs from ``pin``.
    """
    prefix = taxonomy_prefix(pin.name, pin.version)
    data = _read_pinned_manifest(cache_dir, bucket, prefix, pin)
    manifest = TaxonomyManifest.model_validate_json(data)
    tables = _read_tables(cache_dir, bucket, prefix, manifest.files)
    return Taxonomy(
        manifest=manifest,
        benthic_attributes=tables["benthic_attributes.parquet"],
        growth_forms=tables["growth_forms.parquet"],
    )
