"""Open synced dataset and taxonomy versions from the local cache, without S3 access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from segdata_spec._marker import load_synced, marker_files
from segdata_spec.errors import NotSynced, PinMismatch
from segdata_spec.layout import (
    FEATURE_INDEX_NAME,
    MANIFEST_NAME,
    TABLES_MARKER_ID,
    cache_path,
    dataset_prefix,
    feature_set_prefix,
    resize_id,
    resized_cache_path,
    synced_feature_marker_dir,
    synced_marker_dir,
    synced_resize_ids,
    taxonomy_prefix,
)
from segdata_spec.manifest import (
    FeatureSetManifest,
    FeatureSetPin,
    Manifest,
    Pin,
    TaxonomyManifest,
    _check_dataset_pin,
    manifest_sha256,
)
from segdata_spec.resize import ResizeSpec, scale_images, scale_points


@dataclass(frozen=True)
class DatasetVersion:
    """A synced dataset version: its manifest, its tables and its cached image paths.

    With ``resize`` set, the tables and paths describe the images resized per that spec.
    ``synced_groups`` is the set of groups whose images ``sync`` cached, or None when all of them.
    """

    manifest: Manifest
    images: pa.Table
    points: pa.Table | None
    labels: pa.Table
    label_counts: pa.Table
    cache_dir: Path
    resize: ResizeSpec | None = None
    synced_groups: frozenset[str] | None = None

    def _object_path(self, key: str) -> Path:
        store = self.manifest.image_store
        if self.resize is None:
            return cache_path(self.cache_dir, store.bucket, store.prefix + key)
        return resized_cache_path(self.cache_dir, self.resize, store.bucket, store.prefix + key)

    def image_path(self, row: Mapping[str, Any]) -> Path:
        """Return the cached image file for an ``images`` row (a mapping with ``key``)."""
        return self._object_path(row["key"])

    def mask_path(self, row: Mapping[str, Any]) -> Path:
        """Return the cached mask file for an ``images`` row (a mapping with ``mask_key``).

        Raises:
            ValueError: The row has no mask.
        """
        if row["mask_key"] is None:
            raise ValueError(f"image {row['image_id']} has no mask")
        return self._object_path(row["mask_key"])


@dataclass(frozen=True)
class VersionTables:
    """The manifest and tables of a synced dataset version, without its images."""

    manifest: Manifest
    images: pa.Table
    points: pa.Table | None
    labels: pa.Table
    label_counts: pa.Table


@dataclass(frozen=True)
class FeatureSet:
    """A synced feature set: its manifest, its index and the cached feature file of each image.

    ``path`` resolves any indexed image; only the images ``sync_feature_set`` selected are cached.
    """

    manifest: FeatureSetManifest
    index: pa.Table
    cache_dir: Path
    bucket: str
    keys: Mapping[str, str]

    def path(self, image_id: str) -> Path:
        """Return the cached feature file of ``image_id``.

        Raises:
            KeyError: ``image_id`` is not in the index.
        """
        return cache_path(self.cache_dir, self.bucket, self.keys[image_id])


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


def _read_pinned_manifest(cache_dir: Path, bucket: str, prefix: str, sha256: str) -> bytes:
    path = _require(cache_path(cache_dir, bucket, prefix + MANIFEST_NAME))
    data = path.read_bytes()
    if manifest_sha256(data) != sha256:
        raise PinMismatch(f"cached file {path} does not match the pin")
    return data


def _read_tables(
    cache_dir: Path, bucket: str, prefix: str, names: Mapping[str, Any]
) -> dict[str, pa.Table]:
    return {
        name: pq.read_table(_require(cache_path(cache_dir, bucket, prefix + name)))
        for name in names
    }


def _require_marker_dir(cache_dir: Path, bucket: str, pin: Pin, resize: ResizeSpec | None) -> Path:
    directory = synced_marker_dir(cache_dir, resize_id(resize), bucket, pin.name, pin.version)
    if marker_files(directory):
        return directory
    others = synced_resize_ids(cache_dir, bucket, pin.name, pin.version)
    raise NotSynced(
        f"{pin.name} {pin.version} was not synced with resize id {resize_id(resize)!r}; "
        f"resize ids synced for it: {others or 'none'}"
    )


def _read_version(cache_dir: Path, bucket: str, pin: Pin) -> tuple[Manifest, dict[str, pa.Table]]:
    """Return the cached manifest, re-verified against ``pin``, and the tables it lists."""
    prefix = dataset_prefix(pin.name, pin.version)
    manifest = Manifest.model_validate_json(
        _read_pinned_manifest(cache_dir, bucket, prefix, pin.manifest_sha256)
    )
    return manifest, _read_tables(cache_dir, bucket, prefix, manifest.files)


def _without_images(table: pa.Table, image_ids: frozenset[str]) -> pa.Table:
    """Return the rows of ``table`` whose ``image_id`` is not in ``image_ids``."""
    if not image_ids:
        return table
    value_set = pa.array(sorted(image_ids), pa.string())
    return table.filter(pc.invert(pc.is_in(table.column("image_id"), value_set=value_set)))


def open_version(
    cache_dir: Path, bucket: str, pin: Pin, resize: ResizeSpec | None = None
) -> DatasetVersion:
    """Open a dataset version that ``sync`` cached, re-verifying its manifest against ``pin``.

    ``sync`` must have run for the pin with the same ``resize`` (None for the original images);
    the union of its markers supplies ``DatasetVersion.synced_groups``. With ``resize``,
    ``images`` carries the resized ``width`` and ``height`` and ``media_type`` ``image/jpeg``,
    ``points`` are scaled from the
    original sizes into the resized pixel grid, and ``image_path`` and ``mask_path`` resolve into
    ``<cache_dir>/resized/<resize.id>/``.

    Images that a sync skipped because their original does not decode (the markers' ``skipped``
    ids) are left out of ``images``, ``points`` and ``label_counts``.

    Raises:
        NotSynced: No marker exists for ``pin`` and ``resize``.
        CorruptSyncMarker: A marker file cannot be parsed.
        FileNotFoundError: The manifest or a table it lists is not in the cache.
        PinMismatch: The cached manifest differs from ``pin``.
        ValueError: With ``resize``, a point references an ``image_id`` absent from ``images``.
    """
    marker = load_synced(_require_marker_dir(cache_dir, bucket, pin, resize))
    manifest, tables = _read_version(cache_dir, bucket, pin)
    images = _without_images(tables["images.parquet"], marker.skipped)
    points = tables.get("points.parquet")
    if points is not None:
        points = _without_images(points, marker.skipped)
    label_counts = _without_images(tables["stats/label_counts.parquet"], marker.skipped)
    if resize is not None:
        if points is not None:
            points = scale_points(points, images, resize)
        images = scale_images(images, resize)
    return DatasetVersion(
        manifest=manifest,
        images=images,
        points=points,
        labels=tables["labels.parquet"],
        label_counts=label_counts,
        cache_dir=cache_dir,
        resize=resize,
        synced_groups=marker.groups,
    )


def open_tables(cache_dir: Path, bucket: str, pin: Pin) -> VersionTables:
    """Open the tables of a dataset version that ``sync`` cached, with or without its images.

    ``sync_tables`` or ``sync`` (with any resize spec) must have run for the pin. The
    tables describe the original images; the manifest is re-verified against ``pin``.

    Raises:
        NotSynced: No tables-only or image marker exists for ``pin``.
        FileNotFoundError: The manifest or a table it lists is not in the cache.
        PinMismatch: The cached manifest differs from ``pin``.
    """
    tables_dir = synced_marker_dir(cache_dir, TABLES_MARKER_ID, bucket, pin.name, pin.version)
    if not marker_files(tables_dir) and not synced_resize_ids(
        cache_dir, bucket, pin.name, pin.version
    ):
        raise NotSynced(f"{pin.name} {pin.version} has no sync marker; run sync first")
    manifest, tables = _read_version(cache_dir, bucket, pin)
    return VersionTables(
        manifest=manifest,
        images=tables["images.parquet"],
        points=tables.get("points.parquet"),
        labels=tables["labels.parquet"],
        label_counts=tables["stats/label_counts.parquet"],
    )


def open_taxonomy(cache_dir: Path, bucket: str, pin: Pin) -> Taxonomy:
    """Open a taxonomy version that ``sync`` cached, re-verifying its manifest against ``pin``.

    Raises:
        FileNotFoundError: The manifest or a table it lists is not in the cache.
        PinMismatch: The cached manifest differs from ``pin``.
    """
    prefix = taxonomy_prefix(pin.name, pin.version)
    data = _read_pinned_manifest(cache_dir, bucket, prefix, pin.manifest_sha256)
    manifest = TaxonomyManifest.model_validate_json(data)
    tables = _read_tables(cache_dir, bucket, prefix, manifest.files)
    return Taxonomy(
        manifest=manifest,
        benthic_attributes=tables["benthic_attributes.parquet"],
        growth_forms=tables["growth_forms.parquet"],
    )


def open_feature_set(cache_dir: Path, bucket: str, pin: FeatureSetPin) -> FeatureSet:
    """Open a feature set that ``sync_feature_set`` cached, re-verifying its manifest sha256.

    Raises:
        NotSynced: No feature-set marker exists for ``pin``.
        FileNotFoundError: The set manifest or its index is not in the cache.
        PinMismatch: The cached set manifest differs from the pin's ``manifest_sha256``, or records a
            dataset manifest sha256 other than the pin's dataset.
    """
    dataset, family = pin.dataset, pin.family
    directory = synced_feature_marker_dir(cache_dir, family, bucket, dataset.name, dataset.version)
    if not marker_files(directory):
        raise NotSynced(
            f"feature set {family!r} of {dataset.name} {dataset.version} was not synced"
        )
    prefix = feature_set_prefix(dataset.name, dataset.version, family)
    data = _read_pinned_manifest(cache_dir, bucket, prefix, pin.manifest_sha256)
    manifest = FeatureSetManifest.model_validate_json(data)
    _check_dataset_pin(manifest, dataset)
    index = pq.read_table(_require(cache_path(cache_dir, bucket, prefix + FEATURE_INDEX_NAME)))
    ids = index.column("image_id").to_pylist()
    return FeatureSet(
        manifest=manifest,
        index=index,
        cache_dir=cache_dir,
        bucket=bucket,
        keys=dict(zip(ids, index.column("key").to_pylist(), strict=True)),
    )
