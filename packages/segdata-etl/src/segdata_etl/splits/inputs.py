"""Per-source inputs of the split builder, read from pinned dataset and taxonomy versions.

A CoralNet dataset group is ``s<source_id>``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import yaml

from segdata_spec.manifest import Pin
from segdata_spec.read import DatasetVersion, open_version
from segdata_spec.sync import sync
from segdata_spec.taxonomy import roll_up


def source_id_of(group: str) -> int:
    """Return the CoralNet source id of dataset group ``s<source_id>``."""
    return int(group[1:])


def _source_ids(groups: pd.Series) -> pd.Series:
    return groups.map(source_id_of)


def images_per_source(images: pa.Table) -> pd.Series:
    """Distinct images per source id, from ``images.parquet``."""
    df = images.select(["image_id", "group"]).to_pandas()
    return df.groupby(_source_ids(df["group"]))["image_id"].nunique().rename_axis("source_id")


def class_counts(
    label_counts: pa.Table,
    labels: pa.Table,
    images: pa.Table,
    hierarchy: dict[str, str | None],
    subset: set[str],
) -> pd.DataFrame:
    """Points per source id (rows) and training class (columns).

    Each label's ``mermaid_benthic_attribute``, lower-cased, is rolled up into ``subset``; points
    of unmapped labels and of labels with no ancestor in ``subset`` are not counted.
    """
    names = labels.select(["label_id", "mermaid_benthic_attribute"]).to_pandas()
    label_to_cls = {
        int(lid): roll_up(name.lower(), hierarchy, subset) if isinstance(name, str) else None
        for lid, name in zip(names["label_id"], names["mermaid_benthic_attribute"], strict=True)
    }
    counts = label_counts.to_pandas()
    counts["cls"] = counts["label_id"].map(label_to_cls)
    group_of = images.select(["image_id", "group"]).to_pandas().drop_duplicates("image_id")
    counts = counts.dropna(subset=["cls"]).merge(group_of, on="image_id")
    counts["source_id"] = _source_ids(counts["group"])
    summed = counts.groupby(["source_id", "cls"])["n"].sum()
    return summed.unstack(fill_value=0).rename_axis(columns=None)  # noqa: PD010


def image_keys(images: pa.Table) -> pd.DataFrame:
    """``source_id``, ``crc64nvme`` and ``size`` per image, for exact-duplicate matching."""
    df = images.select(["group", "crc64nvme", "size"]).to_pandas()
    return pd.DataFrame(
        {"source_id": _source_ids(df["group"]), "crc64nvme": df["crc64nvme"], "size": df["size"]}
    )


def require_crc64nvme(owners: pd.Series, crc64nvme: pd.Series) -> None:
    """Refuse images without a CRC64NVME, which exact-duplicate matching cannot compare.

    Args:
        owners: The source or group of each image.
        crc64nvme: The CRC64NVME of each image, aligned with ``owners``.

    Raises:
        ValueError: Names the count of such images and their sorted owners.
    """
    missing = owners[crc64nvme.isna()]
    if len(missing):
        raise ValueError(
            f"{len(missing)} images without a crc64nvme in {sorted(set(missing.tolist()))}"
        )


def read_source_list(path: Path) -> set[int]:
    """Source ids from a text file, one per line (``#`` starts a comment)."""
    lines = (ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines())
    return {int(ln) for ln in lines if ln}


def read_class_subset(training_config: Path) -> set[str]:
    """The lower-cased ``training.class_subset`` of a training config."""
    config: dict[str, Any] = yaml.safe_load(training_config.read_text())
    return {c.lower() for c in config["training"]["class_subset"]}


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_tables(
    client: Any,
    pins: Mapping[str, Pin],
    cache_dir: Path,
    bucket: str,
    taxonomy: Pin | None = None,
) -> dict[str, DatasetVersion]:
    """Sync the pinned versions' tables and the taxonomy, but no image; open the datasets.

    A selection of no groups per dataset makes ``sync`` skip every image and mask.
    """
    no_images: dict[str, list[str]] = {name: [] for name in pins}
    sync(client, pins, cache_dir, bucket=bucket, selection=no_images, taxonomy=taxonomy)
    return {name: open_version(cache_dir, bucket, pin) for name, pin in pins.items()}
