"""Helpers shared by the versioned-dataset builders that make no S3 calls."""

from __future__ import annotations

import pyarrow as pa

from segdata_spec.layout import image_prefix
from segdata_spec.manifest import ImageStore
from segdata_spec.schema import LABEL_COUNTS_SCHEMA


def count_labels(points: pa.Table) -> pa.Table:
    """Build ``stats/label_counts.parquet``: points per ``(image_id, label_id)``, sorted by both."""
    counts = (
        points.group_by(["image_id", "label_id"], use_threads=False)
        .aggregate([("row", "count")])
        .sort_by([("image_id", "ascending"), ("label_id", "ascending")])
    )
    return pa.table(
        {
            "image_id": counts.column("image_id"),
            "label_id": counts.column("label_id"),
            "n": counts.column("row_count"),
        }
    ).cast(LABEL_COUNTS_SCHEMA)


def image_store(bucket: str, dataset: str) -> ImageStore:
    """Return the segdata-owned image store of ``dataset`` in ``bucket``."""
    return ImageStore(bucket=bucket, prefix=image_prefix(dataset), owner="segdata")
