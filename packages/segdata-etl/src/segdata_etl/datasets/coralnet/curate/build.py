"""Curated CoralNet v1 tables from a raw version."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from segdata_etl.common.versions import count_labels
from segdata_etl.datasets.coralnet.curate.overrides import OverrideStats, apply_overrides
from segdata_spec.read import DatasetVersion
from segdata_spec.schema import IMAGES_SCHEMA, POINTS_SCHEMA


@dataclass(frozen=True)
class CuratedTables:
    """The curated ``images``, ``points`` and ``label_counts`` tables, with what was dropped."""

    images: pa.Table
    points: pa.Table
    label_counts: pa.Table
    dropped: dict[str, int]
    override_stats: OverrideStats


def _curate_frames(
    images: pd.DataFrame,
    points: pd.DataFrame,
    overrides: pd.DataFrame,
    blacklist_ids: set[str],
) -> CuratedTables:
    """Apply overrides, then drop blacklisted images and their points."""
    points, stats = apply_overrides(points, images, overrides)
    blacklisted = images["image_id"].isin(blacklist_ids)
    images = images[~blacklisted]
    points = points[~points["image_id"].isin(blacklist_ids)]
    points_table = pa.Table.from_pandas(points, schema=POINTS_SCHEMA, preserve_index=False)
    return CuratedTables(
        images=pa.Table.from_pandas(images, schema=IMAGES_SCHEMA, preserve_index=False),
        points=points_table,
        label_counts=count_labels(points_table),
        dropped={
            "blacklisted_images": int(blacklisted.sum()),
            "override_images_not_in_raw": stats.override_images_not_in_raw,
        },
        override_stats=stats,
    )


def curate(
    raw: DatasetVersion, *, image_blacklist: Path, annotation_overrides: Path
) -> CuratedTables:
    """Build the curated tables from ``raw``: overrides first, then the image blacklist.

    Only ``image_id`` of the blacklist parquet is used.

    Raises:
        ValueError: ``raw`` has no points table.
    """
    if raw.points is None:
        raise ValueError("raw version has no points table")
    blacklist = pd.read_parquet(image_blacklist, columns=["image_id"])
    overrides = pd.read_parquet(annotation_overrides)
    return _curate_frames(
        raw.images.to_pandas(),
        raw.points.to_pandas(),
        overrides,
        set(blacklist["image_id"].astype(str)),
    )
