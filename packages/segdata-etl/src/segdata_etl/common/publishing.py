"""Publish helpers shared by every versioned-dataset builder."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa

from segdata_etl.common.version import list_versions, next_version
from segdata_spec.errors import PublishRefused
from segdata_spec.manifest import ImageStore
from segdata_spec.publish import GitState, _guard_git, probe_git
from segdata_spec.schema import LABEL_COUNTS_SCHEMA

_IMAGES_FOLDER = "images"


def checked_git_state(repo_root: Path) -> GitState:
    """Probe git once and refuse a dirty, unpushed or lock-less tree, as ``publish_dataset`` does.

    Raises:
        PublishRefused: The tree cannot be published from.
    """
    state, _lock = _guard_git(repo_root, probe_git(repo_root))
    return state


def pick_version(client: Any, bucket: str, dataset: str, *, git_sha: str, today: date) -> str:
    """Return the next free version name under ``datasets/<dataset>/``."""
    existing = list_versions(client, bucket, f"datasets/{dataset}/")
    return next_version(existing, git_sha=git_sha, today=today)


def require_version_absent(client: Any, bucket: str, dataset: str, version: str) -> None:
    """Refuse a version folder that already holds any object.

    Raises:
        PublishRefused: ``datasets/<dataset>/<version>/`` is not empty.
    """
    prefix = f"datasets/{dataset}/{version}/"
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        if page.get("Contents"):
            raise PublishRefused(f"s3://{bucket}/{prefix} already holds objects")


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


def image_prefix(dataset: str) -> str:
    """Return the destination key prefix of ``dataset``'s images."""
    return f"{_IMAGES_FOLDER}/{dataset}/"


def image_store(bucket: str, dataset: str) -> ImageStore:
    """Return the segdata-owned image store of ``dataset`` in ``bucket``."""
    return ImageStore(bucket=bucket, prefix=image_prefix(dataset), owner="segdata")
