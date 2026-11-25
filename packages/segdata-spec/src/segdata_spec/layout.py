"""S3 key layout and local cache paths shared by publish, sync and read."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from segdata_spec.resize import ResizeSpec

MANIFEST_NAME = "manifest.json"


def dataset_prefix(name: str, version: str) -> str:
    """Return the key prefix of a dataset version, ending in ``/``."""
    return f"datasets/{name}/{version}/"


def taxonomy_prefix(name: str, version: str) -> str:
    """Return the key prefix of a taxonomy version, ending in ``/``."""
    return f"taxonomy/{name}/{version}/"


def cache_path(cache_dir: Path, bucket: str, key: str) -> Path:
    """Return ``<cache_dir>/<bucket>/<key>``, refusing a bucket or key that would leave the cache."""
    if bucket in ("", ".", "..") or "/" in bucket:
        raise ValueError(f"refusing S3 bucket outside the cache layout: {bucket!r}")
    if key.startswith("/") or ".." in key.split("/"):
        raise ValueError(f"refusing S3 key outside the cache layout: {key!r}")
    return cache_dir / bucket / key


def resized_cache_path(cache_dir: Path, spec: ResizeSpec, bucket: str, key: str) -> Path:
    """Return ``<cache_dir>/resized/<spec.id>/<bucket>/<key>``, refusing keys that escape it."""
    return cache_path(cache_dir / "resized" / spec.id, bucket, key)


def resized_index_path(cache_dir: Path, spec: ResizeSpec, bucket: str, key: str) -> Path:
    """Return ``<cache_dir>/resized/<spec.id>/.index/<bucket>/<key>.json``, the output's sidecar.

    S3 bucket names cannot start with ``.``, so ``.index`` never collides with a bucket folder.
    """
    path = cache_path(cache_dir / "resized" / spec.id / ".index", bucket, key)
    return path.with_name(path.name + ".json")


def resize_id(spec: ResizeSpec | None) -> str:
    """Return ``spec.id``, or ``original`` for images cached as is."""
    return "original" if spec is None else spec.id


def synced_marker_dir(cache_dir: Path, rid: str, bucket: str, dataset: str, version: str) -> Path:
    """Return ``<cache_dir>/synced/<rid>/<bucket>/<dataset>/<version>``, the sync marker directory."""
    return cache_path(cache_dir / "synced" / rid, bucket, f"{dataset}/{version}")


def synced_resize_ids(cache_dir: Path, bucket: str, dataset: str, version: str) -> list[str]:
    """Return the sorted resize ids that have a sync marker for this dataset version."""
    root = cache_dir / "synced"
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if any(synced_marker_dir(cache_dir, child.name, bucket, dataset, version).glob("*.json"))
    )
