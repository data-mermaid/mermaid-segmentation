"""S3 key layout, cache paths and error classification shared by publish, sync and read."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from segdata_spec.resize import ResizeSpec

MANIFEST_NAME = "manifest.json"
EXTRACTOR_SPEC_NAME = "_extractor_spec.json"
FEATURE_INDEX_NAME = "index.parquet"
# The marker namespace of a tables-only sync, beside the resize ids.
TABLES_MARKER_ID = "tables"
# The marker namespace of feature-set syncs, beside the resize ids.
FEATURES_MARKER_ID = "features"
_RESERVED_MARKER_IDS = {TABLES_MARKER_ID, FEATURES_MARKER_ID}
_SEGMENT = re.compile(r"[a-z0-9][a-z0-9._-]*")
_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def dataset_prefix(name: str, version: str) -> str:
    return f"datasets/{name}/{version}/"


def taxonomy_prefix(name: str, version: str) -> str:
    return f"taxonomy/{name}/{version}/"


def _check_segment(value: str, what: str) -> str:
    if not _SEGMENT.fullmatch(value) or ".." in value:
        raise ValueError(f"{what} {value!r} is not a valid key segment ([a-z0-9][a-z0-9._-]*)")
    return value


def feature_store_prefix(store: str, family: str) -> str:
    """Return ``features/<store>/<family>/``, the write-once home of a family's feature files.

    Raises:
        ValueError: ``store`` or ``family`` is not a lowercase key segment.
    """
    return f"features/{_check_segment(store, 'store')}/{_check_segment(family, 'family')}/"


def feature_key(store: str, family: str, image_id: str, suffix: str | None = None) -> str:
    """Return the key of one image's feature file, ``<image_id>[~<suffix>].featurevector``.

    Raises:
        ValueError: ``store``, ``family`` or ``suffix`` is not a lowercase key segment.
    """
    stem = image_id if suffix is None else f"{image_id}~{_check_segment(suffix, 'suffix')}"
    return f"{feature_store_prefix(store, family)}{stem}.featurevector"


def feature_set_prefix(name: str, version: str, family: str) -> str:
    """Return ``datasets/<name>/<version>/features/<family>/``, a version's feature-set prefix.

    Raises:
        ValueError: ``family`` is not a lowercase key segment.
    """
    return f"{dataset_prefix(name, version)}features/{_check_segment(family, 'family')}/"


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


def synced_feature_marker_dir(
    cache_dir: Path, family: str, bucket: str, dataset: str, version: str
) -> Path:
    """Return ``<cache_dir>/synced/features/<family>/<bucket>/<dataset>/<version>``.

    Raises:
        ValueError: ``family`` is not a lowercase key segment.
    """
    root = cache_dir / "synced" / FEATURES_MARKER_ID / _check_segment(family, "family")
    return cache_path(root, bucket, f"{dataset}/{version}")


def synced_resize_ids(cache_dir: Path, bucket: str, dataset: str, version: str) -> list[str]:
    """Return the sorted resize ids that have a sync marker for this dataset version.

    The tables-only and feature-set marker namespaces are not resize ids and are skipped.
    """
    root = cache_dir / "synced"
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.name not in _RESERVED_MARKER_IDS
        and any(synced_marker_dir(cache_dir, child.name, bucket, dataset, version).glob("*.json"))
    )


def is_not_found(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return (
        code in _NOT_FOUND_CODES
        or exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404
    )


def is_precondition_failed(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return (
        code == "PreconditionFailed"
        or exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412
    )
