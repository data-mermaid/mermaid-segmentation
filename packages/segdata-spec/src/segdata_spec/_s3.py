"""S3 key layout, cache paths and error classification shared by publish, sync and read."""

from __future__ import annotations

from pathlib import Path

from botocore.exceptions import ClientError

MANIFEST_NAME = "manifest.json"
_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def dataset_prefix(name: str, version: str) -> str:
    return f"datasets/{name}/{version}/"


def taxonomy_prefix(name: str, version: str) -> str:
    return f"taxonomy/{name}/{version}/"


def cache_path(cache_dir: Path, bucket: str, key: str) -> Path:
    """Return ``<cache_dir>/<bucket>/<key>``, refusing a bucket or key that would leave the cache."""
    if bucket in ("", ".", "..") or "/" in bucket:
        raise ValueError(f"refusing S3 bucket outside the cache layout: {bucket!r}")
    if key.startswith("/") or ".." in key.split("/"):
        raise ValueError(f"refusing S3 key outside the cache layout: {key!r}")
    return cache_dir / bucket / key


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
