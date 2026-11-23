"""Local staging of uniform-format dataset versions.

``LocalDirS3`` stands in for the S3 client so ``publish_dataset`` writes a version to disk; a later
promote step uploads the staged tree write-once.
"""

from __future__ import annotations

import hashlib
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from segdata_spec._s3 import cache_path, dataset_prefix

_SHA_LENGTH = 7


def _error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


def _as_bytes(body: Any) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes | memoryview):
        return bytes(body)
    raw = body.read()
    return raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)


class LocalDirS3:
    """A directory-backed S3 client: ``<root>/<Bucket>/<Key>`` holds each object."""

    def __init__(self, root: Path):
        """Root the store at ``root``; buckets become its subdirectories."""
        self._root = root

    def _path(self, bucket: str, key: str) -> Path:
        return cache_path(self._root, bucket, key)

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: Any,
        IfNoneMatch: str | None = None,
        ContentType: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Write the object; ``IfNoneMatch="*"`` on an existing key raises a 412."""
        del ContentType  # parity with boto
        path = self._path(Bucket, Key)
        if IfNoneMatch == "*" and path.exists():
            raise _error("PreconditionFailed", 412, "PutObject")
        data = _as_bytes(Body)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"ETag": _etag(data)}

    def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        """Return size and ETag, or raise a 404 for a missing key."""
        path = self._path(Bucket, Key)
        if not path.is_file():
            raise _error("404", 404, "HeadObject")
        data = path.read_bytes()
        return {"ContentLength": len(data), "ETag": _etag(data)}

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        """Return the whole object as a readable body, or raise a 404 for a missing key."""
        path = self._path(Bucket, Key)
        if not path.is_file():
            raise _error("NoSuchKey", 404, "GetObject")
        data = path.read_bytes()
        return {"Body": BytesIO(data), "ContentLength": len(data), "ETag": _etag(data)}


def _etag(data: bytes) -> str:
    return f'"{hashlib.md5(data).hexdigest()}"'  # noqa: S324


def next_version(stage_root: Path, bucket: str, dataset: str, *, today: date, git_sha: str) -> str:
    """Return ``YYYYMMDD-<sha7>``, with ``-2``, ``-3``, ... appended until the staged dir is free."""
    base = f"{today:%Y%m%d}-{git_sha[:_SHA_LENGTH]}"
    candidate, counter = base, 1
    while cache_path(stage_root, bucket, dataset_prefix(dataset, candidate)).exists():
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate
