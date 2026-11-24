"""Local staging of uniform-format dataset versions.

``LocalDirS3`` stands in for the S3 client so ``publish_dataset`` writes a version to disk; a later
promote step uploads the staged tree write-once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from segdata_etl.common.s3_objects import crc64nvme_b64
from segdata_spec._s3 import cache_path, dataset_prefix
from segdata_spec.publish import GitState, PublishRefused

DEFAULT_CACHE_DIR = Path("outputs") / "segdata-cache"

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
        return _checksums(data)

    def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        """Return size and ETag, or raise a 404 for a missing key."""
        path = self._path(Bucket, Key)
        if not path.is_file():
            raise _error("404", 404, "HeadObject")
        data = path.read_bytes()
        return {"ContentLength": len(data), **_checksums(data)}

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        """Return the whole object as a readable body, or raise a 404 for a missing key."""
        path = self._path(Bucket, Key)
        if not path.is_file():
            raise _error("NoSuchKey", 404, "GetObject")
        data = path.read_bytes()
        return {"Body": BytesIO(data), "ContentLength": len(data), "ETag": _etag(data)}


def _etag(data: bytes) -> str:
    return f'"{hashlib.md5(data).hexdigest()}"'  # noqa: S324


def _checksums(data: bytes) -> dict[str, str]:
    return {
        "ETag": _etag(data),
        "ChecksumCRC64NVME": crc64nvme_b64(data),
    }


def refuse_unpublishable(git: GitState) -> None:
    """Raise ``PublishRefused`` for a dirty or unpushed tree, with the messages ``publish`` uses.

    Raises:
        PublishRefused: The tree has uncommitted tracked changes, or HEAD is on no remote branch.
    """
    if git.dirty:
        raise PublishRefused("working tree has uncommitted changes to tracked files")
    if not git.pushed:
        raise PublishRefused(f"HEAD {git.sha} is not contained in any remote branch")


def version_taken(client: Any, bucket: str, prefix: str) -> bool:
    """Return whether any object exists under the version-folder ``prefix`` in ``bucket``."""
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(response.get("Contents"))


def next_free_version(
    stage_root: Path | None,
    bucket: str,
    prefix_for: Callable[[str], str],
    *,
    git_sha: str,
    today: date | None = None,
    client: Any | None = None,
) -> str:
    """Return ``YYYYMMDD-<sha7>``, with ``-2``, ``-3``, ... appended until the version is free.

    ``prefix_for(version)`` is the version's folder prefix, e.g. ``datasets/<ds>/<version>/``. A version is
    taken when its staged dir exists under ``stage_root`` (skipped when ``None``) or, given ``client``, when
    any object exists under its prefix in ``bucket``, which includes a half-promoted version. ``today``
    defaults to the current UTC date.
    """
    today = today or datetime.now(UTC).date()
    base = f"{today:%Y%m%d}-{git_sha[:_SHA_LENGTH]}"

    def taken(candidate: str) -> bool:
        prefix = prefix_for(candidate)
        if stage_root is not None and cache_path(stage_root, bucket, prefix).exists():
            return True
        return client is not None and version_taken(client, bucket, prefix)

    candidate, counter = base, 1
    while taken(candidate):
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def next_version(
    stage_root: Path,
    bucket: str,
    dataset: str,
    *,
    git_sha: str,
    today: date | None = None,
    client: Any | None = None,
) -> str:
    """Return the first free version of ``datasets/<dataset>/``; see ``next_free_version``."""
    return next_free_version(
        stage_root,
        bucket,
        partial(dataset_prefix, dataset),
        git_sha=git_sha,
        today=today,
        client=client,
    )
