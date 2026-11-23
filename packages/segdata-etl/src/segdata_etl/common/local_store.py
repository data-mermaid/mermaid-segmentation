"""A filesystem object store with the S3 client surface segdata publish and sync use.

Objects live at ``<root>/<Bucket>/<Key>``, the layout ``segdata_spec.read`` reads as a cache.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from segdata_spec._s3 import cache_path
from segdata_spec.sync import _crc64nvme_b64

_TMP_SUFFIX = ".tmp"
# ``.<name>.<uuid4 hex>.tmp``, the in-flight file ``_write_object`` creates.
_TMP_NAME = re.compile(r"\..+\.[0-9a-f]{32}\.tmp")


def _client_error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


def _body_bytes(body: Any) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes | bytearray | memoryview):
        return bytes(body)
    data = body.read()
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


def _etag(data: bytes) -> str:
    return '"' + hashlib.md5(data, usedforsecurity=False).hexdigest() + '"'


def _write_object(path: Path, data: bytes, *, only_if_absent: bool) -> bool:
    """Write ``data`` through a temp file; return False when ``only_if_absent`` and ``path`` exists.

    The hard link fails atomically on an existing name, so concurrent conditional writers of one
    key cannot both succeed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}{_TMP_SUFFIX}"
    try:
        tmp.write_bytes(data)
        if only_if_absent:
            try:
                os.link(tmp, path)
            except FileExistsError:
                return False
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return True


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, store: LocalStore):
        self._store = store

    def paginate(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield self._store.list_objects_v2(**kwargs)


class LocalStore:
    """S3-shaped client writing to ``<root>/<Bucket>/<Key>``.

    ``source_client`` serves the ``CopySource`` of ``copy_object``.
    """

    def __init__(self, root: Path, source_client: Any = None):
        """Store objects under ``root``; ``source_client`` is required only by ``copy_object``."""
        self._root = root
        self._source_client = source_client

    def _path(self, bucket: str, key: str) -> Path:
        return cache_path(self._root, bucket, key)

    def _read(self, bucket: str, key: str, operation: str) -> bytes:
        path = self._path(bucket, key)
        if not path.is_file():
            raise _client_error("NoSuchKey", 404, operation)
        return path.read_bytes()

    def _store(self, bucket: str, key: str, data: bytes, if_none_match: str | None) -> None:
        only_if_absent = if_none_match == "*"
        if not _write_object(self._path(bucket, key), data, only_if_absent=only_if_absent):
            raise _client_error("PreconditionFailed", 412, "PutObject")

    def put_object(
        self, *, Bucket: str, Key: str, Body: Any, IfNoneMatch: str | None = None, **_: Any
    ) -> dict[str, Any]:
        """Write ``Body``; ``IfNoneMatch="*"`` raises a 412 ClientError on an existing key."""
        data = _body_bytes(Body)
        self._store(Bucket, Key, data, IfNoneMatch)
        return {"ETag": _etag(data)}

    def head_object(
        self, *, Bucket: str, Key: str, ChecksumMode: str | None = None, **_: Any
    ) -> dict[str, Any]:
        """Return size and ETag, plus the CRC64NVME when ``ChecksumMode="ENABLED"``."""
        data = self._read(Bucket, Key, "HeadObject")
        response: dict[str, Any] = {"ContentLength": len(data), "ETag": _etag(data)}
        if ChecksumMode == "ENABLED":
            response["ChecksumCRC64NVME"] = _crc64nvme_b64(data)
        return response

    def get_object(
        self, *, Bucket: str, Key: str, IfMatch: str | None = None, **_: Any
    ) -> dict[str, Any]:
        """Return the object; a non-matching ``IfMatch`` raises a 412 ClientError."""
        data = self._read(Bucket, Key, "GetObject")
        etag = _etag(data)
        if IfMatch is not None and IfMatch.strip('"') != etag.strip('"'):
            raise _client_error("PreconditionFailed", 412, "GetObject")
        return {"Body": _Body(data), "ContentLength": len(data), "ETag": etag}

    def copy_object(
        self,
        *,
        CopySource: dict[str, str],
        Bucket: str,
        Key: str,
        IfNoneMatch: str | None = None,
        ChecksumAlgorithm: str | None = None,
        StorageClass: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Copy ``CopySource`` read through ``source_client``; return S3's ETag and CRC64NVME."""
        if self._source_client is None:
            raise ValueError("copy_object needs a source_client")
        response = self._source_client.get_object(
            Bucket=CopySource["Bucket"], Key=CopySource["Key"]
        )
        data = bytes(response["Body"].read())
        self._store(Bucket, Key, data, IfNoneMatch)
        return {
            "CopyObjectResult": {"ETag": _etag(data), "ChecksumCRC64NVME": _crc64nvme_b64(data)}
        }

    def _keys(self, bucket: str) -> list[str]:
        base = self._path(bucket, "")
        if not base.is_dir():
            return []
        return sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file() and not _TMP_NAME.fullmatch(path.name)
        )

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str = "", Delimiter: str | None = None, **_: Any
    ) -> dict[str, Any]:
        """List every match in one page; ``IsTruncated`` is always False."""
        contents: list[dict[str, Any]] = []
        prefixes: list[str] = []
        for key in self._keys(Bucket):
            if not key.startswith(Prefix):
                continue
            rest = key[len(Prefix) :]
            if Delimiter and Delimiter in rest:
                folder = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                if folder not in prefixes:
                    prefixes.append(folder)
                continue
            data = self._path(Bucket, key).read_bytes()
            contents.append({"Key": key, "Size": len(data), "ETag": _etag(data)})
        response: dict[str, Any] = {
            "Contents": contents,
            "KeyCount": len(contents) + len(prefixes),
            "IsTruncated": False,
        }
        if prefixes:
            response["CommonPrefixes"] = [{"Prefix": p} for p in prefixes]
        return response

    def get_paginator(self, operation: str) -> _Paginator:
        """Return a paginator over ``list_objects_v2``, the only paginated operation."""
        if operation != "list_objects_v2":
            raise ValueError(f"LocalStore paginates only list_objects_v2, not {operation!r}")
        return _Paginator(self)
