"""In-memory S3 client for the segdata-etl tests, and the ETag and checksum values S3 reports.

Imported as ``s3_fake``: pytest's default import mode puts ``tests/`` on ``sys.path``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from botocore.exceptions import ClientError

from segdata_spec.sync import _crc64nvme_b64


def client_error(code: str, status: int, op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        op,
    )


def md5_hex(data: bytes) -> str:
    """The unquoted MD5 hex digest of ``data``, the ETag of a single-part upload."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def quoted_etag(data: bytes, *, multipart: bool = False) -> str:
    """The quoted ETag S3 gives ``data``: its MD5, suffixed ``-2`` for a two-part upload."""
    suffix = "-2" if multipart else ""
    return f'"{md5_hex(data)}{suffix}"'


def crc64nvme(data: bytes) -> str:
    """The base64 full-object CRC64NVME S3 reports for ``data``."""
    return _crc64nvme_b64(data)


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _as_bytes(body: Any) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes | bytearray | memoryview):
        return bytes(body)
    raw = body.read()
    return raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)


@dataclass
class FakeS3:
    """In-memory S3 client; ``objects`` maps ``(bucket, key)`` to bytes.

    ``Range`` on ``get_object`` follows ``bytes=start-end`` (inclusive). ``full_reads`` records
    the key of every ``get_object`` without a ``Range``. ``multipart`` holds the objects whose
    ETag is a two-part upload's. ``no_crc`` lists objects whose HEAD omits
    ``ChecksumCRC64NVME``, as an object uploaded without a full-object CRC does.
    ``copy_calls`` holds the keyword arguments of every ``copy_object``; ``copy_error`` makes
    every ``copy_object`` raise that error code with status 403. ``source_changes_on_copy`` maps
    a source ``(bucket, key)`` to the bytes it holds by the time ``copy_object`` reads it, as a
    source overwritten between the HEAD and the copy does.
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    full_reads: list[str] = field(default_factory=list)
    multipart: set[tuple[str, str]] = field(default_factory=set)
    no_crc: set[tuple[str, str]] = field(default_factory=set)
    copy_calls: list[dict[str, Any]] = field(default_factory=list)
    copy_error: str | None = None
    source_changes_on_copy: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def etag(self, bucket: str, key: str) -> str:
        """The quoted ETag S3 gives the object."""
        return quoted_etag(self.objects[(bucket, key)], multipart=(bucket, key) in self.multipart)

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.objects[(bucket, key)] = data

    def put_object(
        self, *, Bucket: str, Key: str, Body: Any, IfNoneMatch: str | None = None, **_: Any
    ) -> dict[str, Any]:
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", 412, "PutObject")
        self.objects[(Bucket, Key)] = _as_bytes(Body)
        return {"ETag": self.etag(Bucket, Key)}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        with open(Filename, "rb") as f:
            self.objects[(Bucket, Key)] = f.read()

    def copy_object(
        self, *, CopySource: dict[str, str], Bucket: str, Key: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.copy_calls.append({"CopySource": CopySource, "Bucket": Bucket, "Key": Key, **kwargs})
        if self.copy_error is not None:
            raise client_error(self.copy_error, 403, "CopyObject")
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", 412, "CopyObject")
        source = (CopySource["Bucket"], CopySource["Key"])
        if source in self.source_changes_on_copy:
            self.objects[source] = self.source_changes_on_copy[source]
        data = self.objects[source]
        self.objects[(Bucket, Key)] = data
        return {
            "CopyObjectResult": {
                "ETag": self.etag(Bucket, Key),
                "ChecksumCRC64NVME": crc64nvme(data),
            }
        }

    def head_object(
        self, *, Bucket: str, Key: str, ChecksumMode: str | None = None
    ) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise client_error("404", 404, "HeadObject")
        data = self.objects[(Bucket, Key)]
        response: dict[str, Any] = {"ContentLength": len(data), "ETag": self.etag(Bucket, Key)}
        if ChecksumMode == "ENABLED" and (Bucket, Key) not in self.no_crc:
            response["ChecksumCRC64NVME"] = crc64nvme(data)
        return response

    def get_object(
        self, *, Bucket: str, Key: str, Range: str | None = None, **_: Any
    ) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise client_error("NoSuchKey", 404, "GetObject")
        full = self.objects[(Bucket, Key)]
        if not Range:
            self.full_reads.append(Key)
            return {"Body": _Body(full), "ContentLength": len(full)}
        assert Range.startswith("bytes=")
        start_str, end_str = Range[len("bytes=") :].split("-")
        start = int(start_str)
        end = int(end_str) if end_str else len(full) - 1
        data = full[start : end + 1]
        return {
            "Body": _Body(data),
            "ContentLength": len(data),
            "ContentRange": f"bytes {start}-{end}/{len(full)}",
        }

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str = "", Delimiter: str | None = None
    ) -> dict[str, Any]:
        """One page; ``Contents`` and ``CommonPrefixes`` are absent when empty, as in S3."""
        contents: list[dict[str, Any]] = []
        folders: set[str] = set()
        for bucket, key in sorted(self.objects):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            rest = key[len(Prefix) :]
            if Delimiter and Delimiter in rest:
                folders.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
            else:
                size = len(self.objects[(bucket, key)])
                contents.append({"Key": key, "ETag": self.etag(bucket, key), "Size": size})
        response: dict[str, Any] = {"KeyCount": len(contents) + len(folders)}
        if contents:
            response["Contents"] = contents
        if folders:
            response["CommonPrefixes"] = [{"Prefix": f} for f in sorted(folders)]
        return response

    def get_paginator(self, operation: str) -> _Paginator:
        if operation != "list_objects_v2":
            raise NotImplementedError(operation)
        return _Paginator(self)


class _Paginator:
    def __init__(self, client: FakeS3):
        self._client = client

    def paginate(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield self._client.list_objects_v2(**kwargs)
