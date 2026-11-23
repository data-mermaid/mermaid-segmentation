"""Fixtures for the common-helper tests: a recording in-memory S3 client."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from botocore.exceptions import ClientError

from segdata_spec.sync import _crc64nvme_b64


def client_error(code: str, status: int, op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        op,
    )


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


@dataclass
class FakeS3:
    """In-memory S3 client for ``copy_images``, publishing and listing; ``objects`` maps ``(bucket, key)`` to bytes.

    ``copy_calls`` holds the keyword arguments of every ``copy_object``. ``no_crc`` lists objects
    whose HEAD omits ``ChecksumCRC64NVME``, as an object uploaded without a full-object CRC does.
    ``copy_error`` makes every ``copy_object`` raise that error code with status 403.
    ``source_changes_on_copy`` maps a source ``(bucket, key)`` to the bytes it holds by the time
    ``copy_object`` reads it, as a source overwritten between the HEAD and the copy does.
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    no_crc: set[tuple[str, str]] = field(default_factory=set)
    copy_calls: list[dict[str, Any]] = field(default_factory=list)
    copy_error: str | None = None
    source_changes_on_copy: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def _etag(self, bucket: str, key: str) -> str:
        return '"' + hashlib.md5(self.objects[(bucket, key)]).hexdigest() + '"'  # noqa: S324

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
                "ETag": self._etag(Bucket, Key),
                "ChecksumCRC64NVME": _crc64nvme_b64(data),
            }
        }

    def head_object(
        self, *, Bucket: str, Key: str, ChecksumMode: str | None = None
    ) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise client_error("404", 404, "HeadObject")
        response: dict[str, Any] = {
            "ContentLength": len(self.objects[(Bucket, Key)]),
            "ETag": self._etag(Bucket, Key),
        }
        if ChecksumMode == "ENABLED" and (Bucket, Key) not in self.no_crc:
            response["ChecksumCRC64NVME"] = _crc64nvme_b64(self.objects[(Bucket, Key)])
        return response

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise client_error("NoSuchKey", 404, "GetObject")
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, IfNoneMatch: str | None = None, **_: Any
    ) -> dict[str, Any]:
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", 412, "PutObject")
        self.objects[(Bucket, Key)] = bytes(Body)
        return {"ETag": self._etag(Bucket, Key)}

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
                etag = self._etag(bucket, key)
                contents.append(
                    {"Key": key, "ETag": etag, "Size": len(self.objects[(bucket, key)])}
                )
        response: dict[str, Any] = {"KeyCount": len(contents) + len(folders)}
        if contents:
            response["Contents"] = contents
        if folders:
            response["CommonPrefixes"] = [{"Prefix": f} for f in sorted(folders)]
        return response

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return _Paginator(self)


class _Paginator:
    def __init__(self, client: FakeS3):
        self._client = client

    def paginate(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield self._client.list_objects_v2(**kwargs)


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
