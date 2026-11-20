"""Shared fixtures for the segdata-spec tests.

Provides a hand-rolled in-memory ``FakeS3`` that implements the boto3 client surface the package
touches, including S3 conditional writes and reads, ETags and an ordered call log.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest
from botocore.exceptions import ClientError


@pytest.fixture(autouse=True)
def _fake_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give boto3 static credentials so it never queries the EC2 metadata endpoint."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


def _client_error(code: str, status: int, op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        op,
    )


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _as_bytes(body: Any) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes | memoryview):
        return bytes(body)
    raw = body.read()
    return raw.encode("utf-8") if isinstance(raw, str) else raw


@dataclass
class FakeS3:
    """In-memory S3 client surface; ``objects`` maps ``(bucket, key)`` to bytes.

    ``calls`` logs every ``put_object`` and ``get_object`` as ``(op, bucket, key, kwargs)`` where
    ``kwargs`` holds the call's other keyword arguments (``IfNoneMatch``, ``IfMatch``, ...).
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    etag_overrides: dict[tuple[str, str], str] = field(default_factory=dict)
    calls: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)

    def put(self, bucket: str, key: str, data: bytes) -> None:
        """Seed an object without logging a call."""
        self.objects[(bucket, key)] = data
        self.etag_overrides.pop((bucket, key), None)

    def set_etag(self, bucket: str, key: str, etag: str) -> None:
        """Give an existing object an explicit ETag, e.g. a multipart-style ``abc-2``."""
        assert (bucket, key) in self.objects
        self.etag_overrides[(bucket, key)] = etag.strip('"')

    def _etag(self, bucket: str, key: str) -> str:
        bare = (
            self.etag_overrides.get((bucket, key))
            or hashlib.md5(  # noqa: S324
                self.objects[(bucket, key)]
            ).hexdigest()
        )
        return f'"{bare}"'

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put", Bucket, Key, dict(kwargs)))
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise _client_error("PreconditionFailed", 412, "PutObject")
        self.put(Bucket, Key, _as_bytes(Body))
        return {"ETag": self._etag(Bucket, Key)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", 404, "HeadObject")
        return {
            "ContentLength": len(self.objects[(Bucket, Key)]),
            "ETag": self._etag(Bucket, Key),
        }

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", Bucket, Key, dict(kwargs)))
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", 404, "GetObject")
        etag = self._etag(Bucket, Key)
        if_match = kwargs.get("IfMatch")
        if if_match is not None and if_match.strip('"') != etag.strip('"'):
            raise _client_error("PreconditionFailed", 412, "GetObject")
        data = self.objects[(Bucket, Key)]
        return {"Body": _Body(data), "ContentLength": len(data), "ETag": etag}


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
