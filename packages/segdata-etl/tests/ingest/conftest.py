"""Shared fixtures for the external-dataset tests.

Provides a hand-rolled in-memory ``FakeS3`` that implements the boto3 client surface the package
touches, which keeps the suite free of network and AWS dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def _client_error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


@dataclass
class FakeS3:
    """In-memory S3 client surface; ``objects`` maps ``(bucket, key)`` to bytes."""

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.objects[(bucket, key)] = data

    def put_object(self, *, Bucket: str, Key: str, Body, ContentType: str | None = None):
        del ContentType  # parity with boto
        if isinstance(Body, str):
            raw: bytes = Body.encode("utf-8")
        elif isinstance(Body, bytes | memoryview):
            raw = bytes(Body)
        else:
            raw = Body.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
        self.objects[(Bucket, Key)] = raw

    def head_object(self, *, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        with open(Filename, "rb") as f:
            self.objects[(Bucket, Key)] = f.read()

    def get_object(self, *, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", "GetObject")
        data = self.objects[(Bucket, Key)]
        return {"Body": _Body(data), "ContentLength": len(data)}

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", Delimiter: str = ""):
        """Mirror S3: ``Contents`` and ``CommonPrefixes`` are absent when empty."""
        keys = [k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix)]
        common_prefixes: set[str] = set()
        contents = []
        for k in keys:
            rest = k[len(Prefix) :]
            if Delimiter and Delimiter in rest:
                common_prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
            else:
                contents.append({"Key": k})
        response: dict = {"KeyCount": len(common_prefixes) + len(contents)}
        if common_prefixes:
            response["CommonPrefixes"] = [{"Prefix": p} for p in sorted(common_prefixes)]
        if contents:
            response["Contents"] = contents
        return response

    def get_paginator(self, name: str):
        if name != "list_objects_v2":
            raise NotImplementedError(name)
        return _Paginator(self)


class _Paginator:
    def __init__(self, client: FakeS3):
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str = "", Delimiter: str | None = None):
        delim = Delimiter if Delimiter is not None else ""
        yield self._client.list_objects_v2(Bucket=Bucket, Prefix=Prefix, Delimiter=delim)


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
