"""Shared fixtures for the uniform-format dataset helpers.

Provides a hand-rolled in-memory ``FakeS3`` that implements the boto3 client surface the staging
and copy helpers touch: conditional ``put_object`` and ``copy_object``, ``head_object`` with
checksums, ranged ``get_object`` and an ordered call log. Errors carry the codes real S3 raises.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import awscrt.checksums
import pytest
from botocore.exceptions import ClientError
from PIL import Image


@pytest.fixture(autouse=True)
def _fake_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give boto3 static credentials so it never queries the EC2 metadata endpoint."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


def crc64nvme_b64(data: bytes) -> str:
    """Return the CRC64NVME in S3's ``ChecksumCRC64NVME`` form: base64 of the big-endian CRC."""
    return base64.b64encode(awscrt.checksums.crc64nvme(data).to_bytes(8, "big")).decode()


def md5_etag(data: bytes) -> str:
    return f'"{hashlib.md5(data).hexdigest()}"'  # noqa: S324


def jpeg_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 240)).save(buf, format="PNG")
    return buf.getvalue()


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

    ``calls`` logs every write and read as ``(op, bucket, key, kwargs)``. Objects written through
    ``put_object`` or ``copy_object`` carry a CRC64NVME only when the call asked for the
    ``CRC64NVME`` algorithm, as S3 does; ``put`` seeds an object with its CRC and ETag.
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    crcs: dict[tuple[str, str], str] = field(default_factory=dict)
    etags: dict[tuple[str, str], str] = field(default_factory=dict)
    calls: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)

    def put(
        self, bucket: str, key: str, data: bytes, *, etag: str | None = None, crc: bool = True
    ) -> None:
        """Seed an object without logging a call."""
        self.objects[(bucket, key)] = data
        self.etags[(bucket, key)] = etag or md5_etag(data)
        if crc:
            self.crcs[(bucket, key)] = crc64nvme_b64(data)
        else:
            self.crcs.pop((bucket, key), None)

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put", Bucket, Key, dict(kwargs)))
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", 412, "PutObject")
        data = _as_bytes(Body)
        self.put(Bucket, Key, data, crc=kwargs.get("ChecksumAlgorithm") == "CRC64NVME")
        response: dict[str, Any] = {"ETag": self.etags[(Bucket, Key)]}
        if (Bucket, Key) in self.crcs:
            response["ChecksumCRC64NVME"] = self.crcs[(Bucket, Key)]
        return response

    def copy_object(
        self, *, Bucket: str, Key: str, CopySource: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append(("copy", Bucket, Key, {"CopySource": CopySource, **kwargs}))
        src = (CopySource["Bucket"], CopySource["Key"])
        if src not in self.objects:
            raise client_error("NoSuchKey", 404, "CopyObject")
        wanted = kwargs.get("CopySourceIfMatch")
        # S3 accepts the ETag with or without its quotes.
        if wanted is not None and wanted.strip('"') != self.etags[src].strip('"'):
            raise client_error("PreconditionFailed", 412, "CopyObject")
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", 412, "CopyObject")
        self.objects[(Bucket, Key)] = self.objects[src]
        # A single CopyObject writes a single-part object: its ETag is the MD5 of the bytes, so a
        # multipart source ETag is not carried over.
        self.etags[(Bucket, Key)] = md5_etag(self.objects[src])
        if kwargs.get("ChecksumAlgorithm") == "CRC64NVME":
            self.crcs[(Bucket, Key)] = crc64nvme_b64(self.objects[src])
        else:
            self.crcs.pop((Bucket, Key), None)
        result: dict[str, Any] = {"ETag": self.etags[(Bucket, Key)]}
        if (Bucket, Key) in self.crcs:
            result["ChecksumCRC64NVME"] = self.crcs[(Bucket, Key)]
        return {"CopyObjectResult": result}

    def head_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head", Bucket, Key, dict(kwargs)))
        if (Bucket, Key) not in self.objects:
            raise client_error("404", 404, "HeadObject")
        response: dict[str, Any] = {
            "ContentLength": len(self.objects[(Bucket, Key)]),
            "ETag": self.etags[(Bucket, Key)],
            "LastModified": datetime(2026, 8, 12, 6, 30, tzinfo=UTC),
        }
        if kwargs.get("ChecksumMode") == "ENABLED" and (Bucket, Key) in self.crcs:
            response["ChecksumCRC64NVME"] = self.crcs[(Bucket, Key)]
        return response

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", Bucket, Key, dict(kwargs)))
        if (Bucket, Key) not in self.objects:
            raise client_error("NoSuchKey", 404, "GetObject")
        data = self.objects[(Bucket, Key)]
        if kwargs.get("IfMatch") not in (None, self.etags[(Bucket, Key)]):
            raise client_error("PreconditionFailed", 412, "GetObject")
        response: dict[str, Any] = {"ETag": self.etags[(Bucket, Key)]}
        range_header = kwargs.get("Range")
        if range_header is None:
            return {**response, "Body": _Body(data), "ContentLength": len(data)}
        first, last = range_header.removeprefix("bytes=").split("-")
        start, end = int(first), min(int(last), len(data) - 1)
        chunk = data[start : end + 1]
        return {
            **response,
            "Body": _Body(chunk),
            "ContentLength": len(chunk),
            "ContentRange": f"bytes {start}-{end}/{len(data)}",
        }

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str = "", MaxKeys: int = 1000, **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append(("list", Bucket, Prefix, {"MaxKeys": MaxKeys, **kwargs}))
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))[:MaxKeys]
        response: dict[str, Any] = {"KeyCount": len(keys)}
        if keys:
            response["Contents"] = [{"Key": k} for k in keys]
        return response

    def ops(self, op: str) -> list[tuple[str, str, str, dict[str, Any]]]:
        return [call for call in self.calls if call[0] == op]


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
