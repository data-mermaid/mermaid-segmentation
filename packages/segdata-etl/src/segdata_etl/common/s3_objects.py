"""Write-once S3 object helpers: conditional copy and put with idempotent re-runs.

Every write carries ``IfNoneMatch="*"`` so an existing key is never overwritten. A 412 on a key
that already holds identical content counts as done; any other content is a conflict.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError

from segdata_etl.common.jpeg_header import _find_sof_dimensions
from segdata_spec._s3 import is_not_found, is_precondition_failed

_JPEG_SOI = b"\xff\xd8\xff"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_END = 24
_CRC_ALGORITHM = "CRC64NVME"


class ObjectConflict(Exception):
    """The destination key already holds different content than the write would produce."""


class SourceChanged(Exception):
    """The source object's ETag no longer matches the one the copy was planned against."""


@dataclass(frozen=True)
class CopyResult:
    """The destination object a copy produced or found."""

    etag: str
    size: int
    crc64nvme: str
    already_present: bool


@dataclass(frozen=True)
class PutResult:
    """The object a put wrote or found."""

    etag: str
    size: int
    crc64nvme: str
    already_present: bool


def sniff_media_type(head: bytes) -> str | None:
    """Return ``image/jpeg`` or ``image/png`` from the leading bytes, else None."""
    if head.startswith(_JPEG_SOI):
        return "image/jpeg"
    if head.startswith(_PNG_SIGNATURE):
        return "image/png"
    return None


def image_size(head: bytes) -> tuple[int, int] | None:
    """Return ``(width, height)`` from a PNG IHDR or a JPEG SOF in ``head``, else None."""
    if head.startswith(_PNG_SIGNATURE):
        if len(head) < _PNG_IHDR_END or head[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", head[16:_PNG_IHDR_END])
        return width, height
    dims = _find_sof_dimensions(head)
    if dims is None:
        return None
    height, width = dims
    return width, height


def _head_with_checksum(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    """HEAD an object with its checksum, or None when it is absent."""
    try:
        response: dict[str, Any] = client.head_object(
            Bucket=bucket, Key=key, ChecksumMode="ENABLED"
        )
    except ClientError as exc:
        if is_not_found(exc):
            return None
        raise
    return response


def copy_new(
    client: Any,
    *,
    src_bucket: str,
    src_key: str,
    src_etag: str,
    dst_bucket: str,
    dst_key: str,
    expect_size: int,
    expect_crc64nvme: str,
) -> CopyResult:
    """Copy ``src`` to ``dst`` once, guarded by the source ETag and the destination's absence.

    Raises:
        SourceChanged: The source ETag differs from ``src_etag`` (the destination is absent).
        ObjectConflict: The destination exists, or the copy landed, with another size or CRC64NVME.
    """
    try:
        response = client.copy_object(
            Bucket=dst_bucket,
            Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
            IfNoneMatch="*",
            CopySourceIfMatch=src_etag,
            ChecksumAlgorithm=_CRC_ALGORITHM,
        )
    except ClientError as exc:
        if not is_precondition_failed(exc):
            raise
        return _resolve_copy_412(
            client, exc, src_bucket, src_key, dst_bucket, dst_key, expect_size, expect_crc64nvme
        )
    result = response["CopyObjectResult"]
    crc = result.get("ChecksumCRC64NVME", expect_crc64nvme)
    if crc != expect_crc64nvme:
        raise ObjectConflict(f"s3://{dst_bucket}/{dst_key} copied with CRC64NVME {crc}")
    return CopyResult(result["ETag"], expect_size, crc, already_present=False)


def _resolve_copy_412(
    client: Any,
    exc: ClientError,
    src_bucket: str,
    src_key: str,
    dst_bucket: str,
    dst_key: str,
    expect_size: int,
    expect_crc64nvme: str,
) -> CopyResult:
    existing = _head_with_checksum(client, dst_bucket, dst_key)
    if existing is None:
        raise SourceChanged(f"s3://{src_bucket}/{src_key} no longer matches its ETag") from exc
    size = existing["ContentLength"]
    crc = existing.get("ChecksumCRC64NVME")
    if size != expect_size or crc != expect_crc64nvme:
        raise ObjectConflict(
            f"s3://{dst_bucket}/{dst_key} exists with size {size} and CRC64NVME {crc}"
        ) from exc
    return CopyResult(existing["ETag"], size, crc, already_present=True)


def put_new(client: Any, *, bucket: str, key: str, body: bytes, content_type: str) -> PutResult:
    """Write ``body`` once; an existing object with identical bytes counts as already written.

    Raises:
        ObjectConflict: The key exists with different bytes.
    """
    try:
        response = client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            IfNoneMatch="*",
            ChecksumAlgorithm=_CRC_ALGORITHM,
        )
    except ClientError as exc:
        if not is_precondition_failed(exc):
            raise
        return _accept_identical(client, exc, bucket, key, body)
    return PutResult(
        response["ETag"], len(body), response["ChecksumCRC64NVME"], already_present=False
    )


def _accept_identical(
    client: Any, exc: ClientError, bucket: str, key: str, body: bytes
) -> PutResult:
    existing: bytes = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if existing != body:
        raise ObjectConflict(f"s3://{bucket}/{key} already exists with different bytes") from exc
    head = _head_with_checksum(client, bucket, key)
    if head is None:
        raise ObjectConflict(f"s3://{bucket}/{key} vanished while being verified") from exc
    return PutResult(
        head["ETag"], len(body), head.get("ChecksumCRC64NVME", ""), already_present=True
    )
