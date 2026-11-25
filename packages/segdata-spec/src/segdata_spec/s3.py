"""S3 object helpers shared by every segdata package: existence, reads, ETags and checksums."""

from __future__ import annotations

import base64
from typing import Any

import awscrt.checksums
from botocore.exceptions import ClientError

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _http_status(exc: ClientError) -> int | None:
    status: int | None = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status


def is_not_found(exc: ClientError) -> bool:
    """Return whether ``exc`` says the object is missing.

    An object is missing for the codes ``404``, ``NoSuchKey`` and ``NotFound``, or for an HTTP 404
    whose code is empty or numeric (a HEAD response has no error body). An explicit code such as
    ``NoSuchBucket`` is not a missing object.
    """
    code = _error_code(exc)
    if code in _NOT_FOUND_CODES:
        return True
    return _http_status(exc) == 404 and (code == "" or code.isdigit())


def is_precondition_failed(exc: ClientError) -> bool:
    """Return whether ``exc`` is a failed conditional request (HTTP 412)."""
    return _error_code(exc) == "PreconditionFailed" or _http_status(exc) == 412


def exists(client: Any, bucket: str, key: str) -> bool:
    """Return whether ``key`` exists in ``bucket``, by HEAD.

    Raises:
        botocore.exceptions.ClientError: Any S3 error other than a missing object.
    """
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if is_not_found(exc):
            return False
        raise
    return True


def get_bytes(client: Any, bucket: str, key: str) -> bytes:
    """Return the whole object's bytes.

    Raises:
        botocore.exceptions.ClientError: The GET failed, including a missing object.
    """
    data: bytes = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return data


def norm_etag(etag: str) -> str:
    """Return ``etag`` without its surrounding quotes, the form manifests record."""
    return etag.strip('"')


def crc64nvme_b64(data: bytes) -> str:
    """Return the CRC64NVME in S3's ``ChecksumCRC64NVME`` form: base64 of the big-endian CRC."""
    return base64.b64encode(awscrt.checksums.crc64nvme(data).to_bytes(8, "big")).decode()
