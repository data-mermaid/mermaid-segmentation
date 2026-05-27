"""Pull JPEG image dimensions from S3 using a ranged GET on the file header.

A baseline JPEG carries width/height inside a Start Of Frame (SOFn) marker near the start of the
file. Reading the first ~64 KB is enough for the overwhelming majority of well-formed JPEGs, so we
avoid downloading multi-megabyte image bodies just to learn ``(width, height)``.

If the SOF marker isn't found in the first chunk we widen the read, then fall back to letting PIL
parse the headers from a ``BytesIO`` (still without decoding pixel data). Corrupt files return
``(None, None)`` with a status of ``"corrupt"``; missing files return ``"not_found"``; S3 errors
return ``"s3_error"``.
"""

from __future__ import annotations

import io
import logging
import struct
from typing import Final

from botocore.client import BaseClient
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# SOF markers that carry width/height: SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15.
# SOF4 (0xC4) is DHT, SOF8 (0xC8) is JPG, SOF12 (0xCC) is DAC — none carry dimensions.
_SOF_BYTES: Final[frozenset[int]] = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _find_sof_dimensions(buf: bytes) -> tuple[int, int] | None:
    """Scan a JPEG byte buffer for an SOF marker; return ``(height, width)`` if found.

    Returns None if no SOF marker is present within ``buf``.
    """
    n = len(buf)
    if n < 4 or buf[0] != 0xFF or buf[1] != 0xD8:
        return None
    i = 2
    while i < n - 1:
        if buf[i] != 0xFF:
            i += 1
            continue
        # Walk past padding fill bytes (0xFF 0xFF ...).
        while i < n - 1 and buf[i + 1] == 0xFF:
            i += 1
        marker = buf[i + 1]
        i += 2
        if marker == 0xD8 or marker == 0xD9:
            # SOI / EOI have no payload.
            continue
        if marker == 0xDA:
            # Start of Scan — past header data; SOF (if any) would have appeared by now.
            return None
        if i + 2 > n:
            return None
        seg_len = struct.unpack(">H", buf[i : i + 2])[0]
        if marker in _SOF_BYTES:
            # Segment: length(2) precision(1) height(2) width(2) ...
            if i + 7 > n:
                return None
            height = struct.unpack(">H", buf[i + 3 : i + 5])[0]
            width = struct.unpack(">H", buf[i + 5 : i + 7])[0]
            return height, width
        i += seg_len
    return None


def _file_size_from_content_range(response: dict) -> int | None:
    """Pull the total object size out of a 206 Partial Content's ``Content-Range``.

    Headers look like ``bytes 0-65535/4359133``; we just take the trailing integer. Falls back to
    ``None`` if the header is missing or malformed (the row's ``file_size`` then ends up null rather
    than wrong).
    """
    cr = response.get("ContentRange") or response.get("contentRange")
    if not cr or "/" not in cr:
        return None
    total = cr.rsplit("/", 1)[-1].strip()
    if total in ("", "*"):
        return None
    try:
        return int(total)
    except ValueError:
        return None


def read_jpeg_size_from_s3(
    client: BaseClient,
    bucket: str,
    key: str,
    *,
    first_chunk: int = 65_536,
    max_chunk: int = 1_048_576,
) -> tuple[int | None, int | None, int | None, str]:
    """Return ``(width, height, file_size, status)`` for a JPEG on S3.

    Status values:
        ``ok``                   — SOF parsed from the first ranged GET.
        ``sof_not_found_extended`` — SOF parsed only after extended read.
        ``pil_fallback``          — PIL parsed the partial body.
        ``corrupt``               — bytes downloaded but no parser could read them.
        ``not_found``             — S3 returned 404.
        ``s3_error``              — other ClientError.

    Only does a ranged GET (no preflight ``head_object``); the total object
    size is read from the 206 response's ``Content-Range`` header. This halves
    the S3 request count compared to a head + get sequence — meaningful at
    1.2M+ images.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{first_chunk - 1}")
        head_bytes = response["Body"].read()
    except ClientError as e:
        code = e.response["Error"].get("Code", "")
        if code in ("404", "NoSuchKey"):
            return None, None, None, "not_found"
        logger.warning("ranged get failed for s3://%s/%s: %s", bucket, key, code)
        return None, None, None, "s3_error"

    file_size = _file_size_from_content_range(response)
    range_end = first_chunk - 1

    dims = _find_sof_dimensions(head_bytes)
    if dims is not None:
        height, width = dims
        return width, height, file_size, "ok"

    # Extended read — pull up to max_chunk in case headers are unusually large.
    if file_size and file_size > len(head_bytes):
        ext_end = min(max_chunk, file_size) - 1
        if ext_end > range_end:
            try:
                response = client.get_object(
                    Bucket=bucket, Key=key, Range=f"bytes={range_end + 1}-{ext_end}"
                )
                extra = response["Body"].read()
            except ClientError as e:
                logger.warning(
                    "extended ranged get failed for s3://%s/%s: %s",
                    bucket,
                    key,
                    e.response["Error"].get("Code", ""),
                )
                return None, None, file_size, "s3_error"
            head_bytes = head_bytes + extra
            dims = _find_sof_dimensions(head_bytes)
            if dims is not None:
                height, width = dims
                return width, height, file_size, "sof_not_found_extended"

    # PIL fallback — never decodes pixels but can recover dimensions from many
    # malformed headers.
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:  # pragma: no cover - pillow is a hard dep
        return None, None, file_size, "corrupt"

    try:
        with Image.open(io.BytesIO(head_bytes)) as img:
            width, height = img.size
        return int(width), int(height), file_size, "pil_fallback"
    except (UnidentifiedImageError, OSError, ValueError):
        return None, None, file_size, "corrupt"
