"""Image-byte builders for the uniform-format dataset helper tests; the S3 fake is ``s3_fake``."""

from __future__ import annotations

import io

from PIL import Image


def jpeg_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 240)).save(buf, format="PNG")
    return buf.getvalue()
