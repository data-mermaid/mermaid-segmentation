"""Media-type detection from an image file's leading bytes."""

from __future__ import annotations

from typing import Literal

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Bytes a caller must read for ``sniff_media_type`` to recognise every supported type.
SNIFF_BYTES = len(_PNG_MAGIC)
# File extension, without the dot, of each media type ``sniff_media_type`` returns.
EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png"}


def sniff_media_type(head: bytes) -> Literal["image/jpeg", "image/png"]:
    """Return the media type whose magic number starts ``head``.

    Raises:
        ValueError: ``head`` starts with neither the JPEG nor the PNG magic number.
    """
    if head.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if head.startswith(_PNG_MAGIC):
        return "image/png"
    raise ValueError(f"not a JPEG or PNG: leading bytes {head[:SNIFF_BYTES]!r}")
