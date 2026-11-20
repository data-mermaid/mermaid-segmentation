"""Image inspection, profiling and decodability checks.

Detects and categorizes image issues: corrupted data, unsupported formats, invalid channel counts.
Enables graceful skipping and detailed logging. ``find_undecodable`` checks that files decode.
"""

from __future__ import annotations

import io
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from PIL import Image

logger = logging.getLogger(__name__)

# libjpeg decodes at 1/8 scale at best; draft() picks the largest reduction fitting the box.
_DRAFT_SCALE = 8


class IssueType(StrEnum):
    """Categories of image quality issues."""

    VALID = "valid"
    UNSUPPORTED_FORMAT = "unsupported_format"
    INVALID_CHANNELS = "invalid_channels"
    DECODE_FAILURE = "decode_failure"
    TRUNCATED = "truncated"
    ZERO_SIZE = "zero_size"
    OVERSIZED = "oversized"
    UNKNOWN = "unknown"


class ImageInspection(NamedTuple):
    """Result of inspecting a single image."""

    is_valid: bool
    issue_type: IssueType
    format: str | None  # e.g., "JPEG", "PNG"
    channels: int | None  # 1=grayscale, 3=RGB, 4=RGBA
    width: int | None
    height: int | None
    error_message: str | None


def _draft_jpeg(img: Image.Image, box: tuple[int, int]) -> None:
    """Ask the JPEG decoder to decode at the largest reduced scale that fits ``box``; no-op otherwise."""
    if img.format != "JPEG":
        return
    img.draft(img.mode, box)


def _draft_to_max_edge(img: Image.Image, max_edge: int) -> None:
    """Ask the JPEG decoder to decode at a reduced scale that keeps the longest edge >= max_edge.

    The box is scaled by ``max_edge / longest edge`` so wide images reduce along their longest edge.
    """
    width, height = img.size
    longest = max(width, height)
    if longest <= max_edge:
        return
    scale = max_edge / longest
    _draft_jpeg(img, (math.ceil(width * scale), math.ceil(height * scale)))


def inspect_image(image_bytes: io.BytesIO | None) -> ImageInspection:
    """Inspect image bytes for format, channels, and corruption."""
    return decode_image(image_bytes)[1]


def decode_image(
    image_bytes: io.BytesIO | None, *, max_edge: int | None = None
) -> tuple[Image.Image | None, ImageInspection]:
    """Decode image bytes once, inspecting format, channels, and corruption.

    Args:
        image_bytes: Seeked BytesIO with image data
        max_edge: When set, a JPEG whose longest edge exceeds it is decoded at a reduced scale, so
            the returned image can be smaller than the original but is at least ``max_edge`` on its
            longest edge. The inspection always reports the original size.

    Returns:
        ``(image, inspection)``; ``image`` is the loaded image when valid, otherwise None.
    """
    if image_bytes is None or image_bytes.getbuffer().nbytes == 0:
        return None, ImageInspection(
            is_valid=False,
            issue_type=IssueType.ZERO_SIZE,
            format=None,
            channels=None,
            width=None,
            height=None,
            error_message="Empty or missing image data",
        )

    try:
        image_bytes.seek(0)
        img = Image.open(image_bytes)

        format_str = img.format or "UNKNOWN"
        width, height = img.size  # original size, read from the header before any draft
        if max_edge is not None:
            _draft_to_max_edge(img, max_edge)

        # Attempt full decode to catch corrupted data
        img.load()
        num_channels = len(img.getbands())

        # Validate channel count: 1 (grayscale L), 3 (RGB), or 4 (RGBA). Grayscale and RGBA are
        # accepted because the resize step converts them to RGB before JPEG encoding; rejecting
        # them would discard recoverable images.
        if num_channels not in (1, 3, 4):
            return None, ImageInspection(
                is_valid=False,
                issue_type=IssueType.INVALID_CHANNELS,
                format=format_str,
                channels=num_channels,
                width=width,
                height=height,
                error_message=f"Expected 1 (L), 3 (RGB), or 4 (RGBA) channels, got {num_channels}",
            )

        return img, ImageInspection(
            is_valid=True,
            issue_type=IssueType.VALID,
            format=format_str,
            channels=num_channels,
            width=width,
            height=height,
            error_message=None,
        )

    except Image.DecompressionBombError as e:
        return None, ImageInspection(
            is_valid=False,
            issue_type=IssueType.OVERSIZED,
            format=None,
            channels=None,
            width=None,
            height=None,
            error_message=f"{type(e).__name__}: {e}",
        )

    except Image.UnidentifiedImageError as e:
        return None, ImageInspection(
            is_valid=False,
            issue_type=IssueType.UNSUPPORTED_FORMAT,
            format=None,
            channels=None,
            width=None,
            height=None,
            error_message=f"PIL cannot recognize image format: {e}",
        )

    except OSError as e:
        # OSError often indicates truncated/corrupted file
        error_msg = str(e).lower()
        if "truncated" in error_msg or "eof" in error_msg:
            issue = IssueType.TRUNCATED
        else:
            issue = IssueType.DECODE_FAILURE

        return None, ImageInspection(
            is_valid=False,
            issue_type=issue,
            format=None,
            channels=None,
            width=None,
            height=None,
            error_message=f"PIL decode error: {e}",
        )

    except Exception as e:
        return None, ImageInspection(
            is_valid=False,
            issue_type=IssueType.UNKNOWN,
            format=None,
            channels=None,
            width=None,
            height=None,
            error_message=f"{type(e).__name__}: {e}",
        )


def _decode_error(path: Path) -> str | None:
    """Return an error description if ``path`` cannot be decoded, else None."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            _draft_jpeg(img, (max(1, width // _DRAFT_SCALE), max(1, height // _DRAFT_SCALE)))
            img.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def find_undecodable(paths: list[Path], *, workers: int) -> list[tuple[Path, str]]:
    """Decode each path and return ``(path, error)`` for those that fail.

    JPEG validity is checked at reduced scale (up to 1/8 per edge) because proof of
    decodability does not need full pixels, and full-size mosaics decoded by parallel
    workers would multiply peak memory. Other formats are fully decoded.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = list(pool.map(_decode_error, paths))
    return [(path, err) for path, err in zip(paths, errors, strict=True) if err is not None]
