"""Image inspection and profiling for data quality analysis.

Detects and categorizes image issues: corrupted data, unsupported formats, invalid channel counts.
Enables graceful skipping and detailed logging.
"""

from __future__ import annotations

import io
import logging
from enum import StrEnum
from typing import NamedTuple

from PIL import Image

logger = logging.getLogger(__name__)


class IssueType(StrEnum):
    """Categories of image quality issues."""

    VALID = "valid"
    UNSUPPORTED_FORMAT = "unsupported_format"
    INVALID_CHANNELS = "invalid_channels"
    CORRUPTED_HEADER = "corrupted_header"
    DECODE_FAILURE = "decode_failure"
    TRUNCATED = "truncated"
    ZERO_SIZE = "zero_size"
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


def inspect_image(image_bytes: io.BytesIO | None) -> ImageInspection:
    """Inspect image bytes for format, channels, and corruption.

    Args:
        image_bytes: Seeked BytesIO with image data

    Returns:
        ImageInspection with is_valid flag and detailed issue info
    """
    if image_bytes is None or image_bytes.getbuffer().nbytes == 0:
        return ImageInspection(
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

        # Attempt full decode to catch corrupted data
        img.load()

        format_str = img.format or "UNKNOWN"
        width, height = img.size
        num_channels = len(img.getbands())

        # Validate channel count: 1 (grayscale L), 3 (RGB), or 4 (RGBA). Grayscale and RGBA are
        # accepted because the resize step converts them to RGB before JPEG encoding; rejecting
        # them would discard recoverable images.
        if num_channels not in (1, 3, 4):
            return ImageInspection(
                is_valid=False,
                issue_type=IssueType.INVALID_CHANNELS,
                format=format_str,
                channels=num_channels,
                width=width,
                height=height,
                error_message=f"Expected 1 (L), 3 (RGB), or 4 (RGBA) channels, got {num_channels}",
            )

        return ImageInspection(
            is_valid=True,
            issue_type=IssueType.VALID,
            format=format_str,
            channels=num_channels,
            width=width,
            height=height,
            error_message=None,
        )

    except Image.UnidentifiedImageError as e:
        return ImageInspection(
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
        elif "bad" in error_msg or "invalid" in error_msg:
            issue = IssueType.CORRUPTED_HEADER
        else:
            issue = IssueType.DECODE_FAILURE

        return ImageInspection(
            is_valid=False,
            issue_type=issue,
            format=None,
            channels=None,
            width=None,
            height=None,
            error_message=f"PIL decode error: {e}",
        )

    except Exception as e:
        return ImageInspection(
            is_valid=False,
            issue_type=IssueType.UNKNOWN,
            format=None,
            channels=None,
            width=None,
            height=None,
            error_message=f"{type(e).__name__}: {e}",
        )
