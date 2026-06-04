"""Image inspection and profiling for data quality analysis.

Detects and categorizes image issues: corrupted data, unsupported formats, invalid channel counts.
Enables graceful skipping and detailed logging.
"""

from __future__ import annotations

import io
import logging
from enum import StrEnum
from typing import Any, NamedTuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

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


def inspect_image(image_bytes: io.BytesIO) -> ImageInspection:
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


def inspect_source_images(
    df_images: pd.DataFrame,
    bucket: str,
    source_id: int,
    s3_client: Any,
    workers: int = 8,
) -> pd.DataFrame:
    """Profile all images in a source for quality issues.

    Args:
        df_images: Images parquet with [source_id, image_id, width, height, needs_resize]
        bucket: S3 bucket name
        source_id: Source ID to profile
        s3_client: Boto3 S3 client
        workers: Concurrent workers for S3 downloads

    Returns:
        DataFrame with per-image inspection results + aggregate stats
        Columns: image_id, is_valid, issue_type, format, channels, width, height, error_message
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from mermaidseg.datasets.utils import get_image_s3

    df_source = df_images[df_images["source_id"] == source_id].copy()

    if len(df_source) == 0:
        logger.warning("No images found for source_id=%s", source_id)
        return pd.DataFrame(
            columns=[
                "image_id",
                "is_valid",
                "issue_type",
                "format",
                "channels",
                "width",
                "height",
                "error_message",
            ]
        )

    results = []

    def inspect_one(row: pd.Series) -> dict[str, Any]:
        image_id = str(row["image_id"])
        s3_key = f"etl-outputs/coralnet/s{source_id}/images/{image_id}.jpg"

        try:
            image_bytes = get_image_s3(s3=s3_client, bucket=bucket, key=s3_key)
            inspection = inspect_image(image_bytes)
        except Exception as e:
            logger.warning("Failed to download %s/%s: %s", source_id, image_id, e)
            inspection = ImageInspection(
                is_valid=False,
                issue_type=IssueType.UNKNOWN,
                format=None,
                channels=None,
                width=None,
                height=None,
                error_message=f"Download failed: {type(e).__name__}",
            )

        return {
            "image_id": image_id,
            "is_valid": inspection.is_valid,
            "issue_type": inspection.issue_type.value,
            "format": inspection.format,
            "channels": inspection.channels,
            "width": inspection.width,
            "height": inspection.height,
            "error_message": inspection.error_message,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(inspect_one, row): idx for idx, row in df_source.iterrows()}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Inspecting source {source_id}"
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error("Inspection failed: %s", e)

    df_results = pd.DataFrame(results) if results else pd.DataFrame()

    # Log summary stats
    if len(df_results) > 0:
        valid_count = df_results["is_valid"].sum()
        corrupted_count = len(df_results) - valid_count
        logger.info(
            "Source %s inspection: %d valid, %d corrupted (%.1f%%)",
            source_id,
            valid_count,
            corrupted_count,
            (corrupted_count / len(df_results)) * 100 if len(df_results) > 0 else 0,
        )

        # Log breakdown by issue type
        issue_counts = df_results[~df_results["is_valid"]]["issue_type"].value_counts()
        for issue_type, count in issue_counts.items():
            logger.info("  %s: %d images", issue_type, count)

    return df_results
