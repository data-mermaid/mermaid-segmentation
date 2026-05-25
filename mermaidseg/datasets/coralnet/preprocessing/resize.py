"""Phase 1 (scan) and Phase 2 (resize) image preprocessing pipeline."""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


CHECKPOINT_SCHEMA = {
    "source_id": "int32",
    "image_id": "string",
    "status": "string",
    "resize_timestamp": "object",  # datetime
    "error_message": "string",
}


def resize_image_to_threshold(
    image_bytes: io.BytesIO,
    threshold: int = 2048,
) -> io.BytesIO:
    """Resize image so longest edge = threshold, maintaining aspect ratio.

    If longest edge <= threshold, returns original bytes unchanged.

    Args:
        image_bytes: Seeked BytesIO with JPEG image data
        threshold: Target longest edge (pixels)

    Returns:
        BytesIO with resized JPEG (or original if no resize needed)
    """
    img = Image.open(image_bytes)
    img.load()

    width, height = img.size
    longest_edge = max(width, height)

    # No resize needed
    if longest_edge <= threshold:
        image_bytes.seek(0)
        return image_bytes

    # Calculate new dimensions
    scale = threshold / longest_edge
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

    # Save to BytesIO
    output = io.BytesIO()
    img_resized.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output


def _s3_key_for(prefix: str, source_id: int, image_id: str) -> str:
    """S3 key for a CoralNet original image."""
    return f"{prefix}/s{source_id}/images/{image_id}.jpg"


def _resized_s3_key_for(prefix: str, source_id: int, image_id: str, threshold: int) -> str:
    """S3 key for a resized image."""
    return f"{prefix}/resized/{threshold}/s{source_id}/images/{image_id}.jpg"


def _check_resized_exists(
    bucket: str,
    output_prefix: str,
    source_id: int,
    image_id: str,
    threshold: int,
    s3_client: Any,
) -> bool:
    """Check if resized image exists on S3."""
    key = _resized_s3_key_for(output_prefix, source_id, image_id, threshold)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def phase_1_scan_for_resize(
    df_images: pd.DataFrame,
    bucket: str,
    output_prefix: str,
    threshold: int = 2048,
    s3_client: Any | None = None,
    workers: int = 32,
) -> pd.DataFrame:
    """Phase 1: Scan which images need resizing and don't yet exist on S3.

    Args:
        df_images: Images parquet with columns [source_id, image_id, width, height, needs_resize]
        bucket: S3 bucket name
        output_prefix: S3 prefix for ETL outputs
        threshold: Resize threshold (longest edge, pixels)
        s3_client: Boto3 S3 client (created if None)
        workers: ThreadPoolExecutor concurrency for S3 checks

    Returns:
        DataFrame with columns [source_id, image_id, width, height, original_s3_key, output_s3_key]
        for images that need resizing and don't yet exist on S3.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    # Filter to images that need resizing
    df_todo = df_images[df_images["needs_resize"]].copy()

    if len(df_todo) == 0:
        logger.info("No images need resizing")
        return pd.DataFrame(
            columns=["source_id", "image_id", "width", "height", "original_s3_key", "output_s3_key"]
        )

    # Check which ones don't yet exist on S3
    def check_and_build_row(row: pd.Series) -> dict[str, Any] | None:
        source_id = int(row["source_id"])
        image_id = str(row["image_id"])
        exists = _check_resized_exists(
            bucket=bucket,
            output_prefix=output_prefix,
            source_id=source_id,
            image_id=image_id,
            threshold=threshold,
            s3_client=s3_client,
        )
        if exists:
            return None  # Skip, already resized
        return {
            "source_id": source_id,
            "image_id": image_id,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "original_s3_key": _s3_key_for(output_prefix, source_id, image_id),
            "output_s3_key": _resized_s3_key_for(output_prefix, source_id, image_id, threshold),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(check_and_build_row, row): idx for idx, row in df_todo.iterrows()
        }
        rows = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1: Scanning"):
            try:
                result = future.result()
                if result is not None:
                    rows.append(result)
            except Exception as e:
                logger.error("Error checking S3 for resized image: %s", e)

    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=["source_id", "image_id", "width", "height", "original_s3_key", "output_s3_key"]
        )
    )


@dataclass
class ResizeConfig:
    """Configuration for image resizing."""

    bucket: str
    output_prefix: str
    threshold: int = 2048
    workers: int = 16
    checkpoint_every: int = 500
    temp_dir: str = "/tmp/coralnet-resize-checkpoint"


def write_checkpoint(checkpoint_path: Path | str, df: pd.DataFrame) -> None:
    """Write checkpoint parquet to disk.

    Args:
        checkpoint_path: Path to write checkpoint parquet
        df: DataFrame with columns [source_id, image_id, status, resize_timestamp, error_message]
    """
    df = df.copy()
    df["source_id"] = df["source_id"].astype("int32")
    df["image_id"] = df["image_id"].astype("string")
    df["status"] = df["status"].astype("string")
    df.to_parquet(checkpoint_path, engine="pyarrow", index=False)
    logger.info("Checkpoint written: %s", checkpoint_path)


def read_checkpoint(checkpoint_path: Path | str) -> pd.DataFrame:
    """Read checkpoint parquet from disk.

    Args:
        checkpoint_path: Path to checkpoint parquet

    Returns:
        DataFrame with checkpoint data, or empty DataFrame with correct schema if file doesn't exist.
    """
    if not Path(checkpoint_path).exists():
        return pd.DataFrame(columns=list(CHECKPOINT_SCHEMA.keys()))
    return pd.read_parquet(checkpoint_path)


def get_pending_items(df_checkpoint: pd.DataFrame) -> pd.DataFrame:
    """Extract only pending items from checkpoint.

    Args:
        df_checkpoint: Checkpoint DataFrame

    Returns:
        Subset of df_checkpoint with status == "pending"
    """
    return df_checkpoint[df_checkpoint["status"] == "pending"].copy()


def init_checkpoint_from_todo(df_todo: pd.DataFrame) -> pd.DataFrame:
    """Create initial checkpoint from todo list.

    Args:
        df_todo: Todo list DataFrame with columns [source_id, image_id, ...]

    Returns:
        Checkpoint DataFrame with all items marked as "pending"
    """
    return pd.DataFrame(
        {
            "source_id": df_todo["source_id"],
            "image_id": df_todo["image_id"],
            "status": "pending",
            "resize_timestamp": None,
            "error_message": None,
        }
    )


# Placeholder functions for later tasks
def run_phase_1_scan(*args, **kwargs):
    """Phase 1: Scan for images that need resizing."""
    raise NotImplementedError("Task 5")


def run_phase_2_resize(*args, **kwargs):
    """Phase 2: Resize images and upload to S3."""
    raise NotImplementedError("Task 5")
