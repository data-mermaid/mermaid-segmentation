"""Image resizing preprocessing pipeline: scan for missing resized images, then resize and
upload."""

from __future__ import annotations

import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config
from PIL import Image
from tqdm import tqdm

from mermaidseg.datasets.coralnet.etl.config import DEFAULT_PREFIX

logger = logging.getLogger(__name__)

# Thread lock for checkpoint updates
_checkpoint_lock = threading.Lock()
_s3_thread_local = threading.local()


def make_resize_s3_client(*, max_pool_connections: int = 10) -> Any:
    """Build an S3 client sized for concurrent resize/scan workers."""
    config = Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=max_pool_connections,
    )
    return boto3.client("s3", config=config)


def _init_s3_worker(max_pool_connections: int) -> None:
    _s3_thread_local.s3 = make_resize_s3_client(max_pool_connections=max_pool_connections)


def _resolve_s3_client(explicit_client: Any | None) -> Any:
    if explicit_client is not None:
        return explicit_client
    client = getattr(_s3_thread_local, "s3", None)
    if client is None:
        client = make_resize_s3_client(max_pool_connections=10)
    return client


CHECKPOINT_SCHEMA = {
    "source_id": "int32",
    "image_id": "string",
    "status": "string",
    "resize_timestamp": "object",  # datetime
    "error_message": "string",
}

SCAN_OUTPUT_COLUMNS = [
    "source_id",
    "image_id",
    "width",
    "height",
    "original_s3_key",
    "output_s3_key",
]


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
    return f"{prefix}/resized/s{source_id}/images/{image_id}.jpg"


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


def scan_for_missing_resized_images(
    df_images: pd.DataFrame,
    bucket: str,
    output_prefix: str,
    threshold: int = 2048,
    s3_client: Any | None = None,
    workers: int = 32,
    source_prefix: str | None = None,
) -> pd.DataFrame:
    """Scan which images need resizing and don't yet exist on S3.

    Args:
        df_images: Images parquet with columns [source_id, image_id, width, height, needs_resize]
            and optionally ``s3_key`` for the original object location.
        bucket: S3 bucket name
        output_prefix: S3 prefix for resized image outputs
        threshold: Resize threshold (longest edge, pixels)
        s3_client: Boto3 S3 client (created if None)
        workers: ThreadPoolExecutor concurrency for S3 checks
        source_prefix: Fallback prefix for original keys when ``s3_key`` is absent

    Returns:
        DataFrame with columns [source_id, image_id, width, height, original_s3_key, output_s3_key]
        for images that need resizing and don't yet exist on S3.
    """
    source_prefix = source_prefix or DEFAULT_PREFIX
    pool_size = workers + 4

    # Filter to images that need resizing
    df_todo = df_images[df_images["needs_resize"]].copy()

    if len(df_todo) == 0:
        logger.info("No images need resizing")
        return pd.DataFrame(columns=SCAN_OUTPUT_COLUMNS)

    def check_and_build_row(row: pd.Series) -> dict[str, Any] | None:
        client = _resolve_s3_client(s3_client)
        source_id = int(row["source_id"])
        image_id = str(row["image_id"])
        exists = _check_resized_exists(
            bucket=bucket,
            output_prefix=output_prefix,
            source_id=source_id,
            image_id=image_id,
            threshold=threshold,
            s3_client=client,
        )
        if exists:
            return None  # Skip, already resized
        if "s3_key" in row.index and pd.notna(row["s3_key"]):
            original_s3_key = str(row["s3_key"])
        else:
            original_s3_key = _s3_key_for(source_prefix, source_id, image_id)
        return {
            "source_id": source_id,
            "image_id": image_id,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "original_s3_key": original_s3_key,
            "output_s3_key": _resized_s3_key_for(output_prefix, source_id, image_id, threshold),
        }

    with ThreadPoolExecutor(
        max_workers=workers,
        initializer=_init_s3_worker,
        initargs=(pool_size,),
    ) as executor:
        futures = {
            executor.submit(check_and_build_row, row): idx for idx, row in df_todo.iterrows()
        }
        rows = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Scanning for missing resized images"
        ):
            try:
                result = future.result()
                if result is not None:
                    rows.append(result)
            except Exception as e:
                logger.error("Error checking S3 for resized image: %s", e)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=SCAN_OUTPUT_COLUMNS)


def build_todo_from_checkpoint(
    df_images: pd.DataFrame,
    checkpoint_path: Path | str,
    output_prefix: str,
    threshold: int = 2048,
    source_prefix: str | None = None,
) -> pd.DataFrame:
    """Build a resize todo list from pending checkpoint rows and an images parquet."""
    source_prefix = source_prefix or DEFAULT_PREFIX
    pending = get_pending_items(read_checkpoint(checkpoint_path))
    if pending.empty:
        return pd.DataFrame(columns=SCAN_OUTPUT_COLUMNS)

    df_keys = df_images.drop_duplicates(subset=["source_id", "image_id"])
    merged = pending.merge(df_keys, on=["source_id", "image_id"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=SCAN_OUTPUT_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        source_id = int(row["source_id"])
        image_id = str(row["image_id"])
        if "s3_key" in row.index and pd.notna(row["s3_key"]):
            original_s3_key = str(row["s3_key"])
        else:
            original_s3_key = _s3_key_for(source_prefix, source_id, image_id)
        rows.append(
            {
                "source_id": source_id,
                "image_id": image_id,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "original_s3_key": original_s3_key,
                "output_s3_key": _resized_s3_key_for(output_prefix, source_id, image_id, threshold),
            }
        )
    return pd.DataFrame(rows)


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


def _update_checkpoint(
    checkpoint_path: Path | str,
    source_id: int,
    image_id: str,
    status: str,
    resize_timestamp: datetime | None = None,
    error_message: str | None = None,
) -> None:
    """Atomically update checkpoint status for a single image.

    Args:
        checkpoint_path: Path to checkpoint parquet
        source_id: Source dataset ID
        image_id: Image ID
        status: New status ('completed' or 'failed')
        resize_timestamp: Timestamp of completion
        error_message: Error message if status='failed'
    """
    with _checkpoint_lock:
        df_checkpoint = read_checkpoint(checkpoint_path)
        mask = (df_checkpoint["source_id"] == source_id) & (df_checkpoint["image_id"] == image_id)
        df_checkpoint.loc[mask, "status"] = status
        if resize_timestamp is not None:
            df_checkpoint.loc[mask, "resize_timestamp"] = resize_timestamp
        if error_message is not None:
            df_checkpoint.loc[mask, "error_message"] = error_message
        write_checkpoint(checkpoint_path, df_checkpoint)


def resize_and_upload_image(
    todo_item: dict[str, Any],
    bucket: str,
    checkpoint_path: Path | str,
    threshold: int = 2048,
    s3_client: Any | None = None,
) -> None:
    """Resize a single image: download, resize, upload, update checkpoint.

    Args:
        todo_item: Dict with keys [source_id, image_id, original_s3_key, output_s3_key, ...]
        bucket: S3 bucket name
        checkpoint_path: Path to checkpoint parquet
        threshold: Resize threshold
        s3_client: Boto3 S3 client
    """
    if s3_client is None:
        s3_client = _resolve_s3_client(None)

    source_id = todo_item["source_id"]
    image_id = todo_item["image_id"]
    original_key = todo_item["original_s3_key"]
    output_key = todo_item["output_s3_key"]

    try:
        # Download
        logger.debug("Downloading %s", original_key)
        response = s3_client.get_object(Bucket=bucket, Key=original_key)
        original_bytes = io.BytesIO(response["Body"].read())

        # Resize
        logger.debug("Resizing %s/%s", source_id, image_id)
        resized_bytes = resize_image_to_threshold(original_bytes, threshold=threshold)

        # Upload
        logger.debug("Uploading to %s", output_key)
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=resized_bytes.getvalue(),
            ContentType="image/jpeg",
        )

        # Verify upload
        s3_client.head_object(Bucket=bucket, Key=output_key)

        # Update checkpoint
        _update_checkpoint(
            checkpoint_path,
            source_id,
            image_id,
            "completed",
            resize_timestamp=datetime.now(),
        )

        logger.info("Resized: %s/%s -> %s", source_id, image_id, output_key)

    except Exception as e:
        logger.error("Failed to resize %s/%s: %s", source_id, image_id, e)
        _update_checkpoint(
            checkpoint_path,
            source_id,
            image_id,
            "failed",
            error_message=f"{type(e).__name__}: {e}",
        )


def resize_and_upload_all_images(
    df_todo: pd.DataFrame,
    bucket: str,
    output_prefix: str,
    checkpoint_path: Path | str,
    threshold: int = 2048,
    workers: int = 16,
    s3_client: Any | None = None,
) -> tuple[int, int, int]:
    """Download, resize, and upload all images with atomic checkpoint updates.

    Args:
        df_todo: DataFrame with columns [source_id, image_id, ..., original_s3_key, output_s3_key]
        bucket: S3 bucket name
        output_prefix: S3 prefix (for logging)
        checkpoint_path: Path to checkpoint parquet
        threshold: Resize threshold
        workers: ThreadPoolExecutor concurrency
        s3_client: Boto3 S3 client

    Returns:
        Tuple (num_resized, num_skipped, num_failed)
    """
    pool_size = workers + 4

    # Initialize checkpoint if not exists
    if not Path(checkpoint_path).exists():
        checkpoint_df = init_checkpoint_from_todo(df_todo)
        write_checkpoint(checkpoint_path, checkpoint_df)

    # Get pending items
    checkpoint_df = read_checkpoint(checkpoint_path)
    df_pending = get_pending_items(checkpoint_df)

    if len(df_pending) == 0:
        logger.info("No pending items in checkpoint")
        completed = len(checkpoint_df[checkpoint_df["status"] == "completed"])
        failed = len(checkpoint_df[checkpoint_df["status"] == "failed"])
        return completed, 0, failed

    # Merge pending with todo to get S3 keys
    df_work = df_pending.merge(
        df_todo[["source_id", "image_id", "original_s3_key", "output_s3_key"]],
        on=["source_id", "image_id"],
    )

    with ThreadPoolExecutor(
        max_workers=workers,
        initializer=_init_s3_worker,
        initargs=(pool_size,),
    ) as executor:
        futures = {
            executor.submit(
                resize_and_upload_image,
                row.to_dict(),
                bucket,
                checkpoint_path,
                threshold,
                None if s3_client is None else s3_client,
            ): idx
            for idx, row in df_work.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing images"):
            try:
                future.result()
            except Exception as e:
                logger.error("Resize failed: %s", e)

    # Final counts
    checkpoint_df = read_checkpoint(checkpoint_path)
    num_resized = len(checkpoint_df[checkpoint_df["status"] == "completed"])
    num_failed = len(checkpoint_df[checkpoint_df["status"] == "failed"])
    num_skipped = len(df_todo) - len(df_pending)

    logger.info(
        "Resizing complete: %d resized, %d skipped (already on S3), %d failed",
        num_resized,
        num_skipped,
        num_failed,
    )

    return num_resized, num_skipped, num_failed
