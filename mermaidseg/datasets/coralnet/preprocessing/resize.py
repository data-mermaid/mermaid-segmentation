"""Image resizing preprocessing pipeline: scan for missing resized images, then resize and
upload."""

from __future__ import annotations

import io
import logging
import os
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
from mermaidseg.datasets.coralnet.preprocessing.inspect import (
    ImageInspection,
    IssueType,
    inspect_image,
)

logger = logging.getLogger(__name__)

# Rows between checkpoint flushes. Workers are pure (they return result rows); the orchestrator
# applies them to an in-memory checkpoint and flushes to disk every this many completions, so a
# crash loses at most this many results instead of rewriting the whole parquet per image.
DEFAULT_CHECKPOINT_EVERY = 1000


def make_resize_s3_client(*, max_pool_connections: int = 10) -> Any:
    """Build an S3 client sized for concurrent resize/scan workers.

    boto3 clients are thread-safe, so the orchestrator builds one client with ``max_pool_connections
    >= workers`` and shares it across all worker threads.
    """
    config = Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=max_pool_connections,
    )
    return boto3.client("s3", config=config)


CHECKPOINT_SCHEMA = {
    "source_id": "int32",
    "image_id": "string",
    "status": "string",
    "resize_timestamp": "object",  # datetime
    "error_message": "string",
    "skip_reason": "string",  # "already_exists", "corrupted_[issue_type]", or null
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
    image_bytes.seek(0)  # callers may have already read this buffer (e.g. during inspection)
    img = Image.open(image_bytes)
    img.load()

    width, height = img.size
    longest_edge = max(width, height)
    needs_resize = longest_edge > threshold
    # JPEG can only encode RGB or grayscale; alpha/palette modes (RGBA, LA, P) must be converted.
    needs_convert = img.mode not in ("RGB", "L")

    # Fast path: already a JPEG in a JPEG-encodable mode and small enough — return bytes unchanged.
    if not needs_resize and not needs_convert and img.format == "JPEG":
        image_bytes.seek(0)
        return image_bytes

    if needs_resize:
        scale = threshold / longest_edge
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output


def _s3_key_for(prefix: str, source_id: int, image_id: str) -> str:
    """S3 key for a CoralNet original image."""
    return f"{prefix}/s{source_id}/images/{image_id}.jpg"


def _resized_s3_key_for(prefix: str, source_id: int, image_id: str, threshold: int) -> str:
    """S3 key for a resized image."""
    return f"{prefix}/resized/s{source_id}/images/{image_id}.jpg"


def _list_existing_resized_keys(s3_client: Any, bucket: str, output_prefix: str) -> set[str]:
    """Return every object key under ``<output_prefix>/resized/`` via paginated LIST.

    One LIST request covers 1000 objects, so this replaces hundreds of thousands of per-image
    ``head_object`` calls with a few hundred sequential pages.
    """
    resized_prefix = f"{output_prefix}/resized/"
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=resized_prefix):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys


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

    Lists the resized output prefix once and diffs against the expected keys in memory — no
    per-image S3 requests.

    Args:
        df_images: Images parquet with columns [source_id, image_id, width, height, needs_resize]
            and optionally ``s3_key`` for the original object location.
        bucket: S3 bucket name
        output_prefix: S3 prefix for resized image outputs
        threshold: Resize threshold (longest edge, pixels)
        s3_client: Boto3 S3 client (created if None)
        workers: Unused; retained for backward compatibility (the LIST is sequential)
        source_prefix: Fallback prefix for original keys when ``s3_key`` is absent

    Returns:
        DataFrame with columns [source_id, image_id, width, height, original_s3_key, output_s3_key]
        for images that need resizing and don't yet exist on S3.
    """
    source_prefix = source_prefix or DEFAULT_PREFIX

    # Filter to images that need resizing
    df_todo = df_images[df_images["needs_resize"]].copy()

    if len(df_todo) == 0:
        logger.info("No images need resizing")
        return pd.DataFrame(columns=SCAN_OUTPUT_COLUMNS)

    client = s3_client if s3_client is not None else make_resize_s3_client()
    logger.info("Listing existing resized objects under s3://%s/%s/resized/", bucket, output_prefix)
    existing = _list_existing_resized_keys(client, bucket, output_prefix)
    logger.info("Found %d existing resized objects", len(existing))

    rows: list[dict[str, Any]] = []
    has_s3_key = "s3_key" in df_todo.columns
    for row in df_todo.itertuples(index=False):
        source_id = int(row.source_id)
        image_id = str(row.image_id)
        output_key = _resized_s3_key_for(output_prefix, source_id, image_id, threshold)
        if output_key in existing:
            continue  # Skip, already resized
        if has_s3_key and pd.notna(row.s3_key):
            original_s3_key = str(row.s3_key)
        else:
            original_s3_key = _s3_key_for(source_prefix, source_id, image_id)
        rows.append(
            {
                "source_id": source_id,
                "image_id": image_id,
                "width": int(row.width),
                "height": int(row.height),
                "original_s3_key": original_s3_key,
                "output_s3_key": output_key,
            }
        )

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
        df: DataFrame with columns [source_id, image_id, status, resize_timestamp, error_message, skip_reason]
    """
    df = df.copy()
    df["source_id"] = df["source_id"].astype("int32")
    df["image_id"] = df["image_id"].astype("string")
    df["status"] = df["status"].astype("string")
    if "skip_reason" not in df.columns:
        df["skip_reason"] = None
    df["skip_reason"] = df["skip_reason"].astype("string")
    # Write to a sibling temp file then atomically rename, so a crash mid-flush leaves the previous
    # checkpoint intact rather than a half-written parquet.
    checkpoint_path = Path(checkpoint_path)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    df.to_parquet(tmp_path, engine="pyarrow", index=False)
    os.replace(tmp_path, checkpoint_path)
    logger.debug("Checkpoint written: %s (%d rows)", checkpoint_path, len(df))


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
            "skip_reason": None,
        }
    )


def _validate_and_download(
    bucket: str,
    original_key: str,
    source_id: int,
    image_id: str,
    s3_client: Any,
) -> tuple[io.BytesIO | None, ImageInspection]:
    """Download and validate image in one step.

    Returns:
        (image_bytes, inspection) where image_bytes is None if validation failed.
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=original_key)
        image_bytes = io.BytesIO(response["Body"].read())
        inspection = inspect_image(image_bytes)
        if not inspection.is_valid:
            logger.warning(
                "Validation failed for %s/%s: %s — %s",
                source_id,
                image_id,
                inspection.issue_type.value,
                inspection.error_message,
            )
        return (image_bytes if inspection.is_valid else None, inspection)
    except Exception as e:
        logger.warning("Failed to download %s/%s: %s", source_id, image_id, e)
        return (
            None,
            ImageInspection(
                is_valid=False,
                issue_type=IssueType.UNKNOWN,
                format=None,
                channels=None,
                width=None,
                height=None,
                error_message=f"Download failed: {type(e).__name__}",
            ),
        )


def _result_row(
    source_id: int,
    image_id: str,
    status: str,
    *,
    resize_timestamp: datetime | None = None,
    error_message: str | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """Build a checkpoint result row for a single processed image."""
    return {
        "source_id": int(source_id),
        "image_id": str(image_id),
        "status": status,
        "resize_timestamp": resize_timestamp,
        "error_message": error_message,
        "skip_reason": skip_reason,
    }


def resize_and_upload_image(
    todo_item: dict[str, Any],
    bucket: str,
    threshold: int = 2048,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Resize a single image: download, validate, resize, upload.

    Pure worker — it touches S3 but never the checkpoint. The orchestrator applies the returned
    row to the in-memory checkpoint and flushes it periodically.

    Args:
        todo_item: Dict with keys [source_id, image_id, original_s3_key, output_s3_key, ...]
        bucket: S3 bucket name
        threshold: Resize threshold
        s3_client: Boto3 S3 client

    Returns:
        A checkpoint result row (see :func:`_result_row`) with status
        'completed', 'skipped', or 'failed'.
    """
    if s3_client is None:
        s3_client = make_resize_s3_client()

    source_id = todo_item["source_id"]
    image_id = todo_item["image_id"]
    original_key = todo_item["original_s3_key"]
    output_key = todo_item["output_s3_key"]

    try:
        # Download and validate
        logger.debug("Downloading and validating %s", original_key)
        original_bytes, inspection = _validate_and_download(
            bucket=bucket,
            original_key=original_key,
            source_id=source_id,
            image_id=image_id,
            s3_client=s3_client,
        )

        if original_bytes is None:
            return _result_row(
                source_id,
                image_id,
                "skipped",
                skip_reason=f"corrupted_{inspection.issue_type.value}",
            )

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

        # No post-upload head_object verify: S3 PUT is strongly consistent, and a non-2xx response
        # already raises out of put_object above.
        logger.debug("Resized: %s/%s -> %s", source_id, image_id, output_key)
        return _result_row(source_id, image_id, "completed", resize_timestamp=datetime.now())

    except Exception as e:
        logger.error("Failed to resize %s/%s: %s", source_id, image_id, e)
        return _result_row(source_id, image_id, "failed", error_message=f"{type(e).__name__}: {e}")


def _summarise_checkpoint(checkpoint_df: pd.DataFrame) -> tuple[int, int, int, int]:
    """Count (resized, skipped, failed, corrupted) from checkpoint statuses alone.

    Counts come from the checkpoint, never from todo-list arithmetic — on a resume the todo can be
    smaller than the checkpoint, which previously produced negative "skipped" counts.
    """
    num_resized = int((checkpoint_df["status"] == "completed").sum())
    num_failed = int((checkpoint_df["status"] == "failed").sum())
    is_skipped = checkpoint_df["status"] == "skipped"
    is_corrupted = checkpoint_df["skip_reason"].str.startswith("corrupted_").fillna(False)
    num_corrupted = int((is_skipped & is_corrupted).sum())
    num_skipped = int((is_skipped & ~is_corrupted).sum())
    return num_resized, num_skipped, num_failed, num_corrupted


def resize_and_upload_all_images(
    df_todo: pd.DataFrame,
    bucket: str,
    output_prefix: str,
    checkpoint_path: Path | str,
    threshold: int = 2048,
    workers: int = 16,
    s3_client: Any | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> tuple[int, int, int, int]:
    """Download, resize, and upload all images, checkpointing in periodic batches.

    Workers are pure (they return result rows); this orchestrator owns the checkpoint, applies each
    returned row to an in-memory DataFrame, and flushes it to disk every ``checkpoint_every``
    completions. That replaces the old per-image full-file read-modify-write under a global lock.

    Args:
        df_todo: DataFrame with columns [source_id, image_id, ..., original_s3_key, output_s3_key]
        bucket: S3 bucket name
        output_prefix: S3 prefix (for logging)
        checkpoint_path: Path to checkpoint parquet
        threshold: Resize threshold
        workers: ThreadPoolExecutor concurrency
        s3_client: Boto3 S3 client
        checkpoint_every: Rows between checkpoint flushes

    Returns:
        Tuple (num_resized, num_skipped, num_failed, num_corrupted)
    """
    # One shared client for all worker threads (boto3 clients are thread-safe), with the connection
    # pool sized to the concurrency so workers never queue on connections.
    if s3_client is None:
        s3_client = make_resize_s3_client(max_pool_connections=workers + 4)

    # Initialize checkpoint if not exists
    if not Path(checkpoint_path).exists():
        write_checkpoint(checkpoint_path, init_checkpoint_from_todo(df_todo))

    # Get pending items. Reset the index so positional .at updates line up with the row map below.
    checkpoint_df = read_checkpoint(checkpoint_path).reset_index(drop=True)
    df_pending = get_pending_items(checkpoint_df)

    if len(df_pending) == 0:
        logger.info("No pending items in checkpoint")
        return _summarise_checkpoint(checkpoint_df)

    # Merge pending with todo to get S3 keys
    df_work = df_pending.merge(
        df_todo[["source_id", "image_id", "original_s3_key", "output_s3_key"]],
        on=["source_id", "image_id"],
    )

    # Map (source_id, image_id) -> positional row index for O(1) in-memory status updates.
    row_index: dict[tuple[int, str], int] = {
        (int(sid), str(iid)): pos
        for pos, (sid, iid) in enumerate(
            zip(checkpoint_df["source_id"], checkpoint_df["image_id"], strict=True)
        )
    }

    def apply_result(res: dict[str, Any]) -> None:
        pos = row_index[(int(res["source_id"]), str(res["image_id"]))]
        checkpoint_df.loc[pos, "status"] = res["status"]
        if res["resize_timestamp"] is not None:
            checkpoint_df.loc[pos, "resize_timestamp"] = res["resize_timestamp"]
        if res["error_message"] is not None:
            checkpoint_df.loc[pos, "error_message"] = res["error_message"]
        if res["skip_reason"] is not None:
            checkpoint_df.loc[pos, "skip_reason"] = res["skip_reason"]

    since_flush = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                resize_and_upload_image,
                row.to_dict(),
                bucket,
                threshold,
                s3_client,
            ): idx
            for idx, row in df_work.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing images"):
            try:
                result = future.result()
            except Exception as e:
                logger.error("Resize failed: %s", e)
                continue
            apply_result(result)
            since_flush += 1
            if since_flush >= checkpoint_every:
                write_checkpoint(checkpoint_path, checkpoint_df)
                since_flush = 0

    # Final flush of any rows since the last checkpoint.
    write_checkpoint(checkpoint_path, checkpoint_df)

    # Final counts
    num_resized, num_skipped, num_failed, num_corrupted = _summarise_checkpoint(checkpoint_df)

    logger.info(
        "Resizing complete: %d resized, %d already exist, %d corrupted, %d failed",
        num_resized,
        num_skipped,
        num_corrupted,
        num_failed,
    )

    return num_resized, num_skipped, num_failed, num_corrupted
