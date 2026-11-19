"""Resize stage: find images that need a resized copy on S3, then resize and upload them.

S3 is the state: the resized objects under ``<resized_prefix>/resized/`` record what is done, so
a re-run lists them and resumes with the images still missing.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config
from PIL import Image
from tqdm import tqdm

from .config import DEFAULT_WORKERS, RESIZE_THRESHOLD
from .inspect import ImageInspection, inspect_image

logger = logging.getLogger(__name__)

SCAN_OUTPUT_COLUMNS = [
    "source_id",
    "image_id",
    "width",
    "height",
    "original_s3_key",
    "output_s3_key",
]


def make_resize_s3_client(*, max_pool_connections: int = 10) -> Any:
    """Build an S3 client sized for concurrent resize workers.

    boto3 clients are thread-safe, so one client with ``max_pool_connections >= workers`` is
    shared across all worker threads.
    """
    config = Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=max_pool_connections,
    )
    return boto3.client("s3", config=config)


def resize_image_to_threshold(
    image_bytes: io.BytesIO,
) -> io.BytesIO:
    """Resize an image so its longest edge equals ``RESIZE_THRESHOLD``, keeping the aspect ratio.

    An image whose longest edge is at most ``RESIZE_THRESHOLD`` keeps its size; it is still re-encoded
    when it is not a JPEG in a JPEG-encodable mode.

    Args:
        image_bytes: BytesIO with the image data.

    Returns:
        BytesIO with the resized JPEG, or the original bytes when no change is needed.
    """
    image_bytes.seek(0)  # callers may have already read this buffer (e.g. during inspection)
    img: Image.Image = Image.open(image_bytes)
    img.load()

    width, height = img.size
    longest_edge = max(width, height)
    needs_resize = longest_edge > RESIZE_THRESHOLD
    # JPEG can only encode RGB or grayscale; alpha/palette modes (RGBA, LA, P) must be converted.
    needs_convert = img.mode not in ("RGB", "L")

    if not needs_resize and not needs_convert and img.format == "JPEG":
        image_bytes.seek(0)
        return image_bytes

    if needs_resize:
        scale = RESIZE_THRESHOLD / longest_edge
        img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output


def _resized_s3_key_for(output_prefix: str, source_id: int, image_id: str) -> str:
    """Return the S3 key of a resized image."""
    return f"{output_prefix}/resized/s{source_id}/images/{image_id}.jpg"


def list_resized_keys(s3_client: Any, bucket: str, output_prefix: str) -> set[str]:
    """Return every object key under ``<output_prefix>/resized/`` via a paginated LIST.

    One LIST request covers 1000 objects, so this costs a few hundred sequential pages, not one
    request per image.
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
    s3_client: Any,
) -> pd.DataFrame:
    """Return the images that need resizing and have no resized object on S3 yet.

    Lists the resized output prefix once and diffs against the expected keys in memory.

    Args:
        df_images: Images parquet with columns [source_id, image_id, s3_key, width, height,
            needs_resize].
        bucket: S3 bucket name.
        output_prefix: S3 prefix that holds ``resized/``.
        s3_client: boto3 S3 client.

    Returns:
        DataFrame with columns [source_id, image_id, width, height, original_s3_key, output_s3_key].
    """
    df_todo = df_images[df_images["needs_resize"]]
    if df_todo.empty:
        logger.info("No images need resizing")
        return pd.DataFrame(columns=SCAN_OUTPUT_COLUMNS)

    logger.info("Listing existing resized objects under s3://%s/%s/resized/", bucket, output_prefix)
    existing = list_resized_keys(s3_client, bucket, output_prefix)
    logger.info("Found %d existing resized objects", len(existing))

    rows: list[dict[str, Any]] = []
    for row in df_todo.itertuples(index=False):
        source_id = int(row.source_id)
        image_id = str(row.image_id)
        output_key = _resized_s3_key_for(output_prefix, source_id, image_id)
        if output_key in existing:
            continue
        rows.append(
            {
                "source_id": source_id,
                "image_id": image_id,
                "width": int(row.width),
                "height": int(row.height),
                "original_s3_key": str(row.s3_key),
                "output_s3_key": output_key,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=SCAN_OUTPUT_COLUMNS)


def _validate_and_download(
    bucket: str,
    original_key: str,
    source_id: int,
    image_id: str,
    s3_client: Any,
) -> tuple[io.BytesIO | None, ImageInspection]:
    """Download an image and inspect it.

    Returns:
        ``(image_bytes, inspection)``; ``image_bytes`` is None when the inspection failed.

    Raises:
        Exception: Any S3 download error propagates so the caller reports a retryable failure.
    """
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


def _result_row(
    source_id: int,
    image_id: str,
    status: str,
    *,
    resize_timestamp: datetime | None = None,
    error_message: str | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """Build the result row for a single processed image."""
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
    s3_client: Any,
) -> dict[str, Any]:
    """Download, validate, resize and upload a single image.

    Args:
        todo_item: Dict with keys [source_id, image_id, original_s3_key, output_s3_key, ...].
        bucket: S3 bucket name.
        s3_client: boto3 S3 client.

    Returns:
        A result row (see :func:`_result_row`) with status ``completed``, ``skipped`` (with a
        ``corrupted_<issue>`` skip reason, for an image that fails inspection) or ``failed`` (for a
        download, resize or upload error, which is retryable).
    """
    source_id = todo_item["source_id"]
    image_id = todo_item["image_id"]
    original_key = todo_item["original_s3_key"]
    output_key = todo_item["output_s3_key"]

    try:
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

        logger.debug("Resizing %s/%s", source_id, image_id)
        resized_bytes = resize_image_to_threshold(original_bytes)

        logger.debug("Uploading to %s", output_key)
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=resized_bytes.getvalue(),
            ContentType="image/jpeg",
        )

        # S3 PUT is strongly consistent and a non-2xx response raises, so no verify read follows.
        logger.debug("Resized: %s/%s -> %s", source_id, image_id, output_key)
        return _result_row(source_id, image_id, "completed", resize_timestamp=datetime.now())

    except Exception as e:
        logger.error("Failed to resize %s/%s: %s", source_id, image_id, e)
        return _result_row(source_id, image_id, "failed", error_message=f"{type(e).__name__}: {e}")


def _count_results(results: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Count ``(resized, corrupted, failed)`` from worker result rows."""
    statuses = [r["status"] for r in results]
    return statuses.count("completed"), statuses.count("skipped"), statuses.count("failed")


def resize_and_upload_all_images(
    df_todo: pd.DataFrame,
    bucket: str,
    *,
    workers: int = DEFAULT_WORKERS,
    s3_client: Any,
) -> tuple[int, int, int]:
    """Resize and upload every image in ``df_todo`` on a thread pool.

    Args:
        df_todo: Output of :func:`scan_for_missing_resized_images`.
        bucket: S3 bucket name.
        workers: ThreadPoolExecutor concurrency.
        s3_client: boto3 S3 client shared by all workers; size its connection pool to
            ``workers`` (see :func:`make_resize_s3_client`).

    Returns:
        ``(resized, corrupted, failed)`` image counts.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(resize_and_upload_image, item, bucket, s3_client)
            for item in df_todo.to_dict("records")
        ]
        results = [
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing images")
        ]

    num_resized, num_corrupted, num_failed = _count_results(results)
    logger.info(
        "Resizing complete: %d resized, %d corrupted, %d failed",
        num_resized,
        num_corrupted,
        num_failed,
    )
    return num_resized, num_corrupted, num_failed
