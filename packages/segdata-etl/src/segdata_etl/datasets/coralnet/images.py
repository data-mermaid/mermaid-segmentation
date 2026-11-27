"""Per-image dimension scan: turn unique annotation keys into an images parquet.

For every unique ``(source_id, image_id)`` in the annotations DataFrame, this module reads the JPEG
header from S3 and records ``(width, height, longest_edge, file_size, needs_resize)``. With 1M+
images this is the most expensive step, so the header reads run as concurrent ranged GETs.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from botocore.client import BaseClient
from tqdm import tqdm

from coralnet_scraper.s3_io import object_key_image

from ...common.jpeg_header import read_jpeg_size_from_s3
from ...common.s3_objects import make_s3_client
from .annotations import derive_unique_images
from .config import RESIZE_THRESHOLD
from .schemas import IMAGES_PRIMARY_KEY, IMAGES_SCHEMA

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_thread_client() -> BaseClient:
    """Return this thread's boto3 client, building it on first use."""
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = make_s3_client()
    return _thread_local.s3


def _scan_one(
    bucket: str,
    prefix: str,
    source_id: int,
    image_id: str,
    *,
    s3_client: BaseClient | None,
) -> dict[str, Any]:
    """Read one image's JPEG header and return its images-parquet row."""
    client = s3_client or _get_thread_client()
    key = object_key_image(prefix, source_id, image_id)
    width, height, file_size, status = read_jpeg_size_from_s3(client, bucket, key)
    longest_edge: int | None = None
    if width is not None and height is not None:
        longest_edge = int(max(width, height))
    return {
        "source_id": int(source_id),
        "image_id": str(image_id),
        "s3_key": key,
        "width": width,
        "height": height,
        "longest_edge": longest_edge,
        "file_size": file_size,
        "needs_resize": False,  # Authoritative value is computed in _finalise.
        "header_status": status,
        "error_message": None,
    }


def _scan_worker(
    args: tuple[str, str, int, str],
    s3_client: BaseClient | None = None,
) -> dict[str, Any]:
    """Scan one image, recording any exception as a ``worker_exception`` row.

    Used by both the threaded and the sequential paths so a failing image never stops the scan.
    ``s3_client=None`` uses the calling thread's own client.
    """
    bucket, prefix, source_id, image_id = args
    try:
        return _scan_one(bucket, prefix, source_id, image_id, s3_client=s3_client)
    except Exception as e:  # noqa: BLE001 - never crash the worker
        return {
            "source_id": int(source_id),
            "image_id": str(image_id),
            "s3_key": object_key_image(prefix, source_id, image_id),
            "width": None,
            "height": None,
            "longest_edge": None,
            "file_size": None,
            "needs_resize": False,
            "header_status": "worker_exception",
            "error_message": f"{type(e).__name__}: {e}",
        }


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the scanned rows to the images dtypes, derive ``needs_resize`` and sort them."""
    df = df.copy()
    df["source_id"] = df["source_id"].astype("int32")
    df["image_id"] = df["image_id"].astype("string")
    df["s3_key"] = df["s3_key"].astype("string")
    for col in ("width", "height", "longest_edge"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
    df["file_size"] = pd.to_numeric(df["file_size"], errors="coerce").astype("Int64")
    df["needs_resize"] = df["longest_edge"].fillna(0).astype("Int32").gt(RESIZE_THRESHOLD)
    df["header_status"] = df["header_status"].astype("string")
    df["error_message"] = df["error_message"].astype("string")
    return df.sort_values(list(IMAGES_PRIMARY_KEY), kind="mergesort").reset_index(drop=True)


def build_images(
    annotations_df: pd.DataFrame,
    bucket: str,
    prefix: str,
    *,
    workers: int,
    s3_client: BaseClient | None = None,
) -> pd.DataFrame:
    """Scan JPEG headers for every unique image in ``annotations_df``.

    Args:
        annotations_df: Output of :func:`build_annotations`.
        bucket: S3 bucket.
        prefix: Key prefix holding the ``s<N>/`` folders.
        workers: ThreadPoolExecutor size. Each worker gets a thread-local boto3 client.
        s3_client: Pre-built boto3 client (tests). When set, runs sequentially.

    Returns:
        DataFrame conforming to :data:`IMAGES_SCHEMA`; ``needs_resize`` is True when
        ``longest_edge > RESIZE_THRESHOLD``. An image whose scan raises gets a row with
        ``header_status="worker_exception"`` on both the threaded and sequential paths.
    """
    unique = derive_unique_images(annotations_df)
    if unique.empty:
        logger.warning("build_images: no unique images in input; returning empty frame.")
        return _finalise(pd.DataFrame(columns=[f.name for f in IMAGES_SCHEMA]))

    keys = unique[["source_id", "image_id"]].itertuples(index=False, name=None)
    work = [(bucket, prefix, int(sid), str(iid)) for sid, iid in keys]
    logger.info("build_images: images=%d workers=%d", len(work), workers)

    rows: list[dict[str, Any]] = []
    if workers > 1 and s3_client is None:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_scan_worker, item) for item in work]
            for done in tqdm(
                as_completed(futures), total=len(futures), desc="Reading JPEG headers"
            ):
                rows.append(done.result())
    else:
        for item in tqdm(work, desc="Reading JPEG headers"):
            rows.append(_scan_worker(item, s3_client))

    return _finalise(pd.DataFrame(rows))
