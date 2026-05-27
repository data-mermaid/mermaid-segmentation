"""Per-image dimension scan: turn unique annotation keys into an images parquet.

For every unique ``(source_id, image_id)`` in the annotations DataFrame, this module reads the JPEG
header from S3 and records ``(width, height, longest_edge, file_size, needs_resize)``. With 1M+
images this is the most expensive ETL step — concurrent ranged GETs + checkpointing make it
restartable when SageMaker credentials rotate mid-run.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from botocore.client import BaseClient
from tqdm import tqdm

from .annotations import derive_unique_images
from .config import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_IMAGE_WORKERS,
    DEFAULT_RESIZE_THRESHOLD,
    get_bucket,
    get_prefix,
)
from .io import make_s3_client
from .jpeg_header import read_jpeg_size_from_s3
from .schemas import IMAGES_PRIMARY_KEY, IMAGES_SCHEMA, validate

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_thread_client() -> BaseClient:
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = make_s3_client()
    return _thread_local.s3


def _s3_key_for(prefix: str, source_id: int, image_id: str) -> str:
    return f"{prefix}/s{source_id}/images/{image_id}.jpg"


def _scan_one(
    bucket: str,
    prefix: str,
    source_id: int,
    image_id: str,
    *,
    s3_client: BaseClient | None,
) -> dict[str, Any]:
    client = s3_client or _get_thread_client()
    key = _s3_key_for(prefix, source_id, image_id)
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
) -> dict[str, Any]:
    bucket, prefix, source_id, image_id = args
    try:
        return _scan_one(bucket, prefix, source_id, image_id, s3_client=None)
    except Exception as e:  # noqa: BLE001 - never crash the worker
        return {
            "source_id": int(source_id),
            "image_id": str(image_id),
            "s3_key": _s3_key_for(prefix, source_id, image_id),
            "width": None,
            "height": None,
            "longest_edge": None,
            "file_size": None,
            "needs_resize": False,
            "header_status": "worker_exception",
            "error_message": f"{type(e).__name__}: {e}",
        }


def _finalise(df: pd.DataFrame, resize_threshold: int) -> pd.DataFrame:
    df = df.copy()
    df["source_id"] = df["source_id"].astype("int32")
    df["image_id"] = df["image_id"].astype("string")
    df["s3_key"] = df["s3_key"].astype("string")
    for col in ("width", "height", "longest_edge"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
    df["file_size"] = pd.to_numeric(df["file_size"], errors="coerce").astype("Int64")
    df["needs_resize"] = df["longest_edge"].fillna(0).astype("Int32").gt(resize_threshold)
    df["header_status"] = df["header_status"].astype("string")
    df["error_message"] = df["error_message"].astype("string")
    df = df.sort_values(list(IMAGES_PRIMARY_KEY), kind="mergesort").reset_index(drop=True)
    validate(df, IMAGES_SCHEMA)
    return df


def _load_checkpoint(
    checkpoint_path: Path,
) -> tuple[list[pd.DataFrame], set[tuple[int, str]]]:
    """Read all part files alongside ``checkpoint_path``; return frames + processed keys.

    Reads each part exactly once. The caller uses the frames in the final concat and the ``seen``
    set to skip already-processed ``(source_id, image_id)`` pairs.
    """
    parent = checkpoint_path.parent
    stem = checkpoint_path.stem
    parts = sorted(parent.glob(f"{stem}.part_*.parquet"))
    if not parts:
        return [], set()
    logger.info("Found %d existing checkpoint part files; resuming.", len(parts))
    frames: list[pd.DataFrame] = []
    seen: set[tuple[int, str]] = set()
    for part in parts:
        try:
            df = pd.read_parquet(part)
        except Exception as e:  # noqa: BLE001 - skip broken parts, don't crash resume
            logger.warning("Skipping unreadable checkpoint %s: %s", part, e)
            continue
        frames.append(df)
        for sid, iid in zip(df["source_id"].tolist(), df["image_id"].tolist(), strict=False):
            seen.add((int(sid), str(iid)))
    return frames, seen


def _next_part_index(checkpoint_path: Path) -> int:
    """Return the next unused part index from existing ``*.part_NNNNN.parquet`` filenames."""
    parent = checkpoint_path.parent
    stem = checkpoint_path.stem
    max_idx = -1
    for part in parent.glob(f"{stem}.part_*.parquet"):
        suffix = part.stem.removeprefix(f"{stem}.part_")
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def _write_part(rows: list[dict[str, Any]], checkpoint_path: Path, index: int) -> Path:
    parent = checkpoint_path.parent
    stem = checkpoint_path.stem
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / f"{stem}.part_{index:05d}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    logger.info("Wrote checkpoint part %s (%d rows)", out, len(rows))
    return out


def build_images(
    annotations_df: pd.DataFrame,
    bucket: str | None = None,
    prefix: str | None = None,
    *,
    workers: int = DEFAULT_IMAGE_WORKERS,
    resize_threshold: int = DEFAULT_RESIZE_THRESHOLD,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    s3_client: BaseClient | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scan JPEG headers for every unique image in ``annotations_df``.

    Args:
        annotations_df: Output of :func:`build_annotations`.
        bucket / prefix: S3 location overrides (default from env).
        workers: ThreadPoolExecutor size. Each worker gets a thread-local boto3 client.
        resize_threshold: ``needs_resize`` is True when ``longest_edge > resize_threshold``.
        checkpoint_path: Final parquet path; partial results land alongside as
            ``<stem>.part_NNNNN.parquet``. Restarts skip rows present in those parts.
        checkpoint_every: Number of rows between part-file flushes.
        s3_client: Pre-built boto3 client (tests). When set, runs sequentially.
        limit: For dev runs, only process the first N unique keys.

    Returns:
        DataFrame conforming to :data:`IMAGES_SCHEMA`.
    """
    bucket = bucket or get_bucket()
    prefix = prefix or get_prefix()

    unique = derive_unique_images(annotations_df)
    if limit is not None:
        unique = unique.head(limit)
    total = len(unique)
    if total == 0:
        logger.warning("build_images: no unique images in input; returning empty frame.")
        return _finalise(
            pd.DataFrame(columns=[f.name for f in IMAGES_SCHEMA]),
            resize_threshold,
        )

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    existing_frames, seen = _load_checkpoint(checkpoint) if checkpoint else ([], set())

    keys = unique[["source_id", "image_id"]].itertuples(index=False, name=None)
    todo = [(int(s), str(i)) for s, i in keys if (int(s), str(i)) not in seen]

    logger.info(
        "build_images: total=%d already_done=%d remaining=%d workers=%d",
        total,
        len(seen),
        len(todo),
        workers,
    )

    new_rows: list[dict[str, Any]] = []
    part_index = _next_part_index(checkpoint) if checkpoint else 0
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal part_index, buffer
        if not buffer or checkpoint is None:
            return
        existing_frames.append(pd.DataFrame(buffer))
        _write_part(buffer, checkpoint, part_index)
        part_index += 1
        buffer = []

    work = [(bucket, prefix, sid, iid) for sid, iid in todo]

    if workers > 1 and s3_client is None:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_scan_worker, item) for item in work]
            for done in tqdm(
                as_completed(futures), total=len(futures), desc="Reading JPEG headers"
            ):
                row = done.result()
                new_rows.append(row)
                buffer.append(row)
                if len(buffer) >= checkpoint_every:
                    flush()
    else:
        for item in tqdm(work, desc="Reading JPEG headers"):
            row = _scan_one(item[0], item[1], item[2], item[3], s3_client=s3_client)
            new_rows.append(row)
            buffer.append(row)
            if len(buffer) >= checkpoint_every:
                flush()

    flush()

    frames = existing_frames if checkpoint else [pd.DataFrame(new_rows)]
    if not frames or all(f.empty for f in frames):
        return _finalise(pd.DataFrame(columns=[f.name for f in IMAGES_SCHEMA]), resize_threshold)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=list(IMAGES_PRIMARY_KEY), keep="last")
    return _finalise(combined, resize_threshold)
