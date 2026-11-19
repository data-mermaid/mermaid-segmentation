"""S3 publishing helpers: JSON and parquet puts, key listing and bulk file upload."""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}


@dataclass(frozen=True)
class S3Target:
    """Destination bucket and key prefix; ``dry_run`` suppresses every write."""

    bucket: str
    prefix: str
    dry_run: bool


def _key_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in _MISSING_CODES:
            return False
        raise
    return True


def ensure_absent(client: Any, target: S3Target, names: list[str], *, overwrite: bool) -> None:
    """Raise ``FileExistsError`` if any ``<prefix>/<name>`` exists, unless ``overwrite``.

    Runs on dry-run targets too, so a dry run fails the way the real run would.
    """
    if overwrite:
        return
    for name in names:
        key = f"{target.prefix}/{name}"
        if _key_exists(client, target.bucket, key):
            raise FileExistsError(
                f"s3://{target.bucket}/{key} already exists; pass --overwrite to replace it"
            )


def _put_bytes(
    client: Any, target: S3Target, name: str, body: bytes, content_type: str, *, overwrite: bool
) -> str:
    key = f"{target.prefix}/{name}"
    uri = f"s3://{target.bucket}/{key}"
    if target.dry_run:
        logger.info("[dry-run] would write %s", uri)
        return uri
    if not overwrite and _key_exists(client, target.bucket, key):
        raise FileExistsError(f"{uri} already exists; pass --overwrite to replace it")
    client.put_object(Bucket=target.bucket, Key=key, Body=body, ContentType=content_type)
    logger.info("Wrote %s", uri)
    return uri


def put_json(
    client: Any, target: S3Target, name: str, payload: dict[str, Any], *, overwrite: bool
) -> str:
    """Write ``payload`` as JSON to ``<prefix>/<name>`` and return its s3:// URI."""
    body = json.dumps(payload, indent=2).encode("utf-8")
    return _put_bytes(client, target, name, body, "application/json", overwrite=overwrite)


def put_parquet(
    client: Any,
    target: S3Target,
    name: str,
    df: pd.DataFrame,
    *,
    sort_by: list[str],
    overwrite: bool,
) -> str:
    """Write ``df`` sorted by ``sort_by`` as parquet to ``<prefix>/<name>`` and return its URI."""
    ordered = df.sort_values(sort_by, kind="stable").reset_index(drop=True)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(ordered, preserve_index=False), buf)
    return _put_bytes(
        client, target, name, buf.getvalue(), "application/octet-stream", overwrite=overwrite
    )


def list_keys(client: Any, bucket: str, prefix: str) -> set[str]:
    """Return every object key under ``prefix``."""
    keys: set[str] = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.update(obj["Key"] for obj in page.get("Contents", []))
    return keys


def upload_files(
    client: Any,
    target: S3Target,
    files: list[tuple[Path, str]],
    *,
    workers: int,
    skip_existing: bool,
) -> int:
    """Upload ``(local_path, key)`` pairs concurrently; return how many were (or would be) sent."""
    pending = files
    if skip_existing:
        existing = list_keys(client, target.bucket, target.prefix)
        pending = [(path, key) for path, key in files if key not in existing]
    if target.dry_run:
        logger.info("[dry-run] would upload %d of %d files", len(pending), len(files))
        return len(pending)

    def _upload(item: tuple[Path, str]) -> None:
        client.upload_file(str(item[0]), target.bucket, item[1])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Draining the iterator re-raises the first worker failure.
        list(pool.map(_upload, pending))
    return len(pending)
