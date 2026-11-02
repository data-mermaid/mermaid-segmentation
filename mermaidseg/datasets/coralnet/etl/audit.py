"""S3 audit pass: validate per-source CoralNet folders and emit an audit parquet.

Walks ``s3://<bucket>/<prefix>/s<source_id>/`` for every numeric source folder
and records per-source presence of expected files, row counts, and read
failures. The audit parquet is the input to both ``build_annotations`` and
``build_images`` — neither one re-walks S3.

CSV reads dispatch through a :data:`CsvReader` callable. The production path
uses ibis + DuckDB httpfs (port of PR #80) so we get ``null_padding=True``
ragged-row tolerance; tests inject a boto3-backed reader against a FakeS3
fixture. Boto3 still handles the non-CSV operations (paginator, head_object,
images-folder counts).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from botocore.client import BaseClient
from tqdm import tqdm

from .config import DEFAULT_WORKERS, get_bucket, get_prefix, get_workers
from .io import (
    CsvReader,
    make_csv_reader_ibis,
    make_csv_reader_s3,
    make_ibis_connection,
    make_s3_client,
    refresh_thread_ibis_credentials_if_present,
)
from .schemas import AUDIT_SCHEMA

logger = logging.getLogger(__name__)

_thread_local = threading.local()


@dataclass(frozen=True)
class _CsvSpec:
    filename: str
    has_field: str
    count_field: str
    empty_field: str
    read_failed_field: str


_ANNOTATIONS_SPEC = _CsvSpec(
    "annotations.csv",
    "has_annotations_csv",
    "n_annotations",
    "annotations_empty",
    "annotations_csv_read_failed",
)
_OTHER_SPECS: tuple[_CsvSpec, ...] = (
    _CsvSpec(
        "image_list.csv",
        "has_image_list_csv",
        "n_images_csv",
        "image_list_empty",
        "image_list_csv_read_failed",
    ),
    _CsvSpec(
        "labelset.csv",
        "has_labelset_csv",
        "n_labels",
        "labelset_empty",
        "labelset_csv_read_failed",
    ),
    _CsvSpec(
        "metadata.csv",
        "has_metadata_csv",
        "n_metadata_rows",
        "metadata_empty",
        "metadata_csv_read_failed",
    ),
)


def _empty_record(source_id: int, audit_ts: datetime) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "audit_timestamp": audit_ts,
        "has_images_folder": False,
        "has_annotations_csv": False,
        "has_image_list_csv": False,
        "has_labelset_csv": False,
        "has_metadata_csv": False,
        "n_images_s3": 0,
        "n_images_csv": 0,
        "n_annotations": 0,
        "n_unique_images_annotated": 0,
        "n_labels": None,
        "n_metadata_rows": None,
        "annotations_empty": False,
        "image_list_empty": False,
        "labelset_empty": False,
        "metadata_empty": False,
        "annotations_csv_read_failed": False,
        "image_list_csv_read_failed": False,
        "labelset_csv_read_failed": False,
        "metadata_csv_read_failed": False,
        "is_complete": False,
        "image_count_match": False,
        "errors": [],
    }


def list_source_folders(client: BaseClient, bucket: str, prefix: str) -> list[int]:
    """Return numeric source IDs under ``s3://<bucket>/<prefix>/s<N>/``."""
    full_prefix = f"{prefix}/" if not prefix.endswith("/") else prefix
    paginator = client.get_paginator("list_objects_v2")
    source_ids: list[int] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            folder = common["Prefix"].replace(full_prefix, "").rstrip("/")
            if folder.startswith("s") and folder[1:].isdigit():
                source_ids.append(int(folder[1:]))
    return sorted(source_ids)


def _count_images_in_s3(client: BaseClient, bucket: str, images_prefix: str) -> tuple[bool, int]:
    """Return (exists, count) for the images prefix in a single paginator pass."""
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=images_prefix):
        count += page.get("KeyCount", 0)
    return count > 0, count


def _check_other_csv(
    csv_reader: CsvReader,
    bucket: str,
    source_prefix: str,
    spec: _CsvSpec,
    record: dict[str, Any],
) -> str | None:
    key = f"{source_prefix}{spec.filename}"
    try:
        df = csv_reader(bucket, key)
    except Exception as e:  # noqa: BLE001 - CSV reader can raise various types
        # File exists but is unreadable, OR the read failed for some other
        # reason — flag the row as needing manual inspection.
        record[spec.has_field] = True
        record[spec.read_failed_field] = True
        return f"{spec.filename} read failed: {type(e).__name__}: {e}"

    if df is None:
        return None
    record[spec.has_field] = True
    count = len(df)
    record[spec.count_field] = count
    record[spec.empty_field] = count == 0
    return None


def _audit_one_source(
    client: BaseClient,
    csv_reader: CsvReader,
    bucket: str,
    prefix: str,
    source_id: int,
    audit_ts: datetime,
) -> dict[str, Any]:
    source_prefix = f"{prefix}/s{source_id}/"
    record = _empty_record(source_id, audit_ts)

    images_prefix = f"{source_prefix}images/"
    try:
        exists, count = _count_images_in_s3(client, bucket, images_prefix)
        record["has_images_folder"] = exists
        record["n_images_s3"] = count
    except Exception as e:  # noqa: BLE001
        record["errors"].append(f"Error checking images folder: {type(e).__name__}: {e}")

    annotations_key = f"{source_prefix}{_ANNOTATIONS_SPEC.filename}"
    try:
        df_annotations = csv_reader(bucket, annotations_key)
    except Exception as e:  # noqa: BLE001
        record["has_annotations_csv"] = True
        record["annotations_csv_read_failed"] = True
        record["errors"].append(f"annotations.csv read failed: {type(e).__name__}: {e}")
    else:
        if df_annotations is not None:
            record["has_annotations_csv"] = True
            count = len(df_annotations)
            record["n_annotations"] = count
            record["annotations_empty"] = count == 0
            if count > 0:
                for col in ("Name", "image", "image_id"):
                    if col in df_annotations.columns:
                        record["n_unique_images_annotated"] = int(df_annotations[col].nunique())
                        break

    for spec in _OTHER_SPECS:
        try:
            err = _check_other_csv(csv_reader, bucket, source_prefix, spec, record)
            if err:
                record["errors"].append(err)
        except Exception as e:  # noqa: BLE001
            record["errors"].append(f"Error reading {spec.filename}: {type(e).__name__}: {e}")

    record["is_complete"] = bool(
        record["has_images_folder"]
        and record["has_annotations_csv"]
        and record["has_image_list_csv"]
        and record["n_annotations"] > 0
    )
    record["image_count_match"] = record["n_images_s3"] == record["n_images_csv"]
    return record


def _get_thread_context() -> tuple[BaseClient, CsvReader]:
    """Lazy thread-local boto3 client + ibis-backed reader closure."""
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = make_s3_client()
    if not hasattr(_thread_local, "ibis_con"):
        _thread_local.ibis_con = make_ibis_connection()
        _thread_local.csv_reader = make_csv_reader_ibis(_thread_local.ibis_con)
    return _thread_local.s3, _thread_local.csv_reader


def _audit_worker(args: tuple[int, str, str, datetime]) -> dict[str, Any]:
    source_id, bucket, prefix, audit_ts = args
    client, reader = _get_thread_context()
    return _audit_one_source(client, reader, bucket, prefix, source_id, audit_ts)


def _normalise_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["errors"] = df["errors"].apply(lambda x: x if x else None)
    return df


def audit_sources(
    bucket: str | None = None,
    prefix: str | None = None,
    *,
    workers: int = DEFAULT_WORKERS,
    limit_sources: int | None = None,
    source_ids: list[int] | None = None,
    s3_client: BaseClient | None = None,
    csv_reader: CsvReader | None = None,
    credential_refresh_every: int = 100,
) -> pd.DataFrame:
    """Audit every CoralNet source folder under ``s3://<bucket>/<prefix>/``.

    Args:
        bucket / prefix: S3 location. Defaults from env vars.
        workers: ThreadPoolExecutor size. Each worker gets a thread-local
            boto3 client + ibis (DuckDB) connection. Forced to 1 when a custom
            ``s3_client`` or ``csv_reader`` is injected (tests).
        limit_sources: For dev runs, audit only the first N source folders.
        source_ids: Audit exactly these IDs instead of walking S3.
        s3_client: Pre-built boto3 client (for tests). Used for paginator /
            head_object / images-folder counts.
        csv_reader: ``CsvReader`` callable. When None and ``s3_client`` is
            also None, the production ibis path is used; when ``s3_client``
            is given, falls back to a boto3-backed reader against it.
        credential_refresh_every: Re-run the DuckDB ``CREATE OR REPLACE SECRET``
            on every worker's ibis connection after this many completed
            sources, to survive SageMaker IAM rotation.
    """
    bucket = bucket or get_bucket()
    prefix = prefix or get_prefix()
    workers = workers if workers > 0 else get_workers()
    audit_ts = datetime.now(UTC)
    client = s3_client or make_s3_client()

    injection_active = s3_client is not None or csv_reader is not None
    if csv_reader is None:
        csv_reader = (
            make_csv_reader_s3(client)
            if injection_active
            else make_csv_reader_ibis(make_ibis_connection())
        )

    if source_ids is None:
        logger.info("Listing source folders from s3://%s/%s", bucket, prefix)
        source_ids = list_source_folders(client, bucket, prefix)
    logger.info("Found %d source folders to audit", len(source_ids))

    if limit_sources is not None:
        source_ids = source_ids[:limit_sources]

    if not source_ids:
        logger.warning("No source folders to audit; returning empty DataFrame.")
        return pd.DataFrame(columns=[f.name for f in AUDIT_SCHEMA])

    results: list[dict[str, Any]] = []

    if workers > 1 and not injection_active:
        work = [(sid, bucket, prefix, audit_ts) for sid in source_ids]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_audit_worker, item): item[0] for item in work}
            for done in tqdm(as_completed(futures), total=len(futures), desc="Auditing sources"):
                source_id = futures[done]
                try:
                    record = done.result()
                except Exception as e:  # noqa: BLE001
                    logger.error("Source %d worker exception: %s", source_id, e)
                    record = _empty_record(source_id, audit_ts)
                    record["errors"].append(f"Worker exception: {type(e).__name__}: {e}")
                results.append(record)
                if len(results) % credential_refresh_every == 0:
                    refresh_thread_ibis_credentials_if_present(_thread_local)
    else:
        for source_id in tqdm(source_ids, desc="Auditing sources"):
            try:
                record = _audit_one_source(client, csv_reader, bucket, prefix, source_id, audit_ts)
            except Exception as e:  # noqa: BLE001
                logger.error("Source %d failed: %s", source_id, e)
                record = _empty_record(source_id, audit_ts)
                record["errors"].append(f"Worker exception: {type(e).__name__}: {e}")
            results.append(record)

    audit_df = _normalise_errors(pd.DataFrame(results))
    audit_df["source_id"] = audit_df["source_id"].astype("int32")
    if not pd.api.types.is_datetime64_any_dtype(audit_df["audit_timestamp"]):
        audit_df["audit_timestamp"] = pd.to_datetime(audit_df["audit_timestamp"], utc=True)

    complete = int(audit_df["is_complete"].sum())
    logger.info(
        "Audit complete: %d/%d sources are usable (%.1f%%); total S3 images: %d; total annotations: %d",
        complete,
        len(audit_df),
        100.0 * complete / max(1, len(audit_df)),
        int(audit_df["n_images_s3"].sum()),
        int(audit_df["n_annotations"].sum()),
    )
    return audit_df
