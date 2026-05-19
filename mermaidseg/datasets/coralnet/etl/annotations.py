"""Merge per-source CoralNet CSV pairs into a single annotations parquet.

For every source flagged ``is_complete`` in the audit DataFrame:
    1. Read ``annotations.csv`` (Name, Row, Column, Label ID, Status, ...)
    2. Read ``image_list.csv`` (Name, Image Page, ...)
    3. Strip the " - Confirmed" suffix from image_list ``Name`` values.
    4. Derive ``image_id`` from ``Image Page`` (CoralNet's URL fragment).
    5. Left-join on ``Name``, drop rows that fail to match (NaN image_id).
    6. Append a ``source_id`` column.

Concatenate per-source frames, sort, and validate against ``ANNOTATIONS_SCHEMA``.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import DEFAULT_WORKERS, get_bucket, get_prefix
from .io import (
    CsvReader,
    make_csv_reader_ibis,
    make_csv_reader_s3,
    make_ibis_connection,
)
from .schemas import ANNOTATIONS_PRIMARY_KEY, ANNOTATIONS_SCHEMA, validate

logger = logging.getLogger(__name__)

_thread_local = threading.local()

_REQUIRED_ANNOTATIONS_COLS = ("Name", "Row", "Column", "Label ID")
_REQUIRED_IMAGE_LIST_COLS = ("Name", "Image Page")


class _SourceLoadError(Exception):
    """Raised when a single source can't be read; the worker logs and skips."""


def _get_thread_reader() -> CsvReader:
    """Lazy thread-local ibis-backed reader closure."""
    if not hasattr(_thread_local, "ibis_con"):
        _thread_local.ibis_con = make_ibis_connection()
        _thread_local.csv_reader = make_csv_reader_ibis(_thread_local.ibis_con)
    return _thread_local.csv_reader


def _derive_image_id(image_page: pd.Series) -> pd.Series:
    """Extract CoralNet image ID from the ``Image Page`` URL fragment.

    The fragment looks like ``/image/<id>/view/`` (relative) or
    ``https://coralnet.ucsd.edu/image/<id>/view/``
    (absolute). We extract the
    digits between ``/image/`` and the next ``/`` so both forms produce the
    same numeric-string ID. Values that don't match become NaN and are dropped
    downstream.
    """
    return image_page.astype(str).str.extract(r"/image/(\d+)/?", expand=False)


def _load_source(csv_reader: CsvReader, bucket: str, prefix: str, source_id: int) -> pd.DataFrame:
    annotations_key = f"{prefix}/s{source_id}/annotations.csv"
    image_list_key = f"{prefix}/s{source_id}/image_list.csv"

    try:
        df_annotations = csv_reader(bucket, annotations_key)
        df_images = csv_reader(bucket, image_list_key)
    except Exception as e:  # noqa: BLE001 - reader can raise ClientError or DuckDB errors
        raise _SourceLoadError(
            f"csv read failed for source {source_id}: {type(e).__name__}: {e}"
        ) from e

    if df_annotations is None or df_images is None:
        raise _SourceLoadError(f"source {source_id} missing annotations.csv or image_list.csv")

    missing_ann = set(_REQUIRED_ANNOTATIONS_COLS) - set(df_annotations.columns)
    missing_img = set(_REQUIRED_IMAGE_LIST_COLS) - set(df_images.columns)
    if missing_ann:
        raise _SourceLoadError(
            f"source {source_id} annotations.csv missing columns: {sorted(missing_ann)}"
        )
    if missing_img:
        raise _SourceLoadError(
            f"source {source_id} image_list.csv missing columns: {sorted(missing_img)}"
        )

    # CoralNet image_list Names carry the per-image annotation status as a suffix:
    # "<basename>.jpg - Confirmed" / " - Unconfirmed" / " - Unclassified". We
    # capture that into a new column AND strip it from Name in a single regex
    # pass, so the merged parquet preserves status even when annotations.csv
    # lacks a Status column (most pre-2021 sources).
    name_parts = (
        df_images["Name"]
        .astype(str)
        .str.extract(r"^(?P<stripped>.+?)(?:\s-\s(?P<status>Confirmed|Unconfirmed|Unclassified))?$")
    )
    df_images = df_images.assign(
        Name=name_parts["stripped"],
        image_id=_derive_image_id(df_images["Image Page"]),
        image_status=name_parts["status"],
    )

    merged = df_annotations.merge(
        df_images[["Name", "image_id", "image_status"]], on="Name", how="left"
    )

    before = len(merged)
    merged = merged.dropna(subset=["image_id"]).reset_index(drop=True)
    after = len(merged)
    if before != after:
        logger.debug(
            "source %d: dropped %d annotations rows with no image_list match (kept %d)",
            source_id,
            before - after,
            after,
        )

    # Resolve status precedence: an explicit "Status" column on annotations.csv
    # (newer CoralNet exports) wins; otherwise fall back to the suffix captured
    # from image_list.csv (older exports). Both can be NaN if the source carries
    # neither, which is the only legitimate way ``status`` ends up null.
    if "Status" in merged.columns:
        status = merged["Status"].astype("string").fillna(merged["image_status"].astype("string"))
    else:
        status = merged["image_status"].astype("string")

    out = pd.DataFrame(
        {
            "source_id": source_id,
            "image_id": merged["image_id"].astype(str),
            "row": pd.to_numeric(merged["Row"], errors="coerce"),
            "col": pd.to_numeric(merged["Column"], errors="coerce"),
            "coralnet_id": pd.to_numeric(merged["Label ID"], errors="coerce"),
            "status": status,
        }
    )

    bad = out[["row", "col", "coralnet_id"]].isna().any(axis=1)
    if bad.any():
        logger.warning(
            "source %d: dropping %d rows with non-numeric row/col/Label ID",
            source_id,
            int(bad.sum()),
        )
        out = out.loc[~bad].reset_index(drop=True)

    out["row"] = out["row"].astype("int32")
    out["col"] = out["col"].astype("int32")
    out["coralnet_id"] = out["coralnet_id"].astype("int32")
    out["source_id"] = out["source_id"].astype("int32")
    return out


def _load_source_worker(args: tuple[int, str, str]) -> tuple[int, pd.DataFrame | None, str | None]:
    source_id, bucket, prefix = args
    reader = _get_thread_reader()
    try:
        df = _load_source(reader, bucket, prefix, source_id)
        return source_id, df, None
    except _SourceLoadError as e:
        return source_id, None, str(e)
    except Exception as e:  # noqa: BLE001 - we surface to caller, never crash worker
        return source_id, None, f"{type(e).__name__}: {e}"


def build_annotations(
    audit_df: pd.DataFrame,
    bucket: str | None = None,
    prefix: str | None = None,
    *,
    workers: int = DEFAULT_WORKERS,
    s3_client: Any | None = None,
    csv_reader: CsvReader | None = None,
) -> pd.DataFrame:
    """Merge per-source CSV pairs for every ``is_complete`` source into one DataFrame.

    Args:
        audit_df: Output of :func:`audit_sources`. Only sources where
            ``is_complete`` is True are included.
        bucket / prefix: S3 location. Defaults from env vars.
        workers: ThreadPoolExecutor size. Forced to 1 when ``s3_client`` or
            ``csv_reader`` is injected (tests).
        s3_client: Pre-built boto3 client (tests). Implies a boto3-backed reader.
        csv_reader: ``CsvReader`` callable. When None and ``s3_client`` is also
            None, the production ibis path is used (per-thread DuckDB
            connection); when ``s3_client`` is given, falls back to a
            boto3-backed reader against it.

    Returns:
        DataFrame conforming to :data:`ANNOTATIONS_SCHEMA`, sorted by primary key.
    """
    bucket = bucket or get_bucket()
    prefix = prefix or get_prefix()

    if "is_complete" not in audit_df.columns or "source_id" not in audit_df.columns:
        raise ValueError("audit_df must contain 'source_id' and 'is_complete' columns")

    injection_active = s3_client is not None or csv_reader is not None
    if csv_reader is None:
        csv_reader = make_csv_reader_s3(s3_client) if s3_client is not None else None

    ok = audit_df.loc[audit_df["is_complete"].astype(bool), "source_id"].astype(int).tolist()
    if not ok:
        logger.warning("No sources flagged is_complete in audit_df; returning empty frame.")
        return pd.DataFrame(
            {field.name: pd.Series(dtype=str(field.type)) for field in ANNOTATIONS_SCHEMA}
        )

    logger.info("Building annotations parquet from %d sources (workers=%d)", len(ok), workers)

    frames: list[pd.DataFrame] = []
    failures: list[tuple[int, str]] = []

    if workers > 1 and not injection_active:
        work = [(sid, bucket, prefix) for sid in ok]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_load_source_worker, item) for item in work]
            for done in tqdm(as_completed(futures), total=len(futures), desc="Reading sources"):
                source_id, df, err = done.result()
                if err:
                    failures.append((source_id, err))
                    continue
                if df is not None and not df.empty:
                    frames.append(df)
    else:
        if csv_reader is None:
            csv_reader = make_csv_reader_ibis(make_ibis_connection())
        for source_id in tqdm(ok, desc="Reading sources"):
            try:
                df = _load_source(csv_reader, bucket, prefix, source_id)
            except _SourceLoadError as e:
                failures.append((source_id, str(e)))
                continue
            if not df.empty:
                frames.append(df)

    if failures:
        logger.warning("build_annotations: %d source(s) failed to load", len(failures))
        for sid, err in failures[:10]:
            logger.warning("  source %d: %s", sid, err)
        if len(failures) > 10:
            logger.warning("  ... and %d more", len(failures) - 10)

    if not frames:
        logger.warning("No annotation frames produced; returning empty.")
        return pd.DataFrame(
            {field.name: pd.Series(dtype=str(field.type)) for field in ANNOTATIONS_SCHEMA}
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(list(ANNOTATIONS_PRIMARY_KEY), kind="mergesort").reset_index(
        drop=True
    )

    validate(merged, ANNOTATIONS_SCHEMA)
    logger.info(
        "Annotations build complete: %d rows across %d sources",
        len(merged),
        merged["source_id"].nunique(),
    )
    return merged


def derive_unique_images(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique ``(source_id, image_id)`` in ``annotations_df``.

    Helper for ``build_images`` so it doesn't need to re-read the parquet just to enumerate keys.
    """
    return (
        annotations_df[["source_id", "image_id"]]
        .drop_duplicates(subset=["source_id", "image_id"])
        .reset_index(drop=True)
    )
