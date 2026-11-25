"""Merge per-source CoralNet CSVs into one annotations frame and one label-name map.

For every requested source:
    0. Require the scraper's ``_complete.json`` marker; a source without it is excluded as
       ``not_complete`` before any CSV is read.
    1. Read ``annotations.csv`` (Name, Row, Column, Label ID, Status, ...), ``image_list.csv``
       (Name, Image Page, ...) and ``labelset.csv`` (Label ID, Name).
    2. Strip the " - Confirmed" suffix from image_list ``Name`` values.
    3. Derive ``image_id`` from ``Image Page`` (CoralNet's URL fragment).
    4. Left-join on ``Name``; an annotated ``Name`` with no image_list row excludes the source.
    5. Keep only rows with ``status == "Confirmed"``.
    6. Append a ``source_id`` column.

A source that fails a check is excluded under exactly one kind (see :data:`EXCLUSION_KINDS`).
The included sources' labelsets are merged into ``id2name``; a source whose labels are not all
named there is excluded as ``label_unnamed``.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from coralnet_scraper.s3_io import (
    object_key_annotation_csv,
    object_key_complete_marker,
    object_key_image_list_csv,
    object_key_labelset_csv,
)

from ...common.s3_objects import make_s3_client
from .config import DEFAULT_WORKERS
from .io import (
    CsvReader,
    MarkerCheck,
    make_csv_reader_ibis,
    make_csv_reader_s3,
    make_ibis_connection,
    make_marker_check,
)
from .schemas import ANNOTATIONS_PRIMARY_KEY, ANNOTATIONS_SCHEMA, validate

logger = logging.getLogger(__name__)

_thread_local = threading.local()

_REQUIRED_ANNOTATIONS_COLS = ("Name", "Row", "Column", "Label ID")
_REQUIRED_IMAGE_LIST_COLS = ("Name", "Image Page")
_REQUIRED_LABELSET_COLS = ("Label ID", "Name")

# An image_list Name: the basename, then an optional " - <status>" suffix.
_IMAGE_LIST_NAME = r"^(?P<stripped>.+?)(?:\s-\s(?P<status>Confirmed|Unconfirmed|Unclassified))?$"

# (file stem, key builder, required columns), in the order the per-source checks run.
_SOURCE_CSVS: tuple[tuple[str, Callable[[str, int], str], tuple[str, ...]], ...] = (
    ("annotations", object_key_annotation_csv, _REQUIRED_ANNOTATIONS_COLS),
    ("image_list", object_key_image_list_csv, _REQUIRED_IMAGE_LIST_COLS),
    ("labelset", object_key_labelset_csv, _REQUIRED_LABELSET_COLS),
)

EXCLUSION_KINDS: tuple[str, ...] = (
    "not_complete",
    "annotations_missing",
    "image_list_missing",
    "labelset_missing",
    "csv_read_failed",
    "columns_missing",
    "annotations_empty",
    "image_list_incomplete",
    "label_unnamed",
)


@dataclass(frozen=True)
class AnnotationsBuild:
    """Result of :func:`build_annotations`.

    Attributes:
        annotations: Rows conforming to :data:`ANNOTATIONS_SCHEMA`, sorted by primary key.
        id2name: Label ID (as a string) to label name, the union of the included sources'
            ``labelset.csv``.
        exclusions: Exclusion kind to the sorted ids of the sources excluded under it.
    """

    annotations: pd.DataFrame
    id2name: dict[str, str]
    exclusions: dict[str, list[int]]


class _SourceExcluded(Exception):
    """Raised when a source fails a check; ``kind`` is one of :data:`EXCLUSION_KINDS`."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


@dataclass(frozen=True)
class _SourceResult:
    """One included source's Confirmed rows and its labelset ``(Label ID, Name)`` pairs."""

    rows: pd.DataFrame
    labels: list[tuple[int, str]]


def _get_thread_reader(client: Any) -> CsvReader:
    """Return an ibis-backed reader on this thread's DuckDB connection and the shared client."""
    if not hasattr(_thread_local, "ibis_con"):
        _thread_local.ibis_con = make_ibis_connection()
    return make_csv_reader_ibis(_thread_local.ibis_con, client)


def list_source_folders(client: Any, bucket: str, prefix: str) -> list[int]:
    """Return the numeric source IDs of the ``s<N>/`` folders directly under ``prefix``.

    Folders that are not ``s`` followed by digits (e.g. ``temporary/``, ``resized/``) are ignored.

    Args:
        client: boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix holding the per-source folders.

    Returns:
        Source IDs in ascending order.
    """
    full_prefix = f"{prefix}/" if not prefix.endswith("/") else prefix
    paginator = client.get_paginator("list_objects_v2")
    source_ids: list[int] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            folder = common["Prefix"].replace(full_prefix, "").rstrip("/")
            if folder.startswith("s") and folder[1:].isdigit():
                source_ids.append(int(folder[1:]))
    return sorted(source_ids)


def derive_image_id(image_page: pd.Series) -> pd.Series:
    """Extract CoralNet image ID from the ``Image Page`` URL fragment.

    The fragment looks like ``/image/<id>/view/`` (relative) or
    ``https://coralnet.ucsd.edu/image/<id>/view/``
    (absolute). We extract the
    digits between ``/image/`` and the next ``/`` so both forms produce the
    same numeric-string ID. Values that don't match become NaN and are dropped
    downstream.
    """
    return image_page.astype(str).str.extract(r"/image/(\d+)/?", expand=False)


def _read_source_csvs(
    csv_reader: CsvReader, bucket: str, prefix: str, source_id: int
) -> dict[str, pd.DataFrame]:
    """Read a source's three CSVs, keyed by file stem.

    Raises:
        _SourceExcluded: ``<stem>_missing`` for the first missing CSV, else ``csv_read_failed``
            when a read raised.
    """
    frames: dict[str, pd.DataFrame | None] = {}
    read_errors: list[str] = []
    for stem, key_for, _required in _SOURCE_CSVS:
        try:
            frames[stem] = csv_reader(bucket, key_for(prefix, source_id))
        except Exception as e:  # noqa: BLE001 - reader can raise ClientError or DuckDB errors
            read_errors.append(f"{stem}.csv: {type(e).__name__}: {e}")
    for stem, _key_for, _required in _SOURCE_CSVS:
        if stem in frames and frames[stem] is None:
            raise _SourceExcluded(f"{stem}_missing", f"{stem}.csv not found")
    if read_errors:
        raise _SourceExcluded("csv_read_failed", "; ".join(read_errors))
    return {stem: df for stem, df in frames.items() if df is not None}


def _check_source_csvs(frames: dict[str, pd.DataFrame]) -> None:
    """Check the required columns, then that ``annotations.csv`` has rows.

    Raises:
        _SourceExcluded: ``columns_missing`` or ``annotations_empty``.
    """
    for stem, _key_for, required in _SOURCE_CSVS:
        missing = set(required) - set(frames[stem].columns)
        if missing:
            raise _SourceExcluded("columns_missing", f"{stem}.csv lacks {sorted(missing)}")
    if frames["annotations"].empty:
        raise _SourceExcluded("annotations_empty", "annotations.csv has no rows")


def _parse_labelset(df_labelset: pd.DataFrame, source_id: int) -> list[tuple[int, str]]:
    """Return the labelset's ``(Label ID, Name)`` pairs, dropping non-numeric ids."""
    ids = pd.to_numeric(df_labelset["Label ID"], errors="coerce")
    bad = ids.isna()
    if bad.any():
        logger.warning(
            "source %d: dropping %d labelset rows with a non-numeric Label ID",
            source_id,
            int(bad.sum()),
        )
    names = df_labelset["Name"].astype(str)
    return [(int(i), n) for i, n in zip(ids[~bad], names[~bad], strict=True)]


def _check_source_complete(
    marker_check: MarkerCheck, bucket: str, prefix: str, source_id: int
) -> None:
    """Check that the scraper's completion marker exists for the source.

    Raises:
        _SourceExcluded: ``not_complete`` when the marker is absent.
    """
    key = object_key_complete_marker(prefix, source_id)
    if not marker_check(bucket, key):
        raise _SourceExcluded("not_complete", f"{key} not found")


def _load_source(
    csv_reader: CsvReader,
    marker_check: MarkerCheck,
    bucket: str,
    prefix: str,
    source_id: int,
) -> _SourceResult:
    """Read and check one source, returning its Confirmed rows and its labelset.

    Raises:
        _SourceExcluded: If the source fails a check, with the kind of the first failed check.
    """
    _check_source_complete(marker_check, bucket, prefix, source_id)
    frames = _read_source_csvs(csv_reader, bucket, prefix, source_id)
    _check_source_csvs(frames)
    rows = _merge_source(frames["annotations"], frames["image_list"], source_id)
    return _SourceResult(rows=rows, labels=_parse_labelset(frames["labelset"], source_id))


def _merge_source(
    df_annotations: pd.DataFrame, df_images: pd.DataFrame, source_id: int
) -> pd.DataFrame:
    """Join one source's annotations to its image list and keep the Confirmed rows.

    Raises:
        _SourceExcluded: ``image_list_incomplete`` if an annotation row matches no image.
    """
    # CoralNet image_list Names carry the per-image annotation status as a suffix:
    # "<basename>.jpg - Confirmed" / " - Unconfirmed" / " - Unclassified". We
    # capture that into a new column AND strip it from Name in a single regex
    # pass, so the merged parquet preserves status even when annotations.csv
    # lacks a Status column (most pre-2021 sources).
    name_parts = df_images["Name"].astype(str).str.extract(_IMAGE_LIST_NAME)
    df_images = df_images.assign(
        Name=name_parts["stripped"],
        image_id=derive_image_id(df_images["Image Page"]),
        image_status=name_parts["status"],
    )

    merged = df_annotations.merge(
        df_images[["Name", "image_id", "image_status"]], on="Name", how="left"
    )

    unmatched = merged["image_id"].isna()
    if unmatched.any():
        raise _SourceExcluded(
            "image_list_incomplete",
            f"{int(unmatched.sum())} annotation rows match no image_list row",
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

    before_confirmed = len(out)
    out = out.loc[out["status"] == "Confirmed"].reset_index(drop=True)
    if before_confirmed and len(out) < before_confirmed:
        logger.debug(
            "source %d: kept %d/%d annotation rows with Confirmed status",
            source_id,
            len(out),
            before_confirmed,
        )

    out["row"] = out["row"].astype("int32")
    out["col"] = out["col"].astype("int32")
    out["coralnet_id"] = out["coralnet_id"].astype("int32")
    out["source_id"] = out["source_id"].astype("int32")
    return out


_Outcome = _SourceResult | _SourceExcluded


def _load_source_outcome(
    csv_reader: CsvReader, marker_check: MarkerCheck, bucket: str, prefix: str, source_id: int
) -> _Outcome:
    """Load one source, returning its exclusion instead of raising it."""
    try:
        return _load_source(csv_reader, marker_check, bucket, prefix, source_id)
    except _SourceExcluded as e:
        return e


def _load_source_worker(args: tuple[int, Any, MarkerCheck, str, str]) -> tuple[int, _Outcome]:
    """Load one source on a worker thread with that thread's ibis connection."""
    source_id, client, marker_check, bucket, prefix = args
    outcome = _load_source_outcome(
        _get_thread_reader(client), marker_check, bucket, prefix, source_id
    )
    return source_id, outcome


def _sources_with_unnamed_labels(
    used: dict[int, set[int]], named: dict[int, set[int]]
) -> list[int]:
    """Return the sources to exclude because a label they use is named by no included source.

    Excluding a source removes its labelset from the union, which can unname another source's
    labels, so this repeats until no further source is excluded.
    """
    included = set(used)
    while True:
        union = set().union(*(named[sid] for sid in included))
        unnamed = {sid for sid in included if not used[sid] <= union}
        if not unnamed:
            return sorted(set(used) - included)
        included -= unnamed


def _merge_labelsets(labels: dict[int, list[tuple[int, str]]]) -> dict[str, str]:
    """Return the union of the labelsets as ``str(Label ID) -> Name``.

    Raises:
        ValueError: If a Label ID has two different names.
    """
    id2name: dict[str, str] = {}
    for sid in sorted(labels):
        for label_id, name in labels[sid]:
            existing = id2name.setdefault(str(label_id), name)
            if existing != name:
                raise ValueError(
                    f"Label ID {label_id} has conflicting names {existing!r} and {name!r} "
                    f"(source {sid})"
                )
    return id2name


def _load_sources(
    source_ids: list[int],
    bucket: str,
    prefix: str,
    workers: int,
    csv_reader: CsvReader | None,
    client: Any,
    marker_check: MarkerCheck,
) -> dict[int, _Outcome]:
    """Load every source with the shared ``client``, on a thread pool unless a reader is injected."""
    if workers > 1 and csv_reader is None:
        work = [(sid, client, marker_check, bucket, prefix) for sid in source_ids]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_load_source_worker, item) for item in work]
            done = tqdm(as_completed(futures), total=len(futures), desc="Reading sources")
            try:
                return dict(future.result() for future in done)
            except BaseException:
                # Cancel the queued sources so the error surfaces once the running ones finish.
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    reader = csv_reader or make_csv_reader_ibis(make_ibis_connection(), client)
    return {
        sid: _load_source_outcome(reader, marker_check, bucket, prefix, sid)
        for sid in tqdm(source_ids, desc="Reading sources")
    }


def build_annotations(
    source_ids: list[int],
    bucket: str,
    prefix: str,
    *,
    workers: int = DEFAULT_WORKERS,
    s3_client: Any | None = None,
    csv_reader: CsvReader | None = None,
) -> AnnotationsBuild:
    """Merge the CSVs of ``source_ids`` into one annotations frame and one label-name map.

    Args:
        source_ids: CoralNet source IDs to read.
        bucket: S3 bucket.
        prefix: Key prefix holding the ``s<N>/`` folders.
        workers: ThreadPoolExecutor size. Forced to 1 when ``s3_client`` or
            ``csv_reader`` is injected (tests).
        s3_client: Pre-built boto3 client (tests). Implies a boto3-backed reader, and probes the
            completion markers; without it a default client probes them.
        csv_reader: ``CsvReader`` callable. When None and ``s3_client`` is also
            None, the production ibis path is used (per-thread DuckDB
            connection); when ``s3_client`` is given, falls back to a
            boto3-backed reader against it.

    Returns:
        The included sources' annotations and label names, and the excluded sources by kind.

    Raises:
        ValueError: If two included sources give one Label ID different names.
    """
    if csv_reader is None and s3_client is not None:
        csv_reader = make_csv_reader_s3(s3_client)
    logger.info("Building annotations from %d sources (workers=%d)", len(source_ids), workers)
    client = s3_client or make_s3_client(workers=workers)
    marker_check = make_marker_check(client)
    outcomes = _load_sources(
        list(source_ids), bucket, prefix, workers, csv_reader, client, marker_check
    )

    exclusions: dict[str, list[int]] = defaultdict(list)
    results: dict[int, _SourceResult] = {}
    for sid in sorted(outcomes):
        outcome = outcomes[sid]
        if isinstance(outcome, _SourceExcluded):
            logger.warning("source %d excluded: %s", sid, outcome)
            exclusions[outcome.kind].append(sid)
        else:
            results[sid] = outcome

    unnamed = _sources_with_unnamed_labels(
        {sid: set(r.rows["coralnet_id"].tolist()) for sid, r in results.items()},
        {sid: {label_id for label_id, _ in r.labels} for sid, r in results.items()},
    )
    for sid in unnamed:
        logger.warning("source %d excluded: label_unnamed", sid)
        exclusions["label_unnamed"].append(sid)
        del results[sid]

    return AnnotationsBuild(
        annotations=_concat_rows([r.rows for r in results.values()]),
        id2name=_merge_labelsets({sid: r.labels for sid, r in results.items()}),
        exclusions=dict(exclusions),
    )


def _concat_rows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-source rows, sorted by primary key; empty frames are skipped."""
    frames = [df for df in frames if not df.empty]
    if not frames:
        logger.warning("No annotation rows produced; returning empty frame.")
        return ANNOTATIONS_SCHEMA.empty_table().to_pandas()
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
