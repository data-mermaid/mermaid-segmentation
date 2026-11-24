"""``segdata build coralnet raw``: publish a frozen CoralNet build as a ``coralnet-raw`` version.

The build's annotations, images and id2name tables become the version's points, images and labels.
Each kept original is copied server side into the segdata image store. A rerun skips keys the store
already holds with the source's ETag, taking their CRC64NVME from a local checkpoint or a HEAD, and
a parent pin carries forward the parent's objects for images whose source is unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import shlex
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from coralnet_scraper.config import get_bucket, get_prefix, validate_workers
from coralnet_scraper.s3_io import object_key_image_list_csv
from segdata_etl.common.cli_flags import pin_command_parts
from segdata_etl.common.copy import CopyItem, copy_images
from segdata_etl.common.jpeg_header import STATUSES_WITH_DIMENSIONS
from segdata_etl.common.publishing import (
    checked_git_state,
    count_labels,
    image_prefix,
    image_store,
    pick_version,
)
from segdata_etl.common.s3_objects import EXTENSIONS, sniff_media_type
from segdata_etl.common.store import DEFAULT_STORE, open_store
from segdata_etl.common.version import utc_today
from segdata_etl.taxonomy import find_repo_root
from segdata_spec.errors import ObjectExists
from segdata_spec.manifest import ParentRef, SourceInfo
from segdata_spec.publish import fetch_parent_tables, publish_dataset
from segdata_spec.schema import IMAGES_SCHEMA, LABELS_SCHEMA, POINTS_SCHEMA

from .annotations import _IMAGE_LIST_NAME, _derive_image_id
from .config import DEFAULT_OUTPUT_S3_PREFIX, get_workers
from .io import CsvReader, make_csv_reader_s3, make_s3_client

logger = logging.getLogger(__name__)

DATASET = "coralnet-raw"
POOL_HEADROOM = 8
# The images live under the "coralnet" family, not the "coralnet-raw" dataset name.
_IMAGE_FAMILY = "coralnet"
IMAGE_PREFIX = image_prefix(_IMAGE_FAMILY)
STORAGE_CLASS = "INTELLIGENT_TIERING"
ANNOTATOR = "default"
# CoralNet label 2474 is named the literal string "None", which pandas read as NaN into id2name.
_NONE_NAMED_LABEL = "2474"
# Copies per batch, the unit of progress between checkpoint writes.
_CHECKPOINT_BATCH = 1000
# Batches per checkpoint write: a crash loses at most this many batches of finished copies.
_CHECKPOINT_EVERY = 20
# Bytes read to recognise a PNG signature, the longer of the two supported magic numbers.
_SNIFF_BYTES = 8
_COPIED = ["etag", "size", "crc64nvme"]
_CHECKPOINT_COLUMNS = ["dest_key", "source_fingerprint", *_COPIED]


@dataclass(frozen=True)
class BuildTables:
    """A CoralNet build run's annotations and images frames and its id2name map."""

    annotations: pd.DataFrame
    images: pd.DataFrame
    id2name: dict[str, Any]


@dataclass(frozen=True)
class RawTables:
    """The ``coralnet-raw`` tables keyed by file name, and the dropped counts by reason."""

    tables: dict[str, pa.Table]
    dropped: dict[str, int]


def _label_name(label_id: str, value: Any) -> str:
    if isinstance(value, str):
        return value
    if label_id == _NONE_NAMED_LABEL and isinstance(value, float) and math.isnan(value):
        return "None"
    raise ValueError(f"id2name holds a non-string name for label {label_id}: {value!r}")


def labels_table(id2name: dict[str, Any]) -> pa.Table:
    """One identity-mapped label row per id2name entry, sorted by id.

    Raises:
        ValueError: A name other than label 2474's NaN is not a string.
    """
    rows = [
        {
            "label_id": int(label_id),
            "name": _label_name(label_id, value),
            "mermaid_benthic_attribute_id": None,
            "mermaid_benthic_attribute": None,
            "mermaid_growth_form": None,
            "mapping_source": "identity",
        }
        for label_id, value in sorted(id2name.items(), key=lambda item: int(item[0]))
    ]
    return pa.Table.from_pylist(rows, schema=LABELS_SCHEMA)


def _select_images(build: BuildTables) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return the annotated images with dimensions, their annotations, and the dropped counts."""
    images, annotations = build.images, build.annotations
    has_dims = images["header_status"].astype(str).isin(STATUSES_WITH_DIMENSIONS)
    keys = ["source_id", "image_id"]
    listed = annotations.merge(images[keys], on=keys, how="left", indicator=True)
    with_image = (listed["_merge"] == "both").to_numpy()
    with_dims = images[has_dims]
    kept = with_dims[with_dims["image_id"].isin(annotations["image_id"][with_image])]
    points = annotations[annotations["image_id"].isin(kept["image_id"]) & with_image]
    dropped = {
        "no_dimensions": int((~has_dims).sum()),
        "no_annotations": len(with_dims) - len(kept),
        "annotations_without_image": int((~with_image).sum()),
    }
    return kept, points, dropped


def _source_names(image_list: pd.DataFrame) -> pd.DataFrame:
    """Map image_id to the image_list Name without its status suffix, as the build derives it.

    Raises:
        ValueError: One image_id carries two different names.
    """
    names = pd.DataFrame(
        {
            "image_id": _derive_image_id(image_list["Image Page"]),
            "source_name": image_list["Name"].astype(str).str.extract(_IMAGE_LIST_NAME)["stripped"],
        }
    ).dropna(subset=["image_id"])
    names = names.drop_duplicates()
    repeated = names["image_id"][names["image_id"].duplicated()]
    if not repeated.empty:
        raise ValueError(f"image_list names image ids twice: {sorted(set(repeated))[:10]}")
    return names


def _attach_sources(
    kept: pd.DataFrame, fingerprints: dict[str, str], names: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Add each image's source ETag and name; return the rows and the ids with no source object."""
    rows = kept.assign(source_fingerprint=kept["s3_key"].map(fingerprints))
    missing = rows["source_fingerprint"].isna()
    rows = rows[~missing].merge(names, on="image_id", how="left")
    return rows, sorted(kept["image_id"][missing].tolist())


def _image_key(source_id: int, image_id: str, media_type: str) -> str:
    return f"s{source_id}/{image_id}.{EXTENSIONS[media_type]}"


def _carry_forward(rows: pd.DataFrame, parent: pa.Table | None) -> pd.DataFrame:
    """Take the parent's key, media type and object fields for rows whose source is unchanged.

    Rows the parent lacks get null in those fields.
    """
    columns = ["image_id", "source_fingerprint", "key", "media_type", *_COPIED]
    if parent is None:
        parent_rows = pd.DataFrame({name: pd.Series(dtype=object) for name in columns})
    else:
        parent_rows = parent.select(columns).to_pandas()
    return rows.merge(parent_rows, on=["image_id", "source_fingerprint"], how="left")


def _fill_copied(rows: pd.DataFrame, done: pd.DataFrame) -> pd.DataFrame:
    """Fill the object fields of rows not yet filled from ``done``, matched by key and source."""
    match = rows.merge(
        done[_CHECKPOINT_COLUMNS],
        left_on=[IMAGE_PREFIX + rows["key"], "source_fingerprint"],
        right_on=["dest_key", "source_fingerprint"],
        how="left",
        suffixes=("", "_done"),
    )
    open_rows = match["etag"].isna()
    for column in _COPIED:
        match[column] = match[column].astype(object).where(~open_rows, match[f"{column}_done"])
    return match.drop(columns=[f"{c}_done" for c in _COPIED] + ["dest_key"])


def _images_table(rows: pd.DataFrame) -> pa.Table:
    frame = pd.DataFrame(
        {
            "image_id": rows["image_id"].astype(str),
            "key": rows["key"],
            "etag": rows["etag"],
            "size": rows["size"].astype("int64"),
            "crc64nvme": rows["crc64nvme"].where(rows["crc64nvme"].notna(), None),
            "media_type": rows["media_type"],
            "width": rows["width"].astype("int32"),
            "height": rows["height"].astype("int32"),
            "group": "s" + rows["source_id"].astype(str),
            "mask_key": None,
            "mask_etag": None,
            "mask_size": None,
            "mask_crc64nvme": None,
            "source_fingerprint": rows["source_fingerprint"],
            "source_name": rows["source_name"].where(rows["source_name"].notna(), None),
            "attrs": [
                json.dumps({"source_id": int(sid), "source_key": key})
                for sid, key in zip(rows["source_id"], rows["s3_key"], strict=True)
            ],
        }
    ).sort_values("image_id", kind="mergesort")
    return pa.Table.from_pandas(frame, schema=IMAGES_SCHEMA, preserve_index=False)


def _points_and_counts(points: pd.DataFrame) -> tuple[pa.Table, pa.Table]:
    frame = pd.DataFrame(
        {
            "image_id": points["image_id"].astype(str),
            "row": points["row"],
            "col": points["col"],
            "label_id": points["coralnet_id"],
            "annotator": ANNOTATOR,
        }
    ).sort_values(["image_id", "row", "col"], kind="mergesort")
    points_table = pa.Table.from_pandas(frame, schema=POINTS_SCHEMA, preserve_index=False)
    return points_table, count_labels(points_table)


def _list_source_etags(client: Any, bucket: str, prefix: str, source_id: int) -> dict[str, str]:
    """Return ``{key: ETag}`` of every object under the source's ``images/`` folder."""
    etags: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/s{source_id}/images/"):
        etags.update({entry["Key"]: entry["ETag"] for entry in page.get("Contents", [])})
    return etags


def _string_csv_reader(client: Any) -> CsvReader:
    """A boto3 reader yielding every cell as a string, as the DuckDB ``all_varchar`` reader does."""
    return make_csv_reader_s3(client, dtype=str, keep_default_na=False, index_col=False)


def _read_source(
    client: Any, reader: CsvReader, bucket: str, prefix: str, source_id: int
) -> tuple[dict[str, str], pd.DataFrame | None]:
    """Return the source's ``{key: ETag}`` listing and its image_list names, None if unlisted."""
    image_list = reader(bucket, object_key_image_list_csv(prefix, source_id))
    if image_list is None:
        logger.warning("source %d: image_list.csv is missing; source names stay null", source_id)
        names = None
    else:
        names = _source_names(image_list)
    return _list_source_etags(client, bucket, prefix, source_id), names


def _gather_sources(
    client: Any,
    bucket: str,
    prefix: str,
    source_ids: Sequence[int],
    *,
    workers: int,
    csv_reader: CsvReader | None,
) -> tuple[dict[str, str], pd.DataFrame, list[int]]:
    """Return the merged ``{key: ETag}`` listings and names, and the sources with no image_list.

    An injected ``csv_reader`` reads the sources in turn; otherwise the worker threads read
    through the shared boto3 client, never DuckDB, whose aws extension crashes when many threads
    open connections at once.
    """
    if csv_reader is not None:
        results = [_read_source(client, csv_reader, bucket, prefix, sid) for sid in source_ids]
    else:
        reader = _string_csv_reader(client)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(
                    lambda sid: _read_source(client, reader, bucket, prefix, sid),
                    source_ids,
                )
            )
    etags = {key: etag for listing, _names in results for key, etag in listing.items()}
    empty = pd.DataFrame({"image_id": pd.Series(dtype=str), "source_name": []})
    names = pd.concat(
        [empty, *(names for _listing, names in results if names is not None)], ignore_index=True
    )
    unlisted = [
        sid for sid, (_listing, names) in zip(source_ids, results, strict=True) if names is None
    ]
    return etags, names, unlisted


def _media_type(client: Any, bucket: str, key: str, header_status: str) -> str | None:
    """JPEG for a header the JPEG parser read; otherwise sniffed from the first bytes.

    None when the sniffed bytes are neither JPEG nor PNG.
    """
    if header_status == "ok":
        return "image/jpeg"
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{_SNIFF_BYTES - 1}")
    return sniff_media_type(response["Body"].read())


def _fill_media_types(client: Any, bucket: str, rows: pd.DataFrame, workers: int) -> pd.Series:
    """Each row's media type: the carried one, else from its header status or a sniff."""
    open_rows = rows[rows["media_type"].isna()]
    pairs = zip(open_rows["s3_key"], open_rows["header_status"], strict=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = list(
            pool.map(lambda pair: _media_type(client, bucket, pair[0], str(pair[1])), pairs)
        )
    media = rows["media_type"].astype(object)
    media[open_rows.index] = pd.Series(found, index=open_rows.index, dtype=object)
    return media


def _count_unnamed(rows: pd.DataFrame, unlisted: list[int]) -> int:
    """Count the rows with no source name, logging those whose source has an image_list."""
    unnamed = rows["source_name"].isna()
    absent = unnamed & ~rows["source_id"].isin(unlisted)
    if absent.any():
        logger.warning(
            "%d images are absent from their image_list.csv; source names stay null for ids: %s",
            int(absent.sum()),
            sorted(rows["image_id"][absent].astype(str).tolist()),
        )
    return int(unnamed.sum())


def _read_checkpoint(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame({name: pd.Series(dtype=object) for name in _CHECKPOINT_COLUMNS})
    return pd.read_parquet(path, columns=_CHECKPOINT_COLUMNS)


def _write_checkpoint(path: Path, finished: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    latest = finished.drop_duplicates("dest_key", keep="last")
    latest[_CHECKPOINT_COLUMNS].to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _list_dest_objects(client: Any, bucket: str) -> pd.DataFrame:
    """Return the ``dest_key``, ``etag`` and ``size`` of every object under ``IMAGE_PREFIX``."""
    entries: list[tuple[str, str, int]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=IMAGE_PREFIX):
        entries += [(e["Key"], e["ETag"], e["Size"]) for e in page.get("Contents", [])]
    return pd.DataFrame(entries, columns=["dest_key", "etag", "size"])


def _match_held(rows: pd.DataFrame, held: pd.DataFrame, checkpoint: pd.DataFrame) -> pd.DataFrame:
    """Fill the object fields of unfilled rows whose key ``held`` lists with a known object.

    A held key is known when its ETag is the row's source fingerprint, or when a checkpoint row
    with the same key and source fingerprint recorded that ETag; the CRC comes from that
    checkpoint row and stays null without one. A copy rewrites a multipart original as one part,
    so a held key with another ETag and a multipart fingerprint stays unfilled for the copy's
    content check.

    Raises:
        ObjectExists: A listed key's ETag differs from its row's single-part source fingerprint.
    """
    unfilled = rows["etag"].isna().to_numpy()
    listed = held.rename(columns={"etag": "held_etag", "size": "held_size"})
    keys = ["dest_key", "source_fingerprint", "held_etag"]
    recorded = checkpoint.rename(columns={"etag": "held_etag", "crc64nvme": "checkpoint_crc"})
    recorded = (
        recorded[[*keys, "checkpoint_crc"]]
        .drop_duplicates(keys, keep="last")
        .assign(checkpointed=True)
    )
    merged = (
        rows.assign(dest_key=IMAGE_PREFIX + rows["key"])
        .merge(listed, on="dest_key", how="left")
        .merge(recorded, on=keys, how="left")
    )
    present = unfilled & merged["held_etag"].notna().to_numpy()
    same = (merged["held_etag"] == merged["source_fingerprint"]).to_numpy()
    known = present & (same | merged["checkpointed"].notna().to_numpy())
    multipart = merged["source_fingerprint"].astype(str).str.contains("-", regex=False).to_numpy()
    foreign = present & ~known & ~multipart
    if foreign.any():
        held_keys = sorted(merged["dest_key"][foreign])
        raise ObjectExists(
            f"{len(held_keys)} destination keys hold other objects: {held_keys[:10]}"
        )
    for column, source in (("etag", "held_etag"), ("size", "held_size")):
        merged[column] = merged[column].astype(object).where(~known, merged[source])
    merged["crc64nvme"] = merged["crc64nvme"].astype(object).where(~known, merged["checkpoint_crc"])
    return merged.drop(
        columns=["dest_key", "held_etag", "held_size", "checkpoint_crc", "checkpointed"]
    )


def _head_crcs(client: Any, bucket: str, keys: list[str], workers: int) -> list[str | None]:
    """Return each key's CRC64NVME from a HEAD with checksum mode enabled."""

    def head(key: str) -> str | None:
        response = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        return response.get("ChecksumCRC64NVME")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(head, keys))


def _resolve_held(
    store_client: Any,
    rows: pd.DataFrame,
    checkpoint: pd.DataFrame,
    *,
    dest_bucket: str,
    workers: int,
) -> pd.DataFrame:
    """Fill the rows the destination already holds, reading CRCs the checkpoint lacks by HEAD."""
    matched = _match_held(rows, _list_dest_objects(store_client, dest_bucket), checkpoint)
    no_crc = (
        rows["etag"].isna().to_numpy()
        & matched["etag"].notna().to_numpy()
        & matched["crc64nvme"].isna().to_numpy()
    )
    keys = (IMAGE_PREFIX + matched["key"][no_crc]).tolist()
    matched.loc[no_crc, "crc64nvme"] = pd.Series(
        _head_crcs(store_client, dest_bucket, keys, workers),
        index=matched.index[no_crc],
        dtype=object,
    )
    return matched


def _copy_pending(
    store_client: Any,
    source_client: Any,
    pending: pd.DataFrame,
    finished: pd.DataFrame,
    *,
    source_bucket: str,
    dest_bucket: str,
    workers: int,
    checkpoint: Path,
) -> pd.DataFrame:
    """Copy ``pending`` in batches; return the new copies.

    The checkpoint, ``finished`` plus the new copies, is written after every
    ``_CHECKPOINT_EVERY``-th batch and once after the last batch, never twice for one state.
    """
    copies = finished.iloc[0:0]
    starts = range(0, len(pending), _CHECKPOINT_BATCH)
    for number, start in enumerate(starts, 1):
        batch = pending.iloc[start : start + _CHECKPOINT_BATCH]
        items = [
            CopyItem(source_bucket=source_bucket, source_key=s3_key, dest_key=IMAGE_PREFIX + key)
            for s3_key, key in zip(batch["s3_key"], batch["key"], strict=True)
        ]
        copied = copy_images(
            store_client,
            items,
            dest_bucket=dest_bucket,
            workers=workers,
            storage_class=STORAGE_CLASS,
            source_client=source_client,
        )
        done = pd.DataFrame(
            {
                "dest_key": [item.dest_key for item in items],
                "source_fingerprint": batch["source_fingerprint"].tolist(),
                "etag": [copied[item.dest_key].etag for item in items],
                "size": [copied[item.dest_key].size for item in items],
                "crc64nvme": [copied[item.dest_key].crc64nvme for item in items],
            }
        )
        copies = pd.concat([copies, done], ignore_index=True)
        if number % _CHECKPOINT_EVERY == 0 or number == len(starts):
            _write_checkpoint(checkpoint, pd.concat([finished, copies], ignore_index=True))
        logger.info("copied %d of %d images", min(start + len(batch), len(pending)), len(pending))
    return copies


def _drop_unsupported(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop the rows with no media type, logging their ids; return the rest and the count."""
    unsupported = rows["media_type"].isna()
    if unsupported.any():
        logger.warning(
            "%d originals are neither JPEG nor PNG; dropped image ids: %s",
            int(unsupported.sum()),
            sorted(rows["image_id"][unsupported].tolist()),
        )
    return rows[~unsupported].reset_index(drop=True), int(unsupported.sum())


def _resolve_rows(
    source_client: Any,
    build: BuildTables,
    *,
    source_bucket: str,
    prefix: str,
    workers: int,
    csv_reader: CsvReader | None,
    parent_images: pa.Table | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return the kept image rows with source, media type and key, their points, and drops.

    Rows the parent carries keep its key, media type and object fields.
    """
    kept, points, dropped = _select_images(build)
    source_ids = sorted(int(sid) for sid in kept["source_id"].unique())
    etags, names, unlisted = _gather_sources(
        source_client, source_bucket, prefix, source_ids, workers=workers, csv_reader=csv_reader
    )
    rows, missing = _attach_sources(kept, etags, names)
    if missing:
        logger.warning(
            "%d source originals are missing; dropped image ids: %s", len(missing), missing
        )
    dropped["source_missing"] = len(missing)
    rows = _carry_forward(rows, parent_images)
    rows["media_type"] = _fill_media_types(source_client, source_bucket, rows, workers)
    rows, dropped["unsupported_format"] = _drop_unsupported(rows)
    dropped["source_name_missing"] = _count_unnamed(rows, unlisted)
    rows["key"] = [
        _image_key(int(sid), str(image_id), media_type) if pd.isna(key) else key
        for sid, image_id, media_type, key in zip(
            rows["source_id"], rows["image_id"], rows["media_type"], rows["key"], strict=True
        )
    ]
    return rows, points[points["image_id"].isin(rows["image_id"])], dropped


def build_raw(
    source_client: Any,
    store_client: Any,
    build: BuildTables,
    *,
    source_bucket: str,
    prefix: str,
    dest_bucket: str,
    workers: int,
    checkpoint: Path,
    parent_images: pa.Table | None = None,
    csv_reader: CsvReader | None = None,
) -> RawTables:
    """Copy the kept originals into ``dest_bucket`` and return the ``coralnet-raw`` tables.

    Rows of ``parent_images`` with the same image id and source fingerprint are reused. Keys the
    destination already holds with the row's source ETag, or with the ETag the checkpoint recorded
    for that source, are not copied again; the checkpoint supplies their CRC64NVME.

    Raises:
        ValueError: A label name is not a string, or an image_list names an image twice.
        segdata_spec.errors.ObjectExists: A destination key holds an object whose ETag is neither
            the row's single-part source fingerprint nor its recorded copy, or holds other bytes
            than a multipart source.
    """
    rows, points, dropped = _resolve_rows(
        source_client,
        build,
        source_bucket=source_bucket,
        prefix=prefix,
        workers=workers,
        csv_reader=csv_reader,
        parent_images=parent_images,
    )
    finished = _read_checkpoint(checkpoint)
    rows = _resolve_held(
        store_client,
        rows,
        finished,
        dest_bucket=dest_bucket,
        workers=workers,
    )
    pending = rows[rows["etag"].isna()]
    logger.info(
        "%d images to copy, %d already in the store", len(pending), len(rows) - len(pending)
    )
    copies = _copy_pending(
        store_client,
        source_client,
        pending,
        finished,
        source_bucket=source_bucket,
        dest_bucket=dest_bucket,
        workers=workers,
        checkpoint=checkpoint,
    )
    rows = _fill_copied(rows, copies)
    point_table, counts = _points_and_counts(points)
    tables = {
        "images.parquet": _images_table(rows),
        "points.parquet": point_table,
        "labels.parquet": labels_table(build.id2name),
        "stats/label_counts.parquet": counts,
    }
    return RawTables(tables=tables, dropped=dropped)


def _build_key(build_run: str, kind: str, suffix: str) -> str:
    return f"{DEFAULT_OUTPUT_S3_PREFIX}/{build_run}/coralnet_{kind}_{build_run}.{suffix}"


def _get_bytes(client: Any, bucket: str, key: str) -> bytes:
    return bytes(client.get_object(Bucket=bucket, Key=key)["Body"].read())


def read_build(
    client: Any, bucket: str, build_run: str, source_ids: Sequence[int] | None = None
) -> BuildTables:
    """Read a build run's tables, keeping only ``source_ids`` when given.

    id2name is read as JSON so string names stay strings.
    """
    frames = {
        kind: pq.read_table(
            io.BytesIO(_get_bytes(client, bucket, _build_key(build_run, kind, "parquet")))
        ).to_pandas()
        for kind in ("annotations", "images")
    }
    id2name = json.loads(_get_bytes(client, bucket, _build_key(build_run, "id2name", "json")))
    if source_ids is not None:
        frames = {k: f[f["source_id"].isin(source_ids)] for k, f in frames.items()}
    return BuildTables(annotations=frames["annotations"], images=frames["images"], id2name=id2name)


def _source_bucket() -> str:
    """The CoralNet source bucket from ``MERMAID_CORALNET_BUCKET``; there is no default.

    Raises:
        ValueError: The variable is unset or empty.
    """
    return get_bucket()


def _parent_ref(args: argparse.Namespace) -> ParentRef | None:
    """The parent pin, or None; ``commands.raw_flag_errors`` admits both flags or neither."""
    if args.parent_version is None:
        return None
    return ParentRef(
        dataset=DATASET, version=args.parent_version, manifest_sha256=args.parent_manifest_sha256
    )


def checkpoint_path(store: str | Path, build_run: str) -> Path:
    """The copy checkpoint: inside a local store's directory, or under ``outputs/`` for S3."""
    if str(store) == "s3":
        return DEFAULT_STORE.parent / "coralnet-raw-checkpoints" / f"s3-{build_run}.parquet"
    return Path(store) / "checkpoints" / f"{DATASET}-{build_run}.parquet"


def _command_line(args: argparse.Namespace) -> str:
    parts = ["segdata", "build", "coralnet", "raw", "--store", str(args.store)]
    parts += ["--build-run", args.build_run]
    if args.source_ids is not None:
        parts += ["--source-ids", ",".join(str(sid) for sid in args.source_ids)]
    if args.parent_version is not None:
        parts += pin_command_parts(args, "parent-")
    return shlex.join(parts)


def _workers(flag: int | None) -> int:
    return get_workers() if flag is None else validate_workers(flag)


def _pool_size(workers: int) -> int:
    """Connections for the shared S3 client: every worker, plus headroom for overlapping pools."""
    return workers + POOL_HEADROOM


def run_raw(args: argparse.Namespace) -> int:
    """Publish a new ``coralnet-raw`` version and print its version and manifest sha256."""
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    git = checked_git_state(repo_root)
    parent = _parent_ref(args)
    workers = _workers(args.workers)
    source_bucket, prefix = _source_bucket(), get_prefix()
    source_client = make_s3_client(max_pool_connections=_pool_size(workers))
    build = read_build(source_client, source_bucket, args.build_run, args.source_ids)
    client, bucket = open_store(args.store, source_client=source_client)
    parent_images = (
        None if parent is None else fetch_parent_tables(client, bucket, parent)["images.parquet"]
    )
    result = build_raw(
        source_client,
        client,
        build,
        source_bucket=source_bucket,
        prefix=prefix,
        dest_bucket=bucket,
        workers=workers,
        checkpoint=checkpoint_path(args.store, args.build_run),
        parent_images=parent_images,
    )
    version = pick_version(client, bucket, DATASET, git_sha=git.sha, today=utc_today())
    manifest, sha = publish_dataset(
        client,
        bucket=bucket,
        dataset=DATASET,
        version=version,
        kind="points",
        tables=result.tables,
        image_store=image_store(bucket, _IMAGE_FAMILY),
        etl_package="segdata-etl",
        etl_command=_command_line(args),
        source=SourceInfo(
            type="coralnet-scrape", uri=f"s3://{source_bucket}/{prefix}/", revision=args.build_run
        ),
        parents=() if parent is None else (parent,),
        dropped=result.dropped,
        repo_dir=repo_root,
        git=git,
    )
    logger.info("published %s %s; dropped %s", DATASET, manifest.version, result.dropped)
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0
