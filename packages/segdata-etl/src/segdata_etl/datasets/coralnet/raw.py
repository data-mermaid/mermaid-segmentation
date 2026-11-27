"""``segdata build coralnet raw``: stage a frozen CoralNet build as a ``coralnet-raw`` version.

The build's annotations, images and id2name tables become the version's points, images and labels.
Each kept original is probed (HEAD, plus a full read for a multipart or CRC-less object) for the
values its server-side copy will carry, and ``copies.jsonl`` plans that copy for ``segdata
promote``; nothing is written to S3. Width and height come from the build run. Probed rows are
checkpointed under the stage dir so a rerun probes only the rest, and a parent pin carries forward
the parent's rows whose listed source ETag is unchanged; a parent not staged is synced from the
dataset bucket into the cache dir. An image the scrape no longer lists but
the parent holds keeps the parent's row and gets no copy: its object is already published.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import shlex
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from coralnet_scraper.config import get_bucket, get_prefix, validate_workers
from coralnet_scraper.s3_io import object_key_image_list_csv
from segdata_etl.common.cli_flags import pin_command_parts
from segdata_etl.common.copy_plan import (
    ImageOutcome,
    ObjectFacts,
    copy_plan,
    needs_full_read,
    read_full,
)
from segdata_etl.common.jpeg_header import STATUSES_WITH_DIMENSIONS
from segdata_etl.common.s3_objects import (
    EXTENSIONS,
    head_with_checksum,
    make_s3_client,
    md5_hex,
    sniff_media_type,
)
from segdata_etl.common.stage import stage_version
from segdata_etl.common.versions import bare_fingerprints, count_labels, image_store
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ManifestNotFound
from segdata_spec.git import publishable_repo
from segdata_spec.layout import image_prefix
from segdata_spec.manifest import ParentRef, Pin, SourceInfo
from segdata_spec.s3 import crc64nvme_b64, get_bytes, norm_etag
from segdata_spec.schema import IMAGES_SCHEMA, LABELS_SCHEMA, POINTS_SCHEMA
from segdata_spec.stage import read_parent_tables
from segdata_spec.sync import sync_tables

from .annotations import _IMAGE_LIST_NAME, derive_image_id
from .config import DEFAULT_OUTPUT_S3_PREFIX, get_workers
from .io import CsvReader, make_csv_reader_s3

logger = logging.getLogger(__name__)

DATASET = "coralnet-raw"
# The images live under the "coralnet" family, not the "coralnet-raw" dataset name.
_IMAGE_FAMILY = "coralnet"
IMAGE_PREFIX = image_prefix(_IMAGE_FAMILY)
ANNOTATOR = "default"
# Hex digits of the commit sha in a checkpoint name, as in a version name.
_SHA_LENGTH = 7
# CoralNet label 2474 is named the literal string "None", which pandas read as NaN into id2name.
_NONE_NAMED_LABEL = "2474"
# Probes per batch, the unit of progress between checkpoint writes.
_CHECKPOINT_BATCH = 1000
# Batches per checkpoint write: a crash loses at most this many batches of finished probes.
_CHECKPOINT_EVERY = 20
# Bytes read to recognise a PNG signature, the longer of the two supported magic numbers.
_SNIFF_BYTES = 8
_SOURCE_MISSING = "source_missing"
_CARRIED = "carried_from_parent"
_NOT_FOUND = "not_found"
_UNSUPPORTED = "unsupported_format"
_OBJECT_COLUMNS = [
    "image_id",
    "key",
    "media_type",
    "etag",
    "size",
    "crc64nvme",
    "source_fingerprint",
]


@dataclass(frozen=True)
class BuildTables:
    """A CoralNet build run's annotations and images frames and its id2name map."""

    annotations: pd.DataFrame
    images: pd.DataFrame
    id2name: dict[str, Any]


@dataclass(frozen=True)
class RawTables:
    """The ``coralnet-raw`` tables keyed by file name, the dropped counts by reason, and the carried ids.

    ``carried`` holds the image ids whose rows come from the parent with no scrape original.
    """

    tables: dict[str, pa.Table]
    dropped: dict[str, int]
    carried: frozenset[str] = frozenset()


def _label_name(label_id: str, value: Any) -> str:
    if label_id == _NONE_NAMED_LABEL and isinstance(value, float) and math.isnan(value):
        return "None"
    return value


def labels_table(id2name: dict[str, Any]) -> pa.Table:
    """One identity-mapped label row per id2name entry, sorted by id."""
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


def _select_images(
    build: BuildTables, parent_ids: frozenset[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[str]]:
    """Return the annotated images with dimensions, their annotations, the dropped counts and carries.

    An annotated image the build reads as ``not_found`` that ``parent_ids`` holds is carried: its
    points stay and it is not counted as ``no_dimensions``.
    """
    images, annotations = build.images, build.annotations
    status = images["header_status"].astype(str)
    has_dims = status.isin(STATUSES_WITH_DIMENSIONS)
    keys = ["source_id", "image_id"]
    listed = annotations.merge(images[keys], on=keys, how="left", indicator=True)
    with_image = (listed["_merge"] == "both").to_numpy()
    annotated = images["image_id"].isin(annotations["image_id"][with_image])
    lost = (status == _NOT_FOUND) & images["image_id"].astype(str).isin(parent_ids) & annotated
    with_dims = images[has_dims]
    kept = with_dims[with_dims["image_id"].isin(annotations["image_id"][with_image])]
    carried = sorted(images["image_id"][lost].astype(str))
    wanted = annotations["image_id"].isin(kept["image_id"]) | annotations["image_id"].isin(carried)
    points = annotations[wanted & with_image]
    dropped = {
        "no_dimensions": int((~has_dims & ~lost).sum()),
        "no_annotations": len(with_dims) - len(kept),
        "annotations_without_image": int((~with_image).sum()),
    }
    return kept, points, dropped, carried


def _source_names(image_list: pd.DataFrame) -> pd.DataFrame:
    """Map image_id to the image_list Name without its status suffix, as the build derives it.

    Raises:
        ValueError: One image_id carries two different names.
    """
    names = pd.DataFrame(
        {
            "image_id": derive_image_id(image_list["Image Page"]),
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
    """Add each image's unquoted source ETag and name; return the rows and the ids with no source."""
    rows = kept.assign(source_fingerprint=kept["s3_key"].map(fingerprints))
    missing = rows["source_fingerprint"].isna()
    rows = rows[~missing].merge(names, on="image_id", how="left")
    rows["source_fingerprint"] = rows["source_fingerprint"].map(norm_etag)
    return rows, sorted(kept["image_id"][missing].tolist())


def _parent_ids(parent_images: pa.Table | None) -> frozenset[str]:
    return (
        frozenset() if parent_images is None else frozenset(parent_images["image_id"].to_pylist())
    )


def _split_carried(missing: list[str], held: frozenset[str]) -> tuple[list[str], list[str]]:
    """Split the ids absent from the listing into those the parent holds and those it lacks."""
    return [i for i in missing if i in held], [i for i in missing if i not in held]


def _check_carried_sizes(
    kept: pd.DataFrame, parent_images: pa.Table | None, carried: list[str]
) -> None:
    """Refuse a carried id whose build width and height differ from the parent row's.

    Raises:
        ValueError: At least one carried id has a different build size; the first 10 are named.
    """
    if parent_images is None or not carried:
        return
    parent = {
        row["image_id"]: (row["width"], row["height"])
        for row in parent_images.select(["image_id", "width", "height"]).to_pylist()
    }
    built = kept[kept["image_id"].astype(str).isin(carried)]
    differing = sorted(
        str(row["image_id"])
        for row in built.to_dict("records")
        if parent[str(row["image_id"])] != (row["width"], row["height"])
    )
    if differing:
        raise ValueError(
            f"{len(differing)} carried images have a build size other than the parent's: "
            f"{differing[:10]}"
        )


def _check_carried_points(
    points: pd.DataFrame, parent_images: pa.Table | None, carried: list[str]
) -> None:
    """Refuse a carried id with a build point outside the parent row's image.

    A point is outside when its 0-based ``row`` or ``col`` is negative or at or past the height
    or width.

    Raises:
        ValueError: At least one carried id has such a point; the first 10 are named.
    """
    if parent_images is None or not carried:
        return
    size = {
        row["image_id"]: (row["width"], row["height"])
        for row in parent_images.select(["image_id", "width", "height"]).to_pylist()
    }
    ids = points["image_id"].astype(str)
    mask = ids.isin(carried)
    kept = points[mask]
    outside = sorted(
        {
            image_id
            for image_id, row, col in zip(ids[mask], kept["row"], kept["col"], strict=True)
            if not (0 <= row < size[image_id][1] and 0 <= col < size[image_id][0])
        }
    )
    if outside:
        raise ValueError(
            f"{len(outside)} carried images have points outside the parent's size: {outside[:10]}"
        )


def _image_key(source_id: int, image_id: str, media_type: str) -> str:
    return f"s{source_id}/{image_id}.{EXTENSIONS[media_type]}"


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
        return pd.DataFrame({name: pd.Series(dtype=object) for name in _OBJECT_COLUMNS})
    return pd.read_parquet(path, columns=_OBJECT_COLUMNS)


def _write_checkpoint(path: Path, probed: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    latest = probed.drop_duplicates("image_id", keep="last")
    latest[_OBJECT_COLUMNS].to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _split_known(rows: pd.DataFrame, known: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the known rows matching a row's image id and source fingerprint, and the rest.

    Of two known rows for one image id, the later wins.
    """
    pairs = list(zip(rows["image_id"].astype(str), rows["source_fingerprint"], strict=True))
    held = set(zip(known["image_id"].astype(str), known["source_fingerprint"], strict=True))
    wanted = set(pairs)
    known_pairs = zip(known["image_id"].astype(str), known["source_fingerprint"], strict=True)
    reused = known[[pair in wanted for pair in known_pairs]]
    pending = rows[[pair not in held for pair in pairs]]
    return reused.drop_duplicates("image_id", keep="last"), pending


def _object_row(
    source_id: int, image_id: str, facts: ObjectFacts, media_type: str
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "key": _image_key(source_id, image_id, media_type),
        "media_type": media_type,
        "etag": facts.etag,
        "size": facts.size,
        "crc64nvme": facts.crc64nvme,
        "source_fingerprint": facts.source_etag,
    }


def _media_type(
    client: Any, bucket: str, key: str, header_status: str, head: Mapping[str, Any]
) -> str | None:
    """JPEG for a header the JPEG parser read; otherwise sniffed from the first bytes.

    None when the sniffed bytes are neither JPEG nor PNG.
    """
    if header_status == "ok":
        return "image/jpeg"
    if head["ContentLength"] == 0:
        return None  # S3 answers a ranged GET on an empty object with 416
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{_SNIFF_BYTES - 1}")
    return sniff_media_type(response["Body"].read())


def _object_facts(client: Any, bucket: str, key: str, head: Mapping[str, Any]) -> ObjectFacts:
    """The values the copy will carry: the HEAD's, or a whole read's for a multipart or CRC-less object.

    The whole read needs no parseable image header; width and height come from the build run.
    """
    source_etag = norm_etag(head["ETag"])
    if needs_full_read(head):
        data = read_full(client, bucket, key, head["ETag"])
        return ObjectFacts(md5_hex(data), len(data), crc64nvme_b64(data), source_etag)
    return ObjectFacts(source_etag, head["ContentLength"], head["ChecksumCRC64NVME"], source_etag)


def _probe_one(client: Any, bucket: str, image: Mapping[str, Any]) -> ImageOutcome:
    """Probe one original; a drop is ``source_missing`` or ``unsupported_format``."""
    image_id, key = str(image["image_id"]), str(image["s3_key"])
    head = head_with_checksum(client, bucket, key)
    if head is None:
        return ImageOutcome(image_id, dropped=_SOURCE_MISSING)
    media_type = _media_type(client, bucket, key, str(image["header_status"]), head)
    if media_type is None:
        return ImageOutcome(image_id, dropped=_UNSUPPORTED)
    facts = _object_facts(client, bucket, key, head)
    return ImageOutcome(
        image_id, row=_object_row(int(image["source_id"]), image_id, facts, media_type)
    )


def _kept_frame(outcomes: Sequence[ImageOutcome]) -> pd.DataFrame:
    rows = [{c: o.row[c] for c in _OBJECT_COLUMNS} for o in outcomes if o.row is not None]
    return pd.DataFrame(rows, columns=_OBJECT_COLUMNS)


def _probe_pending(
    client: Any,
    bucket: str,
    pending: pd.DataFrame,
    done: pd.DataFrame,
    *,
    workers: int,
    checkpoint: Path,
) -> list[ImageOutcome]:
    """Probe ``pending`` in batches; return every outcome.

    The checkpoint, ``done`` plus the rows kept so far, is written after every
    ``_CHECKPOINT_EVERY``-th batch and once after the last batch, never twice for one state.
    """
    images = pending[["image_id", "source_id", "s3_key", "header_status"]].to_dict("records")
    outcomes: list[ImageOutcome] = []
    starts = range(0, len(images), _CHECKPOINT_BATCH)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, start in enumerate(starts, 1):
            batch = images[start : start + _CHECKPOINT_BATCH]
            outcomes += pool.map(lambda image: _probe_one(client, bucket, image), batch)
            if number % _CHECKPOINT_EVERY == 0 or number == len(starts):
                kept = _kept_frame(outcomes)
                _write_checkpoint(checkpoint, pd.concat([done, kept], ignore_index=True))
            logger.info("probed %d of %d images", start + len(batch), len(images))
    return outcomes


def _record_drops(outcomes: Sequence[ImageOutcome], dropped: dict[str, int]) -> None:
    """Add the probe's drops to ``dropped`` by reason, logging the image ids of each reason."""
    reasons = Counter(o.dropped for o in outcomes if o.dropped is not None)
    for reason in (_SOURCE_MISSING, _UNSUPPORTED):
        ids = sorted(o.image_id for o in outcomes if o.dropped == reason)
        if ids:
            logger.warning("%d images dropped as %s: %s", len(ids), reason, ids)
        dropped[reason] = dropped.get(reason, 0) + reasons[reason]


def _known_rows(parent_images: pa.Table | None, done: pd.DataFrame) -> pd.DataFrame:
    """The parent's object rows followed by the checkpoint's, which win for one image id."""
    if parent_images is None:
        return done
    parent = parent_images.select(_OBJECT_COLUMNS).to_pandas()
    return pd.concat([parent, done], ignore_index=True)


def _select_rows(
    client: Any,
    build: BuildTables,
    *,
    source_bucket: str,
    prefix: str,
    workers: int,
    csv_reader: CsvReader | None,
    parent_images: pa.Table | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[int], list[str]]:
    """Return the kept rows with source fingerprint and name, their points, drops, unlisted, carried.

    The carried ids are those that ``parent_images`` holds and the build or the listing lacks:
    ``not_found`` header rows and images absent from the listing. They are in no returned row and
    counted neither as ``no_dimensions`` nor as ``source_missing``.

    Raises:
        ValueError: A carried image's build size differs from the parent's, or a not_found carried
            image has a point outside the parent's size.
    """
    held = _parent_ids(parent_images)
    kept, points, dropped, lost = _select_images(build, held)
    source_ids = sorted(int(sid) for sid in kept["source_id"].unique())
    etags, names, unlisted = _gather_sources(
        client, source_bucket, prefix, source_ids, workers=workers, csv_reader=csv_reader
    )
    rows, absent = _attach_sources(kept, etags, names)
    listed_out, missing = _split_carried(absent, held)
    _check_carried_sizes(kept, parent_images, listed_out)
    _check_carried_points(points, parent_images, lost)
    carried = sorted([*lost, *listed_out])
    if missing:
        logger.warning(
            "%d source originals are missing; dropped image ids: %s", len(missing), missing
        )
    dropped[_SOURCE_MISSING] = len(missing)
    if parent_images is not None:
        dropped[_CARRIED] = len(carried)
    return rows, points, dropped, unlisted, carried


def _with_objects(rows: pd.DataFrame, objects: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its object fields by image id; rows without them are dropped."""
    keyed = rows.drop(columns=["source_fingerprint"]).assign(_id=rows["image_id"].astype(str))
    found = objects.drop(columns=["image_id"]).assign(_id=objects["image_id"].astype(str))
    return keyed.merge(found, on="_id", how="inner").drop(columns=["_id"])


def _with_carried(images: pa.Table, parent_images: pa.Table | None, carried: list[str]) -> pa.Table:
    """Add the parent's rows of the ``carried`` ids to ``images``, in image_id order."""
    if parent_images is None or not carried:
        return images
    parent_rows = parent_images.filter(pc.is_in(parent_images["image_id"], pa.array(carried)))
    merged = pa.concat_tables([images, parent_rows.select(images.schema.names).cast(images.schema)])
    return merged.sort_by("image_id")


def build_raw(
    client: Any,
    build: BuildTables,
    *,
    source_bucket: str,
    prefix: str,
    workers: int,
    checkpoint: Path,
    parent_images: pa.Table | None = None,
    csv_reader: CsvReader | None = None,
) -> RawTables:
    """Probe the kept originals and return the ``coralnet-raw`` tables.

    Checkpoint rows and ``parent_images`` rows whose image id and source fingerprint match the
    source listing are reused without any request for that image. With a parent, an image the
    listing lacks but the parent holds keeps the parent's row and is counted in
    ``dropped["carried_from_parent"]``; an image the parent lacks too is ``source_missing``. An
    image the build reads as ``not_found`` whose id the parent holds is carried the same way: the
    parent's row, the build's points, counted in ``carried_from_parent`` and not in
    ``no_dimensions``.

    Raises:
        ValueError: A label name is not a string, an image_list names an image twice, or a carried
            image's build size differs from the parent's or a not_found carried image has a point
            outside the parent's size.
        SourceChanged: An original was replaced between its HEAD and its read.
    """
    rows, points, dropped, unlisted, carried = _select_rows(
        client,
        build,
        source_bucket=source_bucket,
        prefix=prefix,
        workers=workers,
        csv_reader=csv_reader,
        parent_images=parent_images,
    )
    done = _read_checkpoint(checkpoint)
    reused, pending = _split_known(rows, _known_rows(parent_images, done))
    logger.info("%d images to probe, %d reused", len(pending), len(rows) - len(pending))
    outcomes = _probe_pending(
        client,
        source_bucket,
        pending,
        done,
        workers=workers,
        checkpoint=checkpoint,
    )
    _record_drops(outcomes, dropped)
    rows = _with_objects(rows, pd.concat([reused, _kept_frame(outcomes)], ignore_index=True))
    dropped["source_name_missing"] = _count_unnamed(rows, unlisted)
    point_ids = [*rows["image_id"].astype(str), *carried]
    point_table, counts = _points_and_counts(points[points["image_id"].astype(str).isin(point_ids)])
    tables = {
        "images.parquet": _with_carried(_images_table(rows), parent_images, carried),
        "points.parquet": point_table,
        "labels.parquet": labels_table(build.id2name),
        "stats/label_counts.parquet": counts,
    }
    return RawTables(tables=tables, dropped=dropped, carried=frozenset(carried))


def _build_key(build_run: str, kind: str, suffix: str) -> str:
    return f"{DEFAULT_OUTPUT_S3_PREFIX}/{build_run}/coralnet_{kind}_{build_run}.{suffix}"


def read_build(
    client: Any, bucket: str, build_run: str, source_ids: Sequence[int] | None = None
) -> BuildTables:
    """Read a build run's tables, keeping only ``source_ids`` when given.

    id2name is read as JSON so string names stay strings.
    """
    frames = {
        kind: pq.read_table(
            io.BytesIO(get_bytes(client, bucket, _build_key(build_run, kind, "parquet")))
        ).to_pandas()
        for kind in ("annotations", "images")
    }
    id2name = json.loads(get_bytes(client, bucket, _build_key(build_run, "id2name", "json")))
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


def _read_parent_tables(
    args: argparse.Namespace, client: Any, parent: ParentRef
) -> dict[str, bytes]:
    """Read the parent's tables from the stage dir, else from the cache dir after a sync.

    A parent staged under ``args.stage_dir`` needs no S3 access. Otherwise its manifest and tables
    are synced from the dataset bucket into ``args.cache_dir``, which the sync reads first.

    Raises:
        PinMismatch: The staged or cached parent manifest differs from its pin.
    """
    try:
        return read_parent_tables((args.stage_dir,), DATASET_BUCKET, parent)
    except ManifestNotFound:
        pin = Pin(
            name=parent.dataset, version=parent.version, manifest_sha256=parent.manifest_sha256
        )
        sync_tables(client, {pin.name: pin}, args.cache_dir, bucket=DATASET_BUCKET)
        return read_parent_tables((args.stage_dir, args.cache_dir), DATASET_BUCKET, parent)


def checkpoint_path(stage_dir: Path, build_run: str, git_sha: str) -> Path:
    """The probe checkpoint of one build run at one commit, beside the staged bucket tree."""
    return stage_dir / "checkpoints" / f"{DATASET}-{build_run}-{git_sha[:_SHA_LENGTH]}.parquet"


def _source_key(row: Mapping[str, Any]) -> str:
    source_key: str = json.loads(row["attrs"])["source_key"]
    return source_key


def _copies(images: pa.Table, source_bucket: str, carried: frozenset[str]) -> list[dict[str, Any]]:
    """Plan each image's server-side copy from the scrape bucket; a carried image needs none."""
    return copy_plan(
        [row for row in images.to_pylist() if row["image_id"] not in carried],
        src_bucket=source_bucket,
        src_key=_source_key,
        dst_prefix=IMAGE_PREFIX,
    )


def _command_line(args: argparse.Namespace) -> str:
    parts = ["segdata", "build", "coralnet", "raw", "--stage-dir", str(args.stage_dir)]
    parts += ["--build-run", args.build_run]
    if args.source_ids is not None:
        parts += ["--source-ids", ",".join(str(sid) for sid in args.source_ids)]
    if args.parent_version is not None:
        parts += pin_command_parts(args, "parent-")
    return shlex.join(parts)


def _workers(flag: int | None) -> int:
    return get_workers() if flag is None else validate_workers(flag)


def run_raw(args: argparse.Namespace) -> int:
    """Stage a new ``coralnet-raw`` version and its copy plan; print its version and sha256.

    Raises:
        PublishRefused: The git tree is dirty or unpushed, or ``uv.lock`` is missing.
        ValueError: ``MERMAID_CORALNET_BUCKET`` is unset.
        PinMismatch: The staged or synced parent's manifest differs from its pin.
        SourceChanged: An original was replaced between its HEAD and its read.
    """
    repo = publishable_repo(Path(__file__).resolve().parent)
    parent = _parent_ref(args)
    workers = _workers(args.workers)
    source_bucket, prefix = _source_bucket(), get_prefix()
    client = make_s3_client(workers=workers)
    build = read_build(client, source_bucket, args.build_run, args.source_ids)
    parent_images = (
        None
        if parent is None
        else bare_fingerprints(
            pq.read_table(io.BytesIO(_read_parent_tables(args, client, parent)["images.parquet"]))
        )
    )
    result = build_raw(
        client,
        build,
        source_bucket=source_bucket,
        prefix=prefix,
        workers=workers,
        checkpoint=checkpoint_path(args.stage_dir, args.build_run, repo.git.sha),
        parent_images=parent_images,
    )
    stage_version(
        args.stage_dir,
        repo=repo,
        client=client,
        dataset=DATASET,
        kind="points",
        tables=result.tables,
        image_store=image_store(DATASET_BUCKET, _IMAGE_FAMILY),
        etl_command=_command_line(args),
        source=SourceInfo(
            type="coralnet-scrape", uri=f"s3://{source_bucket}/{prefix}/", revision=args.build_run
        ),
        parents=() if parent is None else (parent,),
        extra_parent_dirs=(args.cache_dir,),
        dropped=result.dropped,
        copies=_copies(result.tables["images.parquet"], source_bucket, result.carried),
    )
    return 0
