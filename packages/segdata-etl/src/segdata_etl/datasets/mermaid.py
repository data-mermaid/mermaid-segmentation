"""``segdata build mermaid``: stage a uniform-format points dataset from MERMAID's confirmed annotations.

The build reads the nightly annotations parquet and the source image headers, never writes to S3, and
leaves a version under ``--stage-dir`` plus ``copies.jsonl``, the plan a later promote step runs to
copy the images into the dataset bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_etl.common.s3_objects import (
    SourceChanged,
    crc64nvme_b64,
    image_size,
    md5_hex,
    sniff_media_type,
)
from segdata_etl.common.stage import LocalDirS3, next_version
from segdata_spec import DATASET_BUCKET
from segdata_spec._s3 import cache_path, dataset_prefix, is_not_found, is_precondition_failed
from segdata_spec.manifest import ImageStore, LabelMapping, ParentRef, SourceInfo
from segdata_spec.publish import probe_git, publish_dataset
from segdata_spec.schema import IMAGES_SCHEMA, LABEL_COUNTS_SCHEMA, LABELS_SCHEMA, POINTS_SCHEMA

logger = logging.getLogger(__name__)

DATASET = "mermaid"
SOURCE_BUCKET = "coral-reef-training"
SOURCE_IMAGE_PREFIX = "mermaid/"
DEFAULT_ANNOTATIONS_URI = f"s3://{SOURCE_BUCKET}/mermaid/mermaid_confirmed_annotations.parquet"
IMAGE_PREFIX = "images/mermaid/"
GROUP = "mermaid"
ANNOTATOR = "default"
COPIES_NAME = "copies.jsonl"

# The repo root, which holds ``uv.lock`` and the git checkout that built the version.
REPO_ROOT = Path(__file__).resolve().parents[5]

_FIRST_READ = 64 * 1024
_MAX_READ = 1024 * 1024
_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png"}


class AnnotationsChanged(Exception):
    """The annotations parquet was rewritten between the HEAD and the GET of one build."""


@dataclass(frozen=True)
class AnnotationsFile:
    """The annotations table and the object version it was read from."""

    table: pd.DataFrame
    etag: str
    last_modified: str


@dataclass(frozen=True)
class ImageOutcome:
    """What probing one source image decided: a row to keep, a row to carry, or a drop reason."""

    image_id: str
    row: dict[str, Any] | None = None
    carried: bool = False
    dropped: str | None = None
    found_as_jpg: bool = False


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the MERMAID build flags; ``workdir`` is unused because the build stages under ``--stage-dir``."""
    del workdir
    parser.add_argument("--annotations-uri", default=DEFAULT_ANNOTATIONS_URI)
    parser.add_argument(
        "--annotation-date-cutoff",
        required=True,
        help="YYYY-MM-DD; keep annotations updated at or before this midnight UTC",
    )
    parser.add_argument("--parent-version", default=None, help="Staged version to reuse rows from")
    parser.add_argument(
        "--parent-manifest-sha256", default=None, help="Manifest sha256 of --parent-version"
    )


def _split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"not an s3:// object URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_annotations(client: Any, uri: str) -> AnnotationsFile:
    """HEAD the parquet, then GET exactly that version.

    Raises:
        AnnotationsChanged: The object was replaced between the HEAD and the GET.
    """
    bucket, key = _split_uri(uri)
    head = client.head_object(Bucket=bucket, Key=key)
    etag = head["ETag"]
    try:
        body = client.get_object(Bucket=bucket, Key=key, IfMatch=etag)["Body"].read()
    except ClientError as exc:
        if is_precondition_failed(exc):
            raise AnnotationsChanged(
                f"{uri} changed while it was being read; rerun the build"
            ) from exc
        raise
    frame = pq.read_table(io.BytesIO(body)).to_pandas()
    return AnnotationsFile(frame, etag.strip('"'), head["LastModified"].isoformat())


def apply_cutoff(annotations: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, int]:
    """Keep rows with ``updated_on`` at or before midnight UTC of ``cutoff``; return them and the drop count."""
    keep = annotations["updated_on"] <= pd.Timestamp(cutoff, tz="UTC")
    return annotations[keep], int((~keep).sum())


def _read_range(client: Any, key: str, etag: str, last_byte: int) -> bytes:
    response = client.get_object(
        Bucket=SOURCE_BUCKET, Key=key, Range=f"bytes=0-{last_byte}", IfMatch=etag
    )
    data: bytes = response["Body"].read()
    return data


def read_image_header(client: Any, key: str, etag: str, size: int) -> tuple[str, int, int] | None:
    """Return ``(media_type, width, height)`` from the object's leading bytes, or None if unreadable.

    The first 64 KB usually holds the header; a JPEG whose SOF lies further in is read up to 1 MB.

    Raises:
        SourceChanged: The object no longer matches ``etag``.
    """
    try:
        head = _read_range(client, key, etag, _FIRST_READ - 1)
        media_type = sniff_media_type(head)
        if media_type is None:
            return None
        dims = image_size(head)
        if dims is None and media_type == "image/jpeg" and size > _FIRST_READ:
            dims = image_size(_read_range(client, key, etag, _MAX_READ - 1))
    except ClientError as exc:
        if is_precondition_failed(exc):
            raise SourceChanged(f"s3://{SOURCE_BUCKET}/{key} changed while it was read") from exc
        raise
    return None if dims is None else (media_type, dims[0], dims[1])


@dataclass(frozen=True)
class _ObjectFacts:
    """The destination object's ETag, size and CRC64NVME, and the source ETag it was planned from."""

    etag: str
    size: int
    crc64nvme: str
    source_etag: str


def _needs_full_read(head: Mapping[str, Any]) -> bool:
    """A multipart ETag or a missing CRC64NVME means the copy would not carry the source's values."""
    return "-" in head["ETag"] or head.get("ChecksumCRC64NVME") is None


def _read_full(client: Any, key: str, etag: str) -> bytes:
    try:
        data: bytes = client.get_object(Bucket=SOURCE_BUCKET, Key=key, IfMatch=etag)["Body"].read()
    except ClientError as exc:
        if is_precondition_failed(exc):
            raise SourceChanged(f"s3://{SOURCE_BUCKET}/{key} changed while it was read") from exc
        raise
    return data


def _read_whole_object(
    client: Any, key: str, head: Mapping[str, Any]
) -> tuple[tuple[str, int, int], _ObjectFacts] | None:
    """Download the object and derive the header and the values a single-part copy will carry."""
    data = _read_full(client, key, head["ETag"])
    media_type = sniff_media_type(data)
    dims = image_size(data)
    if media_type is None or dims is None:
        return None
    facts = _ObjectFacts(md5_hex(data), len(data), crc64nvme_b64(data), head["ETag"].strip('"'))
    return (media_type, dims[0], dims[1]), facts


def _image_row(
    image_id: str,
    source_key: str,
    facts: _ObjectFacts,
    header: tuple[str, int, int],
    region: tuple[Any, Any],
) -> dict[str, Any]:
    media_type, width, height = header
    return {
        "image_id": image_id,
        "key": image_id + _EXTENSIONS[media_type],
        "etag": facts.etag,
        "size": facts.size,
        "crc64nvme": facts.crc64nvme,
        "media_type": media_type,
        "width": width,
        "height": height,
        "group": GROUP,
        "mask_key": None,
        "mask_etag": None,
        "mask_size": None,
        "mask_crc64nvme": None,
        "source_fingerprint": facts.source_etag,
        "source_name": None,
        "attrs": json.dumps(
            {"source_key": source_key, "region_id": region[0], "region_name": region[1]},
            sort_keys=True,
        ),
    }


def _head_source(client: Any, image_id: str) -> tuple[str, Mapping[str, Any]] | None:
    """HEAD ``mermaid/<id>.png``, then ``.jpg``; return the key found and its HEAD, or None."""
    for extension in (".png", ".jpg"):
        key = f"{SOURCE_IMAGE_PREFIX}{image_id}{extension}"
        try:
            head: Mapping[str, Any] = client.head_object(
                Bucket=SOURCE_BUCKET, Key=key, ChecksumMode="ENABLED"
            )
        except ClientError as exc:
            if is_not_found(exc):
                continue
            raise
        return key, head
    return None


def probe_image(
    client: Any,
    image_id: str,
    region: tuple[Any, Any],
    parent_rows: Mapping[str, dict[str, Any]],
) -> ImageOutcome:
    """Decide one image from its source HEAD and, unless a parent row matches, its bytes.

    A multipart ETag or a missing CRC64NVME means the whole object is read, so the row records the
    ETag and CRC64NVME a single-part copy will have; the source ETag stays the fingerprint.
    """
    found = _head_source(client, image_id)
    if found is None:
        return ImageOutcome(image_id, dropped="image_deleted_at_source")
    source_key, head = found
    as_jpg = source_key.endswith(".jpg")
    source_etag = head["ETag"].strip('"')
    parent = parent_rows.get(image_id)
    if parent is not None and parent["source_fingerprint"] == source_etag:
        return ImageOutcome(image_id, row=parent, carried=True, found_as_jpg=as_jpg)
    if _needs_full_read(head):
        whole = _read_whole_object(client, source_key, head)
        header, facts = whole if whole is not None else (None, None)
    else:
        header = read_image_header(client, source_key, head["ETag"], head["ContentLength"])
        facts = _ObjectFacts(
            source_etag, head["ContentLength"], head["ChecksumCRC64NVME"], source_etag
        )
    if header is None or facts is None:
        logger.warning("dropping image %s: unreadable header in %s", image_id, source_key)
        return ImageOutcome(image_id, dropped="image_unreadable_header")
    row = _image_row(image_id, source_key, facts, header, region)
    return ImageOutcome(image_id, row=row, found_as_jpg=as_jpg)


def first_regions(annotations: pd.DataFrame) -> dict[str, tuple[Any, Any]]:
    """Map each image to the region of its first point by ``point_id``."""
    first = annotations.sort_values(["image_id", "point_id"]).drop_duplicates("image_id")
    return {
        str(row.image_id): (row.region_id, row.region_name) for row in first.itertuples(index=False)
    }


def _gf_id(frame: pd.DataFrame) -> pd.Series:
    return frame["growth_form_id"].fillna("")


def with_label_ids(annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Number the BA+GF pairs 1..N by ``(benthic_attribute_id, growth_form_id or '')``.

    Returns the annotations with a ``label_id`` column and the labels frame.
    """
    frame = annotations.assign(_gf=_gf_id(annotations))
    pairs = (
        frame[["benthic_attribute_id", "_gf", "benthic_attribute_name", "growth_form_name"]]
        .drop_duplicates(["benthic_attribute_id", "_gf"])
        .sort_values(["benthic_attribute_id", "_gf"])
        .reset_index(drop=True)
    )
    pairs["label_id"] = pairs.index + 1
    ids = pairs.set_index(["benthic_attribute_id", "_gf"])["label_id"]
    keyed = frame.set_index(["benthic_attribute_id", "_gf"]).index
    labelled = annotations.assign(label_id=ids.reindex(keyed).to_numpy())
    return labelled, pairs


def labels_table(pairs: pd.DataFrame) -> pa.Table:
    """Build ``labels.parquet`` from the numbered pairs; the name keeps a trailing ``::`` with no growth form."""
    gf_names = (
        pairs["growth_form_name"].astype(object).where(pairs["growth_form_name"].notna(), None)
    )
    return pa.table(
        {
            "label_id": pairs["label_id"].tolist(),
            "name": [
                f"{ba}::{gf or ''}"
                for ba, gf in zip(pairs["benthic_attribute_name"], gf_names, strict=True)
            ],
            "mermaid_benthic_attribute_id": pairs["benthic_attribute_id"].tolist(),
            "mermaid_benthic_attribute": pairs["benthic_attribute_name"].tolist(),
            "mermaid_growth_form": gf_names.tolist(),
            "mapping_source": ["identity"] * len(pairs),
        },
        schema=LABELS_SCHEMA,
    )


def labels_sha256(pairs: pd.DataFrame) -> str:
    """Hash the sorted pair list as compact JSON of ``[ba_id, gf_id or '', ba_name, gf_name or '']``."""
    entries = [
        [ba_id, gf_id, ba_name, gf_name or ""]
        for ba_id, gf_id, ba_name, gf_name in zip(
            pairs["benthic_attribute_id"],
            pairs["_gf"],
            pairs["benthic_attribute_name"],
            pairs["growth_form_name"],
            strict=True,
        )
    ]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def points_table(labelled: pd.DataFrame) -> pa.Table:
    """Build ``points.parquet`` rows from annotations carrying ``label_id``."""
    return pa.table(
        {
            "image_id": labelled["image_id"].astype(str).tolist(),
            "row": labelled["row"].tolist(),
            "col": labelled["col"].tolist(),
            "label_id": labelled["label_id"].tolist(),
            "annotator": [ANNOTATOR] * len(labelled),
        },
        schema=POINTS_SCHEMA,
    )


def label_counts_table(labelled: pd.DataFrame) -> pa.Table:
    """Count points per ``(image_id, label_id)``."""
    counts = labelled.groupby(["image_id", "label_id"]).size().reset_index(name="n")
    return pa.table(
        {
            "image_id": counts["image_id"].astype(str).tolist(),
            "label_id": counts["label_id"].tolist(),
            "n": counts["n"].tolist(),
        },
        schema=LABEL_COUNTS_SCHEMA,
    )


def copy_plan(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe the server-side copy of each image row, in ``image_id`` order.

    ``src_etag`` is the source object's ETag; ``etag``, ``size`` and ``crc64nvme`` are the
    destination object's expected values.
    """
    plan = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        source_key = json.loads(row["attrs"])["source_key"]
        plan.append(
            {
                "src_bucket": SOURCE_BUCKET,
                "src_key": source_key,
                "src_etag": row["source_fingerprint"],
                "etag": row["etag"],
                "dst_bucket": DATASET_BUCKET,
                "dst_key": IMAGE_PREFIX + row["key"],
                "size": row["size"],
                "crc64nvme": row["crc64nvme"],
            }
        )
    return plan


def load_parent_rows(stage_dir: Path, version: str | None) -> dict[str, dict[str, Any]]:
    """Read the staged parent's image rows by ``image_id``; empty without a parent."""
    if version is None:
        return {}
    path = cache_path(
        stage_dir, DATASET_BUCKET, dataset_prefix(DATASET, version) + "images.parquet"
    )
    return {row["image_id"]: row for row in pq.read_table(path).to_pylist()}


def _parent_refs(args: argparse.Namespace) -> list[ParentRef]:
    if (args.parent_version is None) != (args.parent_manifest_sha256 is None):
        raise ValueError("--parent-version and --parent-manifest-sha256 go together")
    if args.parent_version is None:
        return []
    return [
        ParentRef(
            dataset=DATASET,
            version=args.parent_version,
            manifest_sha256=args.parent_manifest_sha256,
        )
    ]


def _command_line(args: argparse.Namespace) -> str:
    parts = [
        "segdata build mermaid",
        f"--stage-dir {args.stage_dir}",
        f"--annotations-uri {args.annotations_uri}",
        f"--annotation-date-cutoff {args.annotation_date_cutoff}",
    ]
    if args.parent_version is not None:
        parts.append(f"--parent-version {args.parent_version}")
        parts.append(f"--parent-manifest-sha256 {args.parent_manifest_sha256}")
    return " ".join(parts)


def _probe_all(
    client: Any,
    regions: Mapping[str, tuple[Any, Any]],
    parent_rows: Mapping[str, dict[str, Any]],
    workers: int,
) -> list[ImageOutcome]:
    image_ids = sorted(regions)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda i: probe_image(client, i, regions[i], parent_rows), image_ids))


def _dropped_counts(
    n_after_cutoff: int, outcomes: Sequence[ImageOutcome], points_per_image: Mapping[str, int]
) -> dict[str, int]:
    deleted = [o.image_id for o in outcomes if o.dropped == "image_deleted_at_source"]
    return {
        "annotation_after_cutoff": n_after_cutoff,
        "image_deleted_at_source": len(deleted),
        "points_of_deleted_images": sum(points_per_image[i] for i in deleted),
        "image_unreadable_header": sum(o.dropped == "image_unreadable_header" for o in outcomes),
    }


def _revision(file: AnnotationsFile, cutoff: str) -> str:
    return json.dumps(
        {"etag": file.etag, "last_modified": file.last_modified, "cutoff": cutoff},
        sort_keys=True,
        separators=(",", ":"),
    )


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a MERMAID points dataset version and its copy plan under ``args.stage_dir``.

    Raises:
        ValueError: Only one of the parent flags is given, or a URI is not ``s3://``.
        AnnotationsChanged: The annotations parquet was replaced during the read.
        SourceChanged: A source image was replaced between its HEAD and its read.
    """
    parents = _parent_refs(args)
    git = probe_git(REPO_ROOT)
    source = read_annotations(client, args.annotations_uri)
    annotations, n_after_cutoff = apply_cutoff(source.table, args.annotation_date_cutoff)
    parent_rows = load_parent_rows(args.stage_dir, args.parent_version)
    outcomes = _probe_all(client, first_regions(annotations), parent_rows, args.workers)
    kept = {o.image_id for o in outcomes if o.row is not None}
    points_per_image = annotations["image_id"].astype(str).value_counts().to_dict()
    dropped = _dropped_counts(n_after_cutoff, outcomes, points_per_image)
    labelled, pairs = with_label_ids(annotations[annotations["image_id"].astype(str).isin(kept)])
    rows = [o.row for o in outcomes if o.row is not None]
    stage = LocalDirS3(args.stage_dir)
    version = next_version(args.stage_dir, DATASET_BUCKET, DATASET, git_sha=git.sha, client=client)
    manifest, _ = publish_dataset(
        stage,
        bucket=DATASET_BUCKET,
        dataset=DATASET,
        version=version,
        kind="points",
        tables={
            "images.parquet": pa.Table.from_pylist(rows, schema=IMAGES_SCHEMA),
            "points.parquet": points_table(labelled),
            "labels.parquet": labels_table(pairs),
            "stats/label_counts.parquet": label_counts_table(labelled),
        },
        image_store=ImageStore(bucket=DATASET_BUCKET, prefix=IMAGE_PREFIX, owner="segdata"),
        etl_package="segdata-etl",
        etl_command=_command_line(args),
        source=SourceInfo(
            type="mermaid-confirmed-annotations",
            uri=args.annotations_uri,
            revision=_revision(source, args.annotation_date_cutoff),
            citation=None,
            license=None,
        ),
        parents=parents,
        label_mapping=LabelMapping(
            source="identity",
            sha256=labels_sha256(pairs),
            n_labels=len(pairs),
            n_mapped=len(pairs),
        ),
        dropped=dropped,
        repo_dir=REPO_ROOT,
        git=git,
    )
    plan = copy_plan(rows)
    _write_copies(args.stage_dir, version, plan)
    jpg_ids = sorted(o.image_id for o in outcomes if o.found_as_jpg and o.row is not None)
    if jpg_ids:
        logger.info("images found only as .jpg keys: %s", jpg_ids)
    logger.info(
        "staged mermaid %s: %d images, %d points, %d copies, %d found only as .jpg, dropped %s",
        version,
        manifest.stats.n_images,
        manifest.stats.n_points,
        len(plan),
        len(jpg_ids),
        dropped,
    )
    return 0


def _write_copies(stage_dir: Path, version: str, plan: Sequence[Mapping[str, Any]]) -> None:
    path = cache_path(stage_dir, DATASET_BUCKET, dataset_prefix(DATASET, version) + COPIES_NAME)
    path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in plan))
