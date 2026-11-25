"""Dataset-independent source probing and the promote plans shared by the staging builders.

``copies.jsonl`` lists server-side copies from a source bucket; ``uploads.jsonl`` lists staged files.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from segdata_etl.common.s3_objects import (
    ObjectConflict,
    SourceChanged,
    crc64nvme_b64,
    image_size,
    md5_hex,
    sniff_media_type,
)
from segdata_spec import DATASET_BUCKET
from segdata_spec._s3 import cache_path, dataset_prefix, is_precondition_failed

logger = logging.getLogger(__name__)

COPIES_NAME = "copies.jsonl"
UPLOADS_NAME = "uploads.jsonl"

_FIRST_READ = 64 * 1024
_MAX_READ = 1024 * 1024


@dataclass(frozen=True)
class ImageOutcome:
    """What probing one source image decided: a row to keep, a row to reuse, or a drop reason."""

    image_id: str
    row: dict[str, Any] | None = None
    reused: bool = False
    dropped: str | None = None
    found_as_jpg: bool = False


@dataclass(frozen=True)
class ObjectFacts:
    """The destination object's ETag, size and CRC64NVME, and the source ETag it was planned from."""

    etag: str
    size: int
    crc64nvme: str
    source_etag: str


def _read_range(client: Any, bucket: str, key: str, etag: str, last_byte: int) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{last_byte}", IfMatch=etag)
    data: bytes = response["Body"].read()
    return data


def read_image_header(
    client: Any, bucket: str, key: str, etag: str, size: int
) -> tuple[str, int, int] | None:
    """Return ``(media_type, width, height)`` from the object's leading bytes, or None if unreadable.

    The first 64 KB usually holds the header; a JPEG whose SOF lies further in is read up to 1 MB.

    Raises:
        SourceChanged: The object no longer matches ``etag``.
    """
    if size == 0:
        return None  # S3 answers a ranged GET on an empty object with 416
    try:
        head = _read_range(client, bucket, key, etag, _FIRST_READ - 1)
        media_type = sniff_media_type(head)
        if media_type is None:
            return None
        dims = image_size(head)
        if dims is None and media_type == "image/jpeg" and size > _FIRST_READ:
            dims = image_size(_read_range(client, bucket, key, etag, _MAX_READ - 1))
    except ClientError as exc:
        if is_precondition_failed(exc):
            raise SourceChanged(f"s3://{bucket}/{key} changed while it was read") from exc
        raise
    return None if dims is None else (media_type, dims[0], dims[1])


def needs_full_read(head: Mapping[str, Any]) -> bool:
    """A multipart ETag or a missing CRC64NVME means the copy would not carry the source's values."""
    return "-" in head["ETag"] or head.get("ChecksumCRC64NVME") is None


def read_full(client: Any, bucket: str, key: str, etag: str) -> bytes:
    """Download the whole object, guarded by ``etag``.

    Raises:
        SourceChanged: The object no longer matches ``etag``.
    """
    try:
        data: bytes = client.get_object(Bucket=bucket, Key=key, IfMatch=etag)["Body"].read()
    except ClientError as exc:
        if is_precondition_failed(exc):
            raise SourceChanged(f"s3://{bucket}/{key} changed while it was read") from exc
        raise
    return data


def read_whole_object(
    client: Any, bucket: str, key: str, head: Mapping[str, Any]
) -> tuple[tuple[str, int, int], ObjectFacts] | None:
    """Download the object and derive the header and the values a single-part copy will carry."""
    data = read_full(client, bucket, key, head["ETag"])
    media_type = sniff_media_type(data)
    dims = image_size(data)
    if media_type is None or dims is None:
        return None
    facts = ObjectFacts(md5_hex(data), len(data), crc64nvme_b64(data), head["ETag"].strip('"'))
    return (media_type, dims[0], dims[1]), facts


def probe_image(
    client: Any,
    bucket: str,
    image_id: str,
    source_key: str,
    head: Mapping[str, Any],
    parent_rows: Mapping[str, dict[str, Any]],
    build_row: Callable[[ObjectFacts, tuple[str, int, int]], dict[str, Any]],
    *,
    found_as_jpg: bool = False,
) -> ImageOutcome:
    """Decide one image from its source HEAD and, unless a parent row matches, its bytes.

    A multipart ETag or a missing CRC64NVME means the whole object is read, so the row records the
    ETag and CRC64NVME a single-part copy will have; the source ETag stays the fingerprint.
    ``build_row`` turns the object facts and header into the dataset's ``images.parquet`` row.
    """
    source_etag = head["ETag"].strip('"')
    parent = parent_rows.get(image_id)
    if parent is not None and parent["source_fingerprint"] == source_etag:
        return ImageOutcome(image_id, row=parent, reused=True, found_as_jpg=found_as_jpg)
    if needs_full_read(head):
        whole = read_whole_object(client, bucket, source_key, head)
        header, facts = whole if whole is not None else (None, None)
    else:
        header = read_image_header(client, bucket, source_key, head["ETag"], head["ContentLength"])
        facts = ObjectFacts(
            source_etag, head["ContentLength"], head["ChecksumCRC64NVME"], source_etag
        )
    if header is None or facts is None:
        logger.warning("dropping image %s: unreadable header in %s", image_id, source_key)
        return ImageOutcome(image_id, dropped="image_unreadable_header")
    return ImageOutcome(image_id, row=build_row(facts, header), found_as_jpg=found_as_jpg)


def copy_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    src_bucket: str,
    src_key: Callable[[Mapping[str, Any]], str],
    dst_prefix: str,
) -> list[dict[str, Any]]:
    """Describe the server-side copy of each image row, in ``image_id`` order.

    ``src_key`` reads a row's source object key. ``src_etag`` is the source object's ETag; ``etag``,
    ``size`` and ``crc64nvme`` are the destination object's expected values.
    """
    plan = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        plan.append(
            {
                "src_bucket": src_bucket,
                "src_key": src_key(row),
                "src_etag": row["source_fingerprint"],
                "etag": row["etag"],
                "dst_bucket": DATASET_BUCKET,
                "dst_key": dst_prefix + row["key"],
                "size": row["size"],
                "crc64nvme": row["crc64nvme"],
            }
        )
    return plan


def write_copies(
    stage_dir: Path, dataset: str, version: str, plan: Sequence[Mapping[str, Any]]
) -> None:
    """Write ``copies.jsonl`` for the staged version, one sorted-key JSON object per line."""
    path = cache_path(stage_dir, DATASET_BUCKET, dataset_prefix(dataset, version) + COPIES_NAME)
    path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in plan))


@dataclass(frozen=True)
class StagedFile:
    """One staged object and the upload that will put it in the dataset bucket."""

    key: str
    size: int
    crc64nvme: str
    content_type: str


def stage_bytes(stage_dir: Path, key: str, data: bytes) -> None:
    """Write ``data`` at dataset-bucket ``key`` in the stage; an existing file must hold the same bytes.

    Raises:
        ObjectConflict: The staged file exists with other bytes.
    """
    path = cache_path(stage_dir, DATASET_BUCKET, key)
    if path.exists():
        if path.read_bytes() != data:
            raise ObjectConflict(f"{path} already holds different bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def upload_entries(files: Sequence[StagedFile]) -> list[dict[str, Any]]:
    """Describe the upload of each staged file, in destination-key order."""
    return [
        {
            "local_path": f"{DATASET_BUCKET}/{f.key}",
            "dst_bucket": DATASET_BUCKET,
            "dst_key": f.key,
            "size": f.size,
            "crc64nvme": f.crc64nvme,
            "content_type": f.content_type,
        }
        for f in sorted(files, key=lambda f: f.key)
    ]


def write_uploads(stage_dir: Path, dataset: str, version: str, files: Sequence[StagedFile]) -> None:
    """Write ``uploads.jsonl`` for the staged version, one sorted-key JSON object per line."""
    path = cache_path(stage_dir, DATASET_BUCKET, dataset_prefix(dataset, version) + UPLOADS_NAME)
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in upload_entries(files)))
