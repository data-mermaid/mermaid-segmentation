"""Reader for the legacy ``external_validation_datasets/<name>/`` S3 prefix.

The legacy prefixes predate the current ingest code: their ``manifest.json`` shapes differ per
dataset, so the drop counts and source metadata are read here rather than from a shared schema.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass
from typing import Any

import pandas as pd
from botocore.client import BaseClient

from segdata_etl.common.jpeg_header import _find_sof_dimensions
from segdata_spec.errors import ImageUndecodable
from segdata_spec.manifest import SourceInfo

DEFAULT_BUCKET = "dev-datamermaid-sm-sources"
_FIRST_CHUNK = 65_536
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_HEADER_LEN = 24  # signature(8) + IHDR length/type(8) + width(4) + height(4)


@dataclass(frozen=True)
class LegacyObject:
    """One object under the legacy prefix: key relative to it, ETag without quotes, size."""

    key: str
    etag: str
    size: int


@dataclass
class LegacyDataset:
    """The tables, manifest and object listing of one legacy dataset prefix."""

    name: str
    bucket: str
    prefix: str
    annotations: pd.DataFrame
    classes: dict[str, int]
    manifest: dict
    manifest_sha256: str
    objects: dict[str, LegacyObject]


def _get_bytes(client: BaseClient, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _list_objects(client: BaseClient, bucket: str, prefix: str) -> dict[str, LegacyObject]:
    objects: dict[str, LegacyObject] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            rel = item["Key"].removeprefix(prefix)
            objects[rel] = LegacyObject(rel, item["ETag"].strip('"'), int(item["Size"]))
    return objects


def read_legacy(
    client: BaseClient,
    name: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    prefix: str | None = None,
) -> LegacyDataset:
    """Read the annotations, classes, manifest and object listing of a legacy dataset."""
    prefix = prefix if prefix is not None else f"external_validation_datasets/{name}/"
    manifest_raw = _get_bytes(client, bucket, f"{prefix}manifest.json")
    annotations = pd.read_parquet(
        io.BytesIO(_get_bytes(client, bucket, f"{prefix}{name}_annotations.parquet"))
    )
    return LegacyDataset(
        name=name,
        bucket=bucket,
        prefix=prefix,
        annotations=annotations,
        classes=json.loads(_get_bytes(client, bucket, f"{prefix}classes.json")),
        manifest=json.loads(manifest_raw),
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        objects=_list_objects(client, bucket, prefix),
    )


def source_revision(ds: LegacyDataset) -> str:
    """Return a revision string covering the manifest bytes and every object's key and ETag."""
    lines = "".join(
        f"{o.key}\t{o.etag}\n" for o in sorted(ds.objects.values(), key=lambda o: o.key)
    )
    objects_digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    return f"manifest-sha256:{ds.manifest_sha256};objects-sha256:{objects_digest}"


def legacy_dropped(name: str, manifest: dict) -> dict[str, int]:
    """Return the drop counts a legacy manifest records.

    Required manifest fields are indexed strictly, so a missing field raises ``KeyError``.
    """
    if name == "catlin_seaview":
        skipped = manifest["ingestion"]["skipped_damaged_or_missing_image_ids"]
        return {"damaged_images": sum(len(ids) for ids in skipped.values())}
    if name == "moorea_labeled_corals":
        return {
            "damaged_images": len(manifest["damaged_images"]),
            "off_annotations": int(manifest["off_annotations_dropped"]),
            "case_merged_labels": len(manifest["case_merged_labels"]),
        }
    if name in ("pacific_labeled_corals", "benthos_yuval"):
        return {}
    raise ValueError(f"unknown legacy dataset {name!r}")


def _png_dimensions(head: bytes) -> tuple[int, int]:
    if len(head) < _PNG_HEADER_LEN:
        raise ImageUndecodable("truncated PNG header")
    width, height = struct.unpack(">II", head[16:24])
    return width, height


def sniff_image(client: BaseClient, bucket: str, key: str) -> tuple[str, int, int]:
    """Return ``(media_type, width, height)`` of a JPEG or PNG from its header bytes."""
    head = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{_FIRST_CHUNK - 1}")[
        "Body"
    ].read()
    if head.startswith(_PNG_SIGNATURE):
        width, height = _png_dimensions(head)
        return "image/png", width, height
    if head[:2] == b"\xff\xd8":
        dims = _find_sof_dimensions(head)
        if dims is None:
            dims = _find_sof_dimensions(_get_bytes(client, bucket, key))
        if dims is None:
            raise ImageUndecodable(f"no JPEG SOF marker in s3://{bucket}/{key}")
        return "image/jpeg", dims[1], dims[0]
    raise ImageUndecodable(f"s3://{bucket}/{key} is neither JPEG nor PNG")


def source_info(ds: LegacyDataset, *, license_override: str | None = None) -> SourceInfo:
    """Describe the legacy prefix as a manifest ``SourceInfo``.

    Raises ``ValueError`` when neither the override nor the manifest supplies a licence.
    """
    source: dict[str, Any] = ds.manifest.get("source", {})
    license_text = license_override or source.get("license")
    if not license_text:
        raise ValueError(f"no licence for legacy dataset {ds.name!r}: pass license_override")
    return SourceInfo(
        type="legacy-prefix",
        uri=f"s3://{ds.bucket}/{ds.prefix}",
        revision=source_revision(ds),
        citation=source.get("citation") or source.get("paper"),
        license=license_text,
    )
