"""Reader for the S3 prefix ``segdata ingest`` writes: tables, typed manifest and object listing."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import pandas as pd
from botocore.client import BaseClient
from pydantic import ValidationError

from segdata_etl.common.manifest import IngestManifest, read_ingest_manifest
from segdata_etl.common.release import (
    CLASSES_NAME,
    INGEST_BUCKET,
    MANIFEST_NAME,
    annotations_parquet_name,
    ingest_prefix,
)
from segdata_spec.manifest import SourceInfo
from segdata_spec.s3 import get_bytes, norm_etag


@dataclass(frozen=True)
class IngestedObject:
    """One object under the ingest prefix: key relative to it, ETag without quotes, size."""

    key: str
    etag: str
    size: int


@dataclass
class IngestedDataset:
    """The tables, manifest and object listing of one ingested dataset prefix."""

    name: str
    bucket: str
    prefix: str
    annotations: pd.DataFrame
    classes: dict[str, int]
    manifest: IngestManifest
    manifest_sha256: str
    objects: dict[str, IngestedObject]


def _list_objects(client: BaseClient, bucket: str, prefix: str) -> dict[str, IngestedObject]:
    objects: dict[str, IngestedObject] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            rel = item["Key"].removeprefix(prefix)
            objects[rel] = IngestedObject(rel, norm_etag(item["ETag"]), int(item["Size"]))
    return objects


def _parse_manifest(manifest_raw: bytes, name: str, bucket: str) -> IngestManifest:
    """Parse the manifest bytes and check it describes the ``name`` ingest prefix."""
    manifest_uri = f"s3://{bucket}/{ingest_prefix(name)}/{MANIFEST_NAME}"
    try:
        manifest = read_ingest_manifest(manifest_raw)
    except ValidationError as exc:
        raise ValueError(
            f"{manifest_uri} is not the manifest shape `segdata ingest` writes; "
            f"re-run `segdata ingest {name}` for this prefix."
        ) from exc
    if manifest.dataset_name != name:
        raise ValueError(
            f"{manifest_uri} describes dataset {manifest.dataset_name!r}, expected {name!r}."
        )
    if manifest.s3.prefix != ingest_prefix(name):
        raise ValueError(
            f"{manifest_uri} records prefix {manifest.s3.prefix!r}, "
            f"expected {ingest_prefix(name)!r}."
        )
    return manifest


def read_ingested(client: BaseClient, name: str, *, bucket: str = INGEST_BUCKET) -> IngestedDataset:
    """Read the annotations, classes, manifest and object listing of an ingested dataset.

    Raises:
        ValueError: The manifest is not the shape ``segdata ingest`` writes, or it names a
            different dataset or prefix than ``name``.
    """
    prefix = f"{ingest_prefix(name)}/"
    manifest_raw = get_bytes(client, bucket, f"{prefix}{MANIFEST_NAME}")
    manifest = _parse_manifest(manifest_raw, name, bucket)
    annotations = pd.read_parquet(
        io.BytesIO(get_bytes(client, bucket, f"{prefix}{annotations_parquet_name(name)}"))
    )
    return IngestedDataset(
        name=name,
        bucket=bucket,
        prefix=prefix,
        annotations=annotations,
        classes=json.loads(get_bytes(client, bucket, f"{prefix}{CLASSES_NAME}")),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        objects=_list_objects(client, bucket, prefix),
    )


def source_revision(ds: IngestedDataset) -> str:
    """Return a revision string covering the manifest bytes and every object's key and ETag."""
    lines = "".join(
        f"{o.key}\t{o.etag}\n" for o in sorted(ds.objects.values(), key=lambda o: o.key)
    )
    objects_digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    return f"manifest-sha256:{ds.manifest_sha256};objects-sha256:{objects_digest}"


def source_info(ds: IngestedDataset) -> SourceInfo:
    """Describe the ingest prefix as a manifest ``SourceInfo``."""
    return SourceInfo(
        type="ingest-prefix",
        uri=f"s3://{ds.bucket}/{ds.prefix}",
        revision=source_revision(ds),
        citation=ds.manifest.source.citation,
        license=ds.manifest.source.license,
    )
