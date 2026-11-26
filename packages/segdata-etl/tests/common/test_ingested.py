"""Tests for the ingest-prefix reader."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from typing import Any

import pandas as pd
from PIL import Image

from segdata_etl.common.ingested import (
    IngestedObject,
    read_ingested,
    source_info,
    source_revision,
)
from segdata_etl.common.manifest import IngestSource, build_manifest
from segdata_etl.common.release import INGEST_BUCKET, S3Target, ingest_prefix

BUCKET = INGEST_BUCKET


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, client: ListingS3):
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:  # noqa: N803
        keys = [k for (b, k) in self._client.objects if b == Bucket and k.startswith(Prefix)]
        if self._client.reverse_listing:
            keys.reverse()
        for i in range(0, len(keys), 2):
            yield {"Contents": [self._entry(Bucket, k) for k in keys[i : i + 2]]}

    def _entry(self, bucket: str, key: str) -> dict[str, Any]:
        data = self._client.objects[(bucket, key)]
        etag = hashlib.md5(data).hexdigest()  # noqa: S324
        return {"Key": key, "ETag": f'"{etag}"', "Size": len(data)}


class ListingS3:
    """In-memory S3 client with a paginated listing and whole-object GETs."""

    def __init__(self, objects: dict[tuple[str, str], bytes], reverse_listing: bool = False):
        self.objects = objects
        self.reverse_listing = reverse_listing

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return _Paginator(self)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        return {"Body": _Body(self.objects[(Bucket, Key)])}


def _image_bytes(fmt: str, size: tuple[int, int], **save_kwargs: Any) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _parquet_bytes() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({"image_id": ["a", "b"], "row": [1, 2], "col": [3, 4]}).to_parquet(buf)
    return buf.getvalue()


MANIFEST = build_manifest(
    "demo",
    IngestSource(citation="A. Author 2020", raw_data_url="https://x/y", license="CC BY 4.0"),
    S3Target(BUCKET, "external_validation_datasets/demo", dry_run=False),
    image_key_template="images/{group}/{image_id}.jpg",
    parquet_columns={"image_id": "string"},
    dropped={"damaged_images": 4},
    images_uploaded=2,
)
MANIFEST_BYTES = MANIFEST.model_dump_json().encode()
CLASSES = {"unlabeled": 0, "coral": 1}


def _ingested_objects(name: str = "demo") -> dict[tuple[str, str], bytes]:
    prefix = f"{ingest_prefix(name)}/"
    return {
        (BUCKET, f"{prefix}{name}_annotations.parquet"): _parquet_bytes(),
        (BUCKET, f"{prefix}classes.json"): json.dumps(CLASSES).encode(),
        (BUCKET, f"{prefix}manifest.json"): MANIFEST_BYTES,
        (BUCKET, f"{prefix}images/g1/a.jpg"): _image_bytes("JPEG", (7, 5)),
        (BUCKET, f"{prefix}images/g2/b.jpg"): _image_bytes("JPEG", (7, 5), quality=50),
    }


def test_read_ingested_returns_tables_manifest_and_stripped_etag_objects() -> None:
    objects = _ingested_objects()
    client = ListingS3(objects)

    ds = read_ingested(client, "demo")

    jpeg = objects[(BUCKET, "external_validation_datasets/demo/images/g1/a.jpg")]
    assert ds.bucket == "dev-datamermaid-sm-sources"
    assert ds.prefix == "external_validation_datasets/demo/"
    assert ds.annotations["image_id"].tolist() == ["a", "b"]
    assert ds.classes == CLASSES
    assert ds.manifest == MANIFEST
    assert ds.manifest.ingestion.dropped == {"damaged_images": 4}
    assert ds.manifest_sha256 == hashlib.sha256(MANIFEST_BYTES).hexdigest()
    assert ds.objects["images/g1/a.jpg"] == IngestedObject(
        key="images/g1/a.jpg",
        etag=hashlib.md5(jpeg).hexdigest(),
        size=len(jpeg),  # noqa: S324
    )
    assert len(ds.objects) == 5


def test_source_revision_ignores_listing_order_and_tracks_every_etag() -> None:
    objects = _ingested_objects()
    forward = read_ingested(ListingS3(objects), "demo")
    backward = read_ingested(ListingS3(objects, reverse_listing=True), "demo")
    changed_objects = dict(objects)
    changed_objects[(BUCKET, "external_validation_datasets/demo/images/g2/b.jpg")] = _image_bytes(
        "JPEG", (7, 5), quality=10
    )
    changed = read_ingested(ListingS3(changed_objects), "demo")

    lines = "".join(
        f"{o.key}\t{o.etag}\n" for o in sorted(forward.objects.values(), key=lambda o: o.key)
    )
    expected = (
        f"manifest-sha256:{forward.manifest_sha256};"
        f"objects-sha256:{hashlib.sha256(lines.encode()).hexdigest()}"
    )
    assert source_revision(forward) == expected
    assert source_revision(backward) == expected
    assert source_revision(changed) != expected


def test_source_info_takes_citation_and_license_from_the_manifest() -> None:
    ds = read_ingested(ListingS3(_ingested_objects()), "demo")

    info = source_info(ds)

    assert info.type == "ingest-prefix"
    assert info.uri == f"s3://{BUCKET}/external_validation_datasets/demo/"
    assert info.revision == source_revision(ds)
    assert info.citation == "A. Author 2020"
    assert info.license == "CC BY 4.0"
