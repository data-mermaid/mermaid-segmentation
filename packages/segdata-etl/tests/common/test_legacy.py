"""Tests for the legacy external_validation_datasets prefix reader."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from segdata_etl.common.legacy import (
    LegacyObject,
    legacy_dropped,
    read_legacy,
    sniff_image,
    source_info,
    source_revision,
)
from segdata_spec.errors import ImageUndecodable

BUCKET = "dev-datamermaid-sm-sources"


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, client: LegacyS3):
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


class LegacyS3:
    """In-memory S3 client with a paginated listing and ranged GETs."""

    def __init__(self, objects: dict[tuple[str, str], bytes], reverse_listing: bool = False):
        self.objects = objects
        self.reverse_listing = reverse_listing
        self.get_calls: list[dict[str, Any]] = []

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2"
        return _Paginator(self)

    def get_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Range: str | None = None
    ) -> dict[str, Any]:
        self.get_calls.append({"Key": Key, "Range": Range})
        data = self.objects[(Bucket, Key)]
        if Range is None:
            return {"Body": _Body(data)}
        start, end = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        return {
            "Body": _Body(data[start : end + 1]),
            "ContentRange": f"bytes {start}-{min(end, len(data) - 1)}/{len(data)}",
        }


def _image_bytes(fmt: str, size: tuple[int, int], **save_kwargs: Any) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _parquet_bytes() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({"image_id": ["a", "b"], "row": [1, 2], "col": [3, 4]}).to_parquet(buf)
    return buf.getvalue()


MANIFEST = {
    "source": {"citation": "A. Author 2020", "paper": "P. Paper 2019", "license": "CC BY 4.0"}
}
CLASSES = {"unlabeled": 0, "coral": 1}


def _legacy_objects(name: str = "demo", prefix: str | None = None) -> dict[tuple[str, str], bytes]:
    prefix = prefix or f"external_validation_datasets/{name}/"
    return {
        (BUCKET, f"{prefix}{name}_annotations.parquet"): _parquet_bytes(),
        (BUCKET, f"{prefix}classes.json"): json.dumps(CLASSES).encode(),
        (BUCKET, f"{prefix}manifest.json"): json.dumps(MANIFEST).encode(),
        (BUCKET, f"{prefix}images/g1/a.jpg"): _image_bytes("JPEG", (7, 5)),
        (BUCKET, f"{prefix}images/g2/b.jpg"): _image_bytes("JPEG", (7, 5), quality=50),
    }


def test_sniff_image_reads_media_type_and_width_height_from_bytes() -> None:
    client = LegacyS3(
        {
            (BUCKET, "a.jpg"): _image_bytes("JPEG", (7, 5)),
            (BUCKET, "b.png"): _image_bytes("PNG", (6, 4)),
        }
    )

    assert sniff_image(client, BUCKET, "a.jpg") == ("image/jpeg", 7, 5)
    assert sniff_image(client, BUCKET, "b.png") == ("image/png", 6, 4)


def test_sniff_image_falls_back_to_full_get_when_sof_is_past_the_first_chunk() -> None:
    data = _image_bytes("JPEG", (9, 3), icc_profile=b"\x00" * 150_000)
    client = LegacyS3({(BUCKET, "big.jpg"): data})

    assert sniff_image(client, BUCKET, "big.jpg") == ("image/jpeg", 9, 3)
    assert [c["Range"] for c in client.get_calls] == ["bytes=0-65535", None]


def test_sniff_image_raises_for_bytes_that_are_not_jpeg_or_png() -> None:
    client = LegacyS3({(BUCKET, "x.gif"): _image_bytes("GIF", (4, 4))})

    with pytest.raises(ImageUndecodable):
        sniff_image(client, BUCKET, "x.gif")


def test_read_legacy_returns_tables_manifest_and_stripped_etag_objects() -> None:
    objects = _legacy_objects()
    client = LegacyS3(objects)

    ds = read_legacy(client, "demo")

    jpeg = objects[(BUCKET, "external_validation_datasets/demo/images/g1/a.jpg")]
    assert ds.bucket == BUCKET
    assert ds.prefix == "external_validation_datasets/demo/"
    assert ds.annotations["image_id"].tolist() == ["a", "b"]
    assert ds.classes == CLASSES
    assert ds.manifest == MANIFEST
    assert ds.manifest_sha256 == hashlib.sha256(json.dumps(MANIFEST).encode()).hexdigest()
    assert ds.objects["images/g1/a.jpg"] == LegacyObject(
        key="images/g1/a.jpg",
        etag=hashlib.md5(jpeg).hexdigest(),
        size=len(jpeg),  # noqa: S324
    )
    assert len(ds.objects) == 5


def test_source_revision_ignores_listing_order_and_tracks_every_etag() -> None:
    objects = _legacy_objects()
    forward = read_legacy(LegacyS3(objects), "demo")
    backward = read_legacy(LegacyS3(objects, reverse_listing=True), "demo")
    changed_objects = dict(objects)
    changed_objects[(BUCKET, "external_validation_datasets/demo/images/g2/b.jpg")] = _image_bytes(
        "JPEG", (7, 5), quality=10
    )
    changed = read_legacy(LegacyS3(changed_objects), "demo")

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


def test_legacy_dropped_counts_catlin_ids_across_regions() -> None:
    manifest = {
        "ingestion": {
            "skipped_damaged_or_missing_image_ids": {"PAC_IDN_PHL": ["a", "b", "c"], "ATL": ["d"]}
        }
    }

    assert legacy_dropped("catlin_seaview", manifest) == {"damaged_images": 4}


def test_legacy_dropped_counts_moorea_damaged_off_and_merged() -> None:
    manifest = {
        "damaged_images": ["2008/a.jpg", "2009/b.jpg"],
        "off_annotations_dropped": 11,
        "case_merged_labels": {"Acrop": "acrop", "TURF": "turf", "Sand": "sand"},
    }

    assert legacy_dropped("moorea_labeled_corals", manifest) == {
        "damaged_images": 2,
        "off_annotations": 11,
        "case_merged_labels": 3,
    }


@pytest.mark.parametrize("name", ["pacific_labeled_corals", "benthos_yuval"])
def test_legacy_dropped_is_empty_for_datasets_without_drop_counts(name: str) -> None:
    assert legacy_dropped(name, {"source": {}, "stats": {"s1": 4}}) == {}


@pytest.mark.parametrize(
    ("name", "manifest"),
    [
        ("catlin_seaview", {"source": {}}),
        ("moorea_labeled_corals", {"damaged_images": [], "off_annotations_dropped": 0}),
    ],
)
def test_legacy_dropped_raises_key_error_when_a_required_field_is_absent(
    name: str, manifest: dict
) -> None:
    with pytest.raises(KeyError):
        legacy_dropped(name, manifest)


def test_source_info_takes_citation_over_paper_and_the_manifest_license() -> None:
    ds = read_legacy(LegacyS3(_legacy_objects()), "demo")

    info = source_info(ds)

    assert info.type == "legacy-prefix"
    assert info.uri == f"s3://{BUCKET}/external_validation_datasets/demo/"
    assert info.revision == source_revision(ds)
    assert info.citation == "A. Author 2020"
    assert info.license == "CC BY 4.0"


def test_source_info_falls_back_to_paper_and_honours_the_license_override() -> None:
    objects = _legacy_objects()
    key = (BUCKET, "external_validation_datasets/demo/manifest.json")
    objects[key] = json.dumps({"source": {"paper": "P. Paper 2019", "license": "x"}}).encode()
    ds = read_legacy(LegacyS3(objects), "demo")

    info = source_info(ds, license_override="CC BY-NC-SA 4.0")

    assert info.citation == "P. Paper 2019"
    assert info.license == "CC BY-NC-SA 4.0"


@pytest.mark.parametrize("source", [{"citation": "A. Author 2020"}, {"license": ""}])
def test_source_info_raises_when_no_licence_is_resolved(source: dict) -> None:
    objects = _legacy_objects()
    key = (BUCKET, "external_validation_datasets/demo/manifest.json")
    objects[key] = json.dumps({"source": source}).encode()
    ds = read_legacy(LegacyS3(objects), "demo")

    with pytest.raises(ValueError):
        source_info(ds)
