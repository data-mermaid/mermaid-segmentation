from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from segdata_etl.common.manifest import IngestSource, build_manifest, read_ingest_manifest
from segdata_etl.common.release import S3Target

SOURCE = IngestSource(citation="Doe 2020", raw_data_url="https://x/y", license="CC-BY")
TARGET = S3Target(bucket="bkt", prefix="ext/demo", dry_run=False)

# The manifest the Catlin ingest wrote before the typed contract: ``paper`` for the citation,
# no licence, and the damaged ids under ``skipped_damaged_or_missing_image_ids``.
OLD_CATLIN_MANIFEST = {
    "dataset_name": "catlin_seaview",
    "source": {"paper": "Gonzalez-Rivero 2019", "raw_data_url": "http://data.qld.edu.au"},
    "created_at_utc": "2026-01-01T00:00:00+00:00",
    "s3": {
        "bucket": "bkt",
        "prefix": "ext/catlin_seaview",
        "annotations_parquet": "ext/catlin_seaview/catlin_seaview_annotations.parquet",
        "image_key_template": "ext/catlin_seaview/images/{region}/{image_id}.jpg",
    },
    "schema": {"parquet_columns": {"image_id": "string"}},
    "ingestion": {"skipped_damaged_or_missing_image_ids": {"ATL": ["2"]}, "images_uploaded": {}},
}


def _built():
    return build_manifest(
        "demo",
        SOURCE,
        TARGET,
        image_key_template="ext/demo/images/{image_id}.jpg",
        parquet_columns={"image_id": "string"},
        dropped={"damaged_images": 2},
        images_uploaded=7,
        details={"excluded_sites": ["s1"]},
    )


def test_manifest_round_trips_through_json_with_every_field():
    manifest = _built()

    read = read_ingest_manifest(manifest.model_dump_json().encode())

    assert read == manifest
    assert datetime.fromisoformat(read.created_at_utc).utcoffset() == timedelta(0)
    body = json.loads(read.model_dump_json())
    del body["created_at_utc"]
    assert body == {
        "dataset_name": "demo",
        "source": {"citation": "Doe 2020", "raw_data_url": "https://x/y", "license": "CC-BY"},
        "s3": {
            "bucket": "bkt",
            "prefix": "ext/demo",
            "annotations_parquet": "ext/demo/demo_annotations.parquet",
            "image_key_template": "ext/demo/images/{image_id}.jpg",
        },
        "schema": {"parquet_columns": {"image_id": "string"}},
        "ingestion": {
            "dropped": {"damaged_images": 2},
            "images_uploaded": 7,
            "details": {"excluded_sites": ["s1"]},
        },
    }


def test_read_ingest_manifest_refuses_the_old_shape():
    with pytest.raises(ValidationError):
        read_ingest_manifest(json.dumps(OLD_CATLIN_MANIFEST).encode())


def test_ingest_source_requires_a_licence():
    with pytest.raises(ValidationError, match="license"):
        IngestSource(citation="Doe 2020", raw_data_url="https://x/y")  # type: ignore[call-arg]
