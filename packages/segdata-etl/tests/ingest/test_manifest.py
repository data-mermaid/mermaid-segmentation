from __future__ import annotations

from datetime import datetime, timedelta

from segdata_etl.common.manifest import Source, build_manifest
from segdata_etl.common.release import S3Target


def test_build_manifest_has_the_settled_shape():
    source = Source(name="demo", citation="Doe 2020", raw_data_url="https://x/y", license="CC-BY")
    target = S3Target(bucket="bkt", prefix="ext/demo", dry_run=False)

    manifest = build_manifest(
        source,
        target,
        image_key_template="ext/demo/images/{image_id}.jpg",
        parquet_columns={"image_id": "string"},
        ingestion={"excluded_sites": ["s1"]},
    )

    created = manifest.pop("created_at_utc")
    assert datetime.fromisoformat(created).utcoffset() == timedelta(0)
    assert manifest == {
        "dataset_name": "demo",
        "source": {"citation": "Doe 2020", "raw_data_url": "https://x/y", "license": "CC-BY"},
        "s3": {
            "bucket": "bkt",
            "prefix": "ext/demo",
            "annotations_parquet": "ext/demo/demo_annotations.parquet",
            "image_key_template": "ext/demo/images/{image_id}.jpg",
        },
        "schema": {"parquet_columns": {"image_id": "string"}},
        "ingestion": {"excluded_sites": ["s1"]},
    }
