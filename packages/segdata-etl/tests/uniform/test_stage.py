from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest
from botocore.exceptions import ClientError

from segdata_etl.common.stage import LocalDirS3, next_version
from segdata_spec._s3 import is_not_found, is_precondition_failed
from segdata_spec.manifest import ImageStore, SourceInfo
from segdata_spec.publish import GitState, publish_dataset

from .conftest import jpeg_bytes, md5_etag


def test_local_put_round_trips_to_get_and_head(tmp_path):
    client = LocalDirS3(tmp_path)
    client.put_object(Bucket="bkt", Key="a/b.bin", Body=b"hello", ContentType="text/plain")
    assert (tmp_path / "bkt" / "a" / "b.bin").read_bytes() == b"hello"
    assert client.get_object(Bucket="bkt", Key="a/b.bin")["Body"].read() == b"hello"
    head = client.head_object(Bucket="bkt", Key="a/b.bin")
    assert head["ContentLength"] == 5
    assert head["ETag"] == md5_etag(b"hello")


def test_local_put_if_none_match_on_existing_key_is_precondition_failed(tmp_path):
    client = LocalDirS3(tmp_path)
    client.put_object(Bucket="bkt", Key="k", Body=b"one")
    with pytest.raises(ClientError) as info:
        client.put_object(Bucket="bkt", Key="k", Body=b"two", IfNoneMatch="*")
    assert is_precondition_failed(info.value)
    assert (tmp_path / "bkt" / "k").read_bytes() == b"one"


def test_local_missing_key_is_not_found(tmp_path):
    client = LocalDirS3(tmp_path)
    with pytest.raises(ClientError) as head_info:
        client.head_object(Bucket="bkt", Key="missing")
    with pytest.raises(ClientError) as get_info:
        client.get_object(Bucket="bkt", Key="missing")
    assert is_not_found(head_info.value)
    assert is_not_found(get_info.value)


def test_publish_dataset_into_local_dir_matches_manifest_checksums(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_bytes(b"lock-bytes")
    jpg = jpeg_bytes(7, 5)
    mask = b"mask"
    images = pa.table(
        {
            "image_id": ["a"],
            "key": ["a.jpg"],
            "etag": [md5_etag(jpg).strip('"')],
            "size": [len(jpg)],
            "crc64nvme": pa.nulls(1, pa.string()),
            "media_type": ["image/jpeg"],
            "width": [7],
            "height": [5],
            "group": ["g1"],
            "mask_key": ["a.png"],
            "mask_etag": [md5_etag(mask).strip('"')],
            "mask_size": [len(mask)],
            "mask_crc64nvme": pa.nulls(1, pa.string()),
            "source_fingerprint": ["fp"],
            "source_name": pa.nulls(1, pa.string()),
            "attrs": pa.nulls(1, pa.string()),
        }
    )
    labels = pa.table(
        {
            "label_id": [5],
            "name": ["x"],
            "mermaid_benthic_attribute_id": pa.nulls(1, pa.string()),
            "mermaid_benthic_attribute": pa.nulls(1, pa.string()),
            "mermaid_growth_form": pa.nulls(1, pa.string()),
            "mapping_source": ["curated"],
        }
    )
    counts = pa.table({"image_id": ["a"], "label_id": [5], "n": [1]})
    client = LocalDirS3(tmp_path / "stage")
    publish_dataset(
        client,
        bucket="bkt",
        dataset="ds",
        version="20261123-abc1234",
        kind="dense",
        tables={
            "images.parquet": images,
            "labels.parquet": labels,
            "stats/label_counts.parquet": counts,
        },
        image_store=ImageStore(bucket="img", prefix="images/ds/", owner="segdata"),
        etl_package="segdata-etl",
        etl_command="test",
        source=SourceInfo(type="test", uri="s3://src/", revision="r1"),
        repo_dir=repo,
        git=GitState(sha="abcdef1234567", dirty=False, pushed=True),
    )
    root = tmp_path / "stage" / "bkt" / "datasets" / "ds" / "20261123-abc1234"
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert set(manifest["files"]) == {
        "images.parquet",
        "labels.parquet",
        "stats/label_counts.parquet",
    }
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == entry["sha256"]


def test_next_version_appends_first_free_counter(tmp_path: Path):
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678"}
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234"
    base = tmp_path / "bkt" / "datasets" / "ds"
    (base / "20261123-abc1234").mkdir(parents=True)
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
    (base / "20261123-abc1234-2").mkdir()
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-3"
