from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from segdata_etl.common.publish import (
    S3Target,
    ensure_absent,
    list_keys,
    preflight,
    publish_dataset,
    put_json,
    put_parquet,
    upload_files,
)

B = "bkt"


def _target(dry_run: bool = False) -> S3Target:
    return S3Target(bucket=B, prefix="ds", dry_run=dry_run)


def _put_json(client, target, overwrite):
    return put_json(client, target, "x.json", {"a": 1}, overwrite=overwrite)


def _put_parquet(client, target, overwrite):
    df = pd.DataFrame({"k": [1]})
    return put_parquet(client, target, "x.parquet", df, sort_by=["k"], overwrite=overwrite)


PUTTERS = pytest.mark.parametrize(
    ("putter", "key"), [(_put_json, "ds/x.json"), (_put_parquet, "ds/x.parquet")]
)


def test_put_json_dry_run_writes_nothing_and_returns_uri(fake_s3):
    uri = _put_json(fake_s3, _target(dry_run=True), overwrite=False)
    assert uri == "s3://bkt/ds/x.json"
    assert fake_s3.objects == {}


@PUTTERS
def test_existing_key_without_overwrite_raises_and_keeps_object(fake_s3, putter, key):
    fake_s3.put(B, key, b"old")
    with pytest.raises(FileExistsError, match=f"s3://bkt/{key}.*--overwrite"):
        putter(fake_s3, _target(), overwrite=False)
    assert fake_s3.objects[(B, key)] == b"old"


@PUTTERS
def test_existing_key_with_overwrite_replaces_object(fake_s3, putter, key):
    fake_s3.put(B, key, b"old")
    uri = putter(fake_s3, _target(), overwrite=True)
    assert uri == f"s3://bkt/{key}"
    assert fake_s3.objects[(B, key)] != b"old"


@pytest.mark.parametrize("dry_run", [False, True])
def test_ensure_absent_raises_when_a_name_exists(fake_s3, dry_run):
    fake_s3.put(B, "ds/b.json", b"old")
    with pytest.raises(FileExistsError, match="s3://bkt/ds/b.json.*--overwrite"):
        ensure_absent(fake_s3, _target(dry_run), ["a.json", "b.json"], overwrite=False)


def test_ensure_absent_passes_when_none_exist_or_overwrite(fake_s3):
    ensure_absent(fake_s3, _target(), ["a.json"], overwrite=False)
    fake_s3.put(B, "ds/a.json", b"old")
    ensure_absent(fake_s3, _target(), ["a.json"], overwrite=True)


def test_put_json_keeps_insertion_order(fake_s3):
    put_json(fake_s3, _target(), "c.json", {"unlabeled": 0, "b": 1, "a": 2}, overwrite=False)
    assert fake_s3.objects[(B, "ds/c.json")] == (b'{\n  "unlabeled": 0,\n  "b": 1,\n  "a": 2\n}')


def test_put_parquet_sorts_rows_and_keeps_columns_and_dtypes(fake_s3):
    df = pd.DataFrame({"name": ["b", "a", "c"], "n": pd.Series([2, 1, 3], dtype="int32")})
    put_parquet(fake_s3, _target(), "x.parquet", df, sort_by=["name"], overwrite=False)
    got = pd.read_parquet(io.BytesIO(fake_s3.objects[(B, "ds/x.parquet")]))
    assert got.to_dict("list") == {"name": ["a", "b", "c"], "n": [1, 2, 3]}
    assert got["n"].dtype == "int32"


def _files(tmp_path: Path) -> list[tuple[Path, str]]:
    out = []
    for name in ("a.bin", "b.bin"):
        p = tmp_path / name
        p.write_bytes(name.encode())
        out.append((p, f"ds/images/{name}"))
    return out


def test_upload_files_skip_existing_uploads_only_missing(fake_s3, tmp_path):
    fake_s3.put(B, "ds/images/a.bin", b"old")
    n = upload_files(fake_s3, _target(), _files(tmp_path), workers=2, skip_existing=True)
    assert n == 1
    assert fake_s3.objects == {(B, "ds/images/a.bin"): b"old", (B, "ds/images/b.bin"): b"b.bin"}


def test_upload_files_without_skip_uploads_all(fake_s3, tmp_path):
    fake_s3.put(B, "ds/images/a.bin", b"old")
    n = upload_files(fake_s3, _target(), _files(tmp_path), workers=2, skip_existing=False)
    assert n == 2
    assert fake_s3.objects[(B, "ds/images/a.bin")] == b"a.bin"


def test_upload_files_dry_run_uploads_nothing_and_counts(fake_s3, tmp_path):
    n = upload_files(
        fake_s3, _target(dry_run=True), _files(tmp_path), workers=2, skip_existing=False
    )
    assert n == 2
    assert fake_s3.objects == {}


def test_list_keys_returns_keys_under_prefix(fake_s3):
    fake_s3.put(B, "ds/a", b"")
    fake_s3.put(B, "other/b", b"")
    assert list_keys(fake_s3, B, "ds/") == {"ds/a"}


def test_publish_dataset_writes_parquet_then_classes_colors_manifest_last(fake_s3):
    publish_dataset(
        fake_s3,
        _target(),
        dataset_name="demo",
        annotations=pd.DataFrame({"k": [1]}),
        sort_by=["k"],
        classes={"c": 1},
        colors={"c": [0, 0, 0]},
        manifest={"m": 1},
        overwrite=False,
    )
    assert list(fake_s3.objects) == [
        (B, "ds/demo_annotations.parquet"),
        (B, "ds/classes.json"),
        (B, "ds/colors.json"),
        (B, "ds/manifest.json"),
    ]


@pytest.mark.parametrize(
    "key",
    ["ds/demo_annotations.parquet", "ds/classes.json", "ds/colors.json", "ds/manifest.json"],
)
def test_preflight_raises_when_any_published_name_exists(fake_s3, key):
    fake_s3.put(B, key, b"old")
    with pytest.raises(FileExistsError, match=key):
        preflight(fake_s3, _target(), "demo", overwrite=False)


@pytest.mark.parametrize(
    "call",
    [
        lambda client, target: ensure_absent(client, target, ["a.json"], overwrite=False),
        lambda client, target: put_json(client, target, "a.json", {"a": 1}, overwrite=False),
    ],
    ids=["ensure_absent", "put_json"],
)
def test_head_error_other_than_missing_propagates_and_writes_nothing(fake_s3, monkeypatch, call):
    denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "HeadObject")

    def head_object(**kwargs):
        raise denied

    monkeypatch.setattr(fake_s3, "head_object", head_object)

    with pytest.raises(ClientError) as excinfo:
        call(fake_s3, _target())

    assert excinfo.value.response["Error"]["Code"] == "AccessDenied"
    assert fake_s3.objects == {}
