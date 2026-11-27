"""Tests for the ``segdata build coralnet`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from segdata_etl import cli
from segdata_etl.datasets.coralnet import build
from segdata_etl.datasets.coralnet.io import make_csv_reader_s3
from segdata_etl.datasets.coralnet.schemas import ANNOTATIONS_SCHEMA

_BUCKET = "b"
_PREFIX = "coralnet-public-images"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", _BUCKET)
    monkeypatch.delenv("MERMAID_CORALNET_PREFIX", raising=False)
    monkeypatch.delenv("MERMAID_CORALNET_ETL_WORKERS", raising=False)


def _patch_to_fake_s3(monkeypatch, fake_s3):
    """Point every S3 client and CSV reader factory at the in-memory FakeS3."""
    monkeypatch.setattr(build, "make_s3_client", lambda *_a, **_kw: fake_s3)
    monkeypatch.setattr(
        "segdata_etl.datasets.coralnet.annotations.make_ibis_connection", lambda: object()
    )
    monkeypatch.setattr(
        "segdata_etl.datasets.coralnet.annotations.make_csv_reader_ibis",
        lambda _con, _client: make_csv_reader_s3(fake_s3),
    )
    monkeypatch.setattr("segdata_etl.datasets.coralnet.images._get_thread_client", lambda: fake_s3)
    monkeypatch.setattr(
        "segdata_etl.datasets.coralnet.annotations.make_s3_client", lambda *_a, **_kw: fake_s3
    )


def _real_jpeg(width: int, height: int) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(0, 128, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def _common(tmp_path: Path, tag: str = "v1") -> list[str]:
    return ["--output-dir", str(tmp_path), "--workers", "1", "--version-tag", tag]


def test_build_annotations_restricts_to_source_ids(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path
):
    for sid in (3, 7, 12):
        populate(
            fake_s3,
            _BUCKET,
            _PREFIX,
            sid,
            sample_csv_factory(source_id=sid, n_images=1, n_points_per_image=1),
        )
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        ["build", "coralnet", "build-annotations", "--source-ids", "3,12", *_common(tmp_path)]
    )

    assert rc == 0
    ann = pd.read_parquet(tmp_path / "v1" / "coralnet_annotations_v1.parquet")
    assert sorted(ann["source_id"].unique().tolist()) == [3, 12]


def _write_annotations(path: Path, image_ids: list[str]) -> None:
    pd.DataFrame(
        {
            "source_id": [1] * len(image_ids),
            "image_id": image_ids,
            "row": [0] * len(image_ids),
            "col": [0] * len(image_ids),
            "coralnet_id": [1] * len(image_ids),
            "status": ["Confirmed"] * len(image_ids),
        }
    ).to_parquet(path, index=False)


def test_build_images_exits_1_on_missing_object_and_still_writes(
    monkeypatch, fake_s3, make_jpeg, tmp_path
):
    ann_path = tmp_path / "ann.parquet"
    _write_annotations(ann_path, ["10", "11"])
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10.jpg", make_jpeg(800, 600))
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        ["build", "coralnet", "build-images", "--annotations", str(ann_path), *_common(tmp_path)]
    )

    assert rc == 1
    images = pd.read_parquet(tmp_path / "v1" / "coralnet_images_v1.parquet")
    assert images["image_id"].tolist() == ["10", "11"]


def _pin_source_1(fake_s3) -> None:
    pin = {"dataset": "coralnet-raw", "version": "v", "manifest_sha256": "sha"}
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/_complete.json", json.dumps({"parent": pin}).encode())


def _build_images_rc(tmp_path: Path, ann_path: Path) -> int:
    argv = ["build", "coralnet", "build-images", "--annotations", str(ann_path)]
    return cli.main([*argv, *_common(tmp_path)])


def test_build_images_exits_0_when_only_a_pinned_source_lacks_objects(
    monkeypatch, fake_s3, make_jpeg, tmp_path, caplog
):
    ann_path = tmp_path / "ann.parquet"
    _write_annotations(ann_path, ["10", "11"])
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10.jpg", make_jpeg(800, 600))
    _pin_source_1(fake_s3)
    _patch_to_fake_s3(monkeypatch, fake_s3)
    caplog.set_level("INFO", logger="segdata_etl.datasets.coralnet")

    rc = _build_images_rc(tmp_path, ann_path)

    assert rc == 0
    assert "not_found in pinned sources: 1" in caplog.text
    images = pd.read_parquet(tmp_path / "v1" / "coralnet_images_v1.parquet")
    assert images["header_status"].tolist() == ["ok", "not_found"]


def test_build_images_exits_1_on_a_corrupt_image_in_a_pinned_source(
    monkeypatch, fake_s3, make_jpeg, tmp_path
):
    ann_path = tmp_path / "ann.parquet"
    _write_annotations(ann_path, ["10", "11"])
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10.jpg", make_jpeg(800, 600))
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/11.jpg", b"not an image at all")
    _pin_source_1(fake_s3)
    _patch_to_fake_s3(monkeypatch, fake_s3)

    assert _build_images_rc(tmp_path, ann_path) == 1


def test_cli_all_pipeline_end_to_end(monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path):
    populate(
        fake_s3,
        _BUCKET,
        _PREFIX,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=2),
    )
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10000.jpg", _real_jpeg(4096, 3000))
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10001.jpg", _real_jpeg(800, 600))
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(["build", "coralnet", "all", *_common(tmp_path)])

    assert rc == 0
    out_dir = tmp_path / "v1"
    ann_df = pd.read_parquet(out_dir / "coralnet_annotations_v1.parquet")
    img_df = pd.read_parquet(out_dir / "coralnet_images_v1.parquet")

    assert len(ann_df) == 4  # 2 imgs * 2 pts
    assert {f.name for f in ANNOTATIONS_SCHEMA}.issubset(set(ann_df.columns))
    assert (img_df["header_status"] == "ok").all()
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "coralnet_annotations_v1.parquet",
        "coralnet_id2name_v1.json",
        "coralnet_images_v1.parquet",
    ]
    assert not any(bucket == _BUCKET and "resized" in key for bucket, key in fake_s3.objects)


@pytest.mark.parametrize("raw", [""])
def test_empty_source_ids_is_a_usage_error(raw):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(
            ["build", "coralnet", "build-annotations", "--source-ids", raw]
        )
    assert exc.value.code == 2


def test_workers_flag_below_one_names_the_flag(tmp_path):
    with pytest.raises(ValueError, match="--workers must be >= 1, got 0"):
        cli.main(
            [
                "build",
                "coralnet",
                "build-images",
                "--annotations",
                "a.parquet",
                "--output-dir",
                str(tmp_path),
                "--workers",
                "0",
            ]
        )


def test_build_annotations_exits_1_when_no_sources_produce_rows(monkeypatch, fake_s3, tmp_path):
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(["build", "coralnet", "build-annotations", *_common(tmp_path)])

    assert rc == 1
    assert (tmp_path / "v1" / "coralnet_annotations_v1.parquet").exists()


def _upload_key(tag: str) -> str:
    return f"etl-outputs/coralnet/{tag}/coralnet_annotations_{tag}.parquet"


def _run_upload(monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path, extra):
    populate(
        fake_s3,
        _BUCKET,
        _PREFIX,
        1,
        sample_csv_factory(source_id=1, n_images=1, n_points_per_image=1),
    )
    _patch_to_fake_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(build, "compute_version_tag", lambda override=None: override or "auto")
    argv = [
        "build",
        "coralnet",
        "build-annotations",
        "--upload-to-s3",
        "--output-dir",
        str(tmp_path),
        "--workers",
        "1",
    ]
    return cli.main([*argv, *extra])


@pytest.mark.parametrize(
    ("tag", "extra"),
    [("v1", ["--version-tag", "v1"])],
    ids=["explicit-tag"],
)
def test_upload_refuses_to_overwrite_an_existing_key(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path, tag, extra
):
    fake_s3.put(_BUCKET, _upload_key(tag), b"old")

    with pytest.raises(RuntimeError, match=rf"{_upload_key(tag)}.*--overwrite"):
        _run_upload(monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path, extra)

    assert fake_s3.objects[(_BUCKET, _upload_key(tag))] == b"old"


def test_upload_overwrites_an_existing_key_with_overwrite_flag(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path
):
    fake_s3.put(_BUCKET, _upload_key("v1"), b"old")

    rc = _run_upload(
        monkeypatch,
        fake_s3,
        sample_csv_factory,
        populate,
        tmp_path,
        ["--version-tag", "v1", "--overwrite"],
    )

    assert rc == 0
    assert (
        fake_s3.objects[(_BUCKET, _upload_key("v1"))]
        == (tmp_path / "v1" / "coralnet_annotations_v1.parquet").read_bytes()
    )


def test_upload_writes_a_new_key(monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path):
    rc = _run_upload(monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path, [])

    assert rc == 0
    assert (
        fake_s3.objects[(_BUCKET, _upload_key("auto"))]
        == (tmp_path / "auto" / "coralnet_annotations_auto.parquet").read_bytes()
    )
    json_key = "etl-outputs/coralnet/auto/coralnet_id2name_auto.json"
    assert (
        fake_s3.objects[(_BUCKET, json_key)]
        == (tmp_path / "auto" / "coralnet_id2name_auto.json").read_bytes()
    )


def test_build_annotations_logs_exclusion_summary_and_exits_1(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path, caplog
):
    for sid, missing in (
        (1, None),
        (2, "image_list.csv"),
        (3, "labelset.csv"),
        (4, "labelset.csv"),
    ):
        files = sample_csv_factory(source_id=sid, n_images=1, n_points_per_image=1)
        if missing:
            files.pop(missing)
        populate(fake_s3, _BUCKET, _PREFIX, sid, files)
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(["build", "coralnet", "build-annotations", *_common(tmp_path)])

    assert rc == 1
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert "3 source(s) excluded" in errors
    assert "labelset_missing: [3, 4]" in errors


@pytest.mark.parametrize(
    ("excluded_source", "image_10001_present"),
    [(True, True), (False, False)],
    ids=["excluded-source", "missing-image"],
)
def test_cli_all_with_one_failing_stage_writes_every_output(
    monkeypatch,
    fake_s3,
    sample_csv_factory,
    populate,
    tmp_path,
    excluded_source: bool,
    image_10001_present: bool,
):
    populate(
        fake_s3,
        _BUCKET,
        _PREFIX,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=1),
    )
    if excluded_source:
        broken = sample_csv_factory(source_id=2, n_images=1, n_points_per_image=1)
        broken.pop("labelset.csv")
        populate(fake_s3, _BUCKET, _PREFIX, 2, broken)
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10000.jpg", _real_jpeg(800, 600))
    if image_10001_present:
        fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10001.jpg", _real_jpeg(800, 600))
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(["build", "coralnet", "all", *_common(tmp_path)])

    assert rc == 1
    out_dir = tmp_path / "v1"
    for name in (
        "coralnet_annotations_v1.parquet",
        "coralnet_images_v1.parquet",
    ):
        assert (out_dir / name).exists(), name
    assert json.loads((out_dir / "coralnet_id2name_v1.json").read_text()) == {
        "1000": "label_0",
        "1001": "label_1",
        "1002": "label_2",
        "1003": "label_3",
        "1004": "label_4",
        "1005": "label_5",
        "1006": "label_6",
    }
    img_df = pd.read_parquet(out_dir / "coralnet_images_v1.parquet")
    assert img_df["source_id"].unique().tolist() == [1]
    assert img_df["image_id"].tolist() == ["10000", "10001"]


def test_importing_cli_does_not_load_heavy_dependencies():
    """A fresh interpreter importing the CLI loads neither coralnet_scraper nor ibis."""
    code = (
        "import sys, segdata_etl.cli; "
        "print(sorted(m for m in ('coralnet_scraper', 'ibis') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
