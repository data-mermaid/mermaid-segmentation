"""Tests for the ``segdata build coralnet`` CLI."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from PIL import Image

from segdata_etl import cli
from segdata_etl.datasets.coralnet import build
from segdata_etl.datasets.coralnet.io import make_csv_reader_s3, write_parquet_deterministic
from segdata_etl.datasets.coralnet.schemas import (
    ANNOTATIONS_PRIMARY_KEY,
    ANNOTATIONS_SCHEMA,
    IMAGES_PRIMARY_KEY,
    IMAGES_SCHEMA,
)

_BUCKET = "b"
_PREFIX = "coralnet-public-images"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", _BUCKET)
    monkeypatch.delenv("MERMAID_CORALNET_PREFIX", raising=False)
    monkeypatch.delenv("MERMAID_CORALNET_RESIZED_PREFIX", raising=False)
    monkeypatch.delenv("MERMAID_CORALNET_ETL_WORKERS", raising=False)


def _patch_to_fake_s3(monkeypatch, fake_s3):
    """Point every S3 client and CSV reader factory at the in-memory FakeS3."""
    monkeypatch.setattr(build, "make_s3_client", lambda: fake_s3)
    monkeypatch.setattr(build, "make_resize_s3_client", lambda **_: fake_s3)
    monkeypatch.setattr(
        "segdata_etl.datasets.coralnet.annotations.make_ibis_connection", lambda: object()
    )
    monkeypatch.setattr(
        "segdata_etl.datasets.coralnet.annotations.make_csv_reader_ibis",
        lambda _con: make_csv_reader_s3(fake_s3),
    )
    monkeypatch.setattr("segdata_etl.datasets.coralnet.images._get_thread_client", lambda: fake_s3)
    monkeypatch.setattr("segdata_etl.datasets.coralnet.annotations.make_s3_client", lambda: fake_s3)


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


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 48)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    "second_image",
    [lambda make_jpeg: make_jpeg(640, 480, padding=80_000), lambda make_jpeg: _png_bytes()],
    ids=["sof_not_found_extended", "pil_fallback"],
)
def test_build_images_exits_0_when_every_image_has_dimensions(
    monkeypatch, fake_s3, make_jpeg, tmp_path, second_image
):
    ann_path = tmp_path / "ann.parquet"
    _write_annotations(ann_path, ["10", "11"])
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10.jpg", make_jpeg(800, 600))
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/11.jpg", second_image(make_jpeg))
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        ["build", "coralnet", "build-images", "--annotations", str(ann_path), *_common(tmp_path)]
    )

    assert rc == 0


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


def _write_images(path: Path, needs_resize: bool) -> None:
    df = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["10"],
            "s3_key": [f"{_PREFIX}/s1/images/10.jpg"],
            "width": [4096 if needs_resize else 800],
            "height": [3000 if needs_resize else 600],
            "longest_edge": [4096 if needs_resize else 800],
            "file_size": [1000],
            "needs_resize": [needs_resize],
            "header_status": ["ok"],
            "error_message": [None],
        }
    )
    write_parquet_deterministic(df, path, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)


@pytest.mark.parametrize(
    ("needs_resize", "counts", "expected_rc"),
    [
        (True, (1, 0, 0), 0),
        (True, (0, 0, 1), 1),
        (True, (0, 1, 0), 1),
        (False, None, 0),
    ],
    ids=["all-completed", "any-failed", "any-corrupted", "nothing-to-do"],
)
def test_resize_exit_code(monkeypatch, fake_s3, tmp_path, needs_resize, counts, expected_rc):
    images_path = tmp_path / "images.parquet"
    _write_images(images_path, needs_resize)
    _patch_to_fake_s3(monkeypatch, fake_s3)
    calls: list[int] = []

    def fake_resize_all(df_todo, bucket, **_kwargs):
        calls.append(len(df_todo))
        return counts

    monkeypatch.setattr(build, "resize_and_upload_all_images", fake_resize_all)

    rc = cli.main(["build", "coralnet", "resize", "--images", str(images_path), *_common(tmp_path)])

    assert rc == expected_rc
    assert calls == ([1] if needs_resize else [])


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
    manifest = pd.read_parquet(out_dir / "coralnet_training_manifest_v1.parquet")

    assert len(ann_df) == 4  # 2 imgs * 2 pts
    assert {f.name for f in ANNOTATIONS_SCHEMA}.issubset(set(ann_df.columns))
    assert (img_df["header_status"] == "ok").all()
    assert (_BUCKET, "dev/images/resized/s1/images/10000.jpg") in fake_s3.objects
    keys = dict(zip(manifest["image_id"], manifest["image_s3_key"], strict=False))
    assert keys == {
        "10000": "dev/images/resized/s1/images/10000.jpg",
        "10001": f"{_PREFIX}/s1/images/10001.jpg",
    }
    assert len(manifest) == 4


def test_cli_all_runs_every_stage_after_a_failure(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path
):
    populate(
        fake_s3,
        _BUCKET,
        _PREFIX,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=2),
    )
    # Image 10001 is missing, so build-images returns 1.
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10000.jpg", _real_jpeg(800, 600))
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(["build", "coralnet", "all", *_common(tmp_path)])

    assert rc == 1
    manifest = pd.read_parquet(tmp_path / "v1" / "coralnet_training_manifest_v1.parquet")
    assert sorted(manifest["image_id"].unique().tolist()) == ["10000"]


@pytest.mark.parametrize("raw", ["", ","])
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


_MANIFEST_COLUMNS = [
    "source_id",
    "image_id",
    "row",
    "col",
    "coralnet_id",
    "source_label_name",
    "image_s3_key",
    "load_width",
    "load_height",
    "uses_resized_image",
]


def test_build_manifest_on_empty_inputs_writes_empty_manifest(monkeypatch, fake_s3, tmp_path):
    ann_path, img_path = tmp_path / "ann.parquet", tmp_path / "img.parquet"
    write_parquet_deterministic(
        ANNOTATIONS_SCHEMA.empty_table().to_pandas(),
        ann_path,
        ANNOTATIONS_SCHEMA,
        ANNOTATIONS_PRIMARY_KEY,
    )
    write_parquet_deterministic(
        IMAGES_SCHEMA.empty_table().to_pandas(), img_path, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY
    )
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        [
            "build",
            "coralnet",
            "build-manifest",
            "--annotations",
            str(ann_path),
            "--images",
            str(img_path),
            *_common(tmp_path),
        ]
    )

    assert rc == 0
    manifest = pd.read_parquet(tmp_path / "v1" / "coralnet_training_manifest_v1.parquet")
    assert manifest.empty
    assert list(manifest.columns) == _MANIFEST_COLUMNS


def test_all_rejects_empty_resized_prefix_before_any_stage(monkeypatch, fake_s3, tmp_path):
    monkeypatch.setenv("MERMAID_CORALNET_RESIZED_PREFIX", "/")
    _patch_to_fake_s3(monkeypatch, fake_s3)
    calls: list[str] = []
    monkeypatch.setattr(build, "build_images", lambda *a, **k: calls.append("build_images"))

    with pytest.raises(ValueError, match="MERMAID_CORALNET_RESIZED_PREFIX"):
        cli.main(["build", "coralnet", "all", "--source-ids", "1", *_common(tmp_path)])

    assert calls == []
    assert not (tmp_path / "v1" / "coralnet_annotations_v1.parquet").exists()


def test_cli_all_uses_custom_resized_prefix(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path
):
    monkeypatch.setenv("MERMAID_CORALNET_RESIZED_PREFIX", "/custom/pre/")
    populate(
        fake_s3,
        _BUCKET,
        _PREFIX,
        1,
        sample_csv_factory(source_id=1, n_images=1, n_points_per_image=1),
    )
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10000.jpg", _real_jpeg(4096, 3000))
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(["build", "coralnet", "all", *_common(tmp_path)])

    assert rc == 0
    assert (_BUCKET, "custom/pre/resized/s1/images/10000.jpg") in fake_s3.objects
    manifest = pd.read_parquet(tmp_path / "v1" / "coralnet_training_manifest_v1.parquet")
    assert manifest["image_s3_key"].tolist() == ["custom/pre/resized/s1/images/10000.jpg"]


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
    [("auto", []), ("v1", ["--version-tag", "v1"])],
    ids=["default-tag", "explicit-tag"],
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
    assert errors == [
        "3 source(s) excluded",
        "image_list_missing: [2]",
        "labelset_missing: [3, 4]",
    ]


def test_cli_all_with_an_excluded_source_writes_every_output(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path
):
    populate(
        fake_s3,
        _BUCKET,
        _PREFIX,
        1,
        sample_csv_factory(source_id=1, n_images=1, n_points_per_image=1),
    )
    broken = sample_csv_factory(source_id=2, n_images=1, n_points_per_image=1)
    broken.pop("labelset.csv")
    populate(fake_s3, _BUCKET, _PREFIX, 2, broken)
    fake_s3.put(_BUCKET, f"{_PREFIX}/s1/images/10000.jpg", _real_jpeg(800, 600))
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(["build", "coralnet", "all", *_common(tmp_path)])

    assert rc == 1
    out_dir = tmp_path / "v1"
    for name in (
        "coralnet_annotations_v1.parquet",
        "coralnet_images_v1.parquet",
        "coralnet_training_manifest_v1.parquet",
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
    manifest = pd.read_parquet(out_dir / "coralnet_training_manifest_v1.parquet")
    assert manifest["source_id"].unique().tolist() == [1]


class _HeadObjectRaises:
    def __init__(self, code: str):
        self._code = code

    def head_object(self, *, Bucket, Key):
        raise ClientError({"Error": {"Code": self._code, "Message": self._code}}, "HeadObject")


@pytest.mark.parametrize("code", ["NoSuchKey", "NotFound"])
def test_s3_object_exists_is_false_for_a_missing_key_code(code):
    assert build._s3_object_exists(_HeadObjectRaises(code), "b", "k") is False


@pytest.mark.parametrize("code", ["403", "500"])
def test_s3_object_exists_reraises_other_errors(code):
    with pytest.raises(ClientError) as exc:
        build._s3_object_exists(_HeadObjectRaises(code), "b", "k")

    assert exc.value.response["Error"]["Code"] == code
