"""Tests for ``segdata promote``: a staged version uploaded write-once, images then tables then manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from segdata_etl import cli, promote
from segdata_etl.common.s3_objects import ObjectConflict
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ChecksumMismatch, PublishRefused

from .conftest import FakeS3, client_error, crc64nvme_b64, jpeg_bytes, png_bytes
from .test_coralscapes_v2 import IMG_A, IMG_B, MASK, MASK_B
from .test_coralscapes_v2 import build as build_coralscapes
from .test_mermaid import (  # noqa: F401
    ENCRUSTING,
    RUBBLE,
    ann,
    only_version,
    repo,
    seed,
)
from .test_mermaid import build as build_mermaid

BLOBS = {"i1": png_bytes(8, 6), "i2": png_bytes(12, 9), "i3": jpeg_bytes(5, 5)}
_TABLES = ("images.parquet", "points.parquet", "labels.parquet", "stats/label_counts.parquet")


def stage_mermaid(tmp_path: Path) -> tuple[FakeS3, Path, str]:
    """Stage a mermaid version; the returned fake holds the source images and nothing else."""
    fake = FakeS3()
    rows = [
        ann("i1", "p1", 10, 20, ENCRUSTING),
        ann("i2", "p2", 1, 2, RUBBLE),
        ann("i3", "p3", 7, 8, RUBBLE),
    ]
    seed(fake, rows, BLOBS)
    stage = tmp_path / "stage"
    assert build_mermaid(fake, stage) == 0
    fake.calls.clear()
    return fake, stage, only_version(stage).name


def version_dir(stage: Path, dataset: str, version: str) -> Path:
    return stage / DATASET_BUCKET / "datasets" / dataset / version


def run_promote(client: Any, stage: Path, version: str, dataset: str = "mermaid") -> None:
    promote.promote(client, stage_dir=stage, dataset=dataset, version=version, workers=2)


def writes(fake: FakeS3) -> list[tuple[str, str, str, dict[str, Any]]]:
    return [c for c in fake.calls if c[0] in ("put", "copy")]


def test_images_precede_tables_and_the_manifest_is_the_last_put(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)

    run_promote(fake, stage, version)

    log = writes(fake)
    prefix = f"datasets/mermaid/{version}/"
    kinds = [(op, key.startswith(prefix)) for op, _, key, _ in log]
    assert kinds == [("copy", False)] * 3 + [("put", True)] * 5
    assert log[-1][2] == prefix + "manifest.json"
    assert {k for _, _, k, _ in log[3:-1]} == {prefix + t for t in _TABLES}
    assert all(kwargs["IfNoneMatch"] == "*" for _, _, _, kwargs in log)
    assert all(b == DATASET_BUCKET for _, b, _, _ in log)
    published = fake.objects[(DATASET_BUCKET, prefix + "manifest.json")]
    assert published == (version_dir(stage, "mermaid", version) / "manifest.json").read_bytes()
    assert (DATASET_BUCKET, prefix + "copies.jsonl") not in fake.objects
    for image_id, data in BLOBS.items():
        (key,) = [k for b, k in fake.objects if k.startswith(f"images/mermaid/{image_id}.")]
        assert fake.objects[(DATASET_BUCKET, key)] == data


def test_copy_landing_with_another_crc_raises_before_any_table_is_put(repo, tmp_path):  # noqa: F811
    class WrongCrc(FakeS3):
        def copy_object(self, **kwargs: Any) -> dict[str, Any]:
            response = super().copy_object(**kwargs)
            response["CopyObjectResult"]["ChecksumCRC64NVME"] = crc64nvme_b64(b"other")
            return response

    fake, stage, version = stage_mermaid(tmp_path)
    bad = WrongCrc(**{k: getattr(fake, k) for k in ("objects", "crcs", "etags")})

    with pytest.raises(ObjectConflict):
        run_promote(bad, stage, version)

    prefix = f"datasets/mermaid/{version}/"
    assert not [k for _, k in bad.objects if k.startswith(prefix)]
    assert bad.ops("put") == []


def test_published_manifest_refuses_before_any_write(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    key = f"datasets/mermaid/{version}/manifest.json"
    fake.put(DATASET_BUCKET, key, b"{}")
    fake.calls.clear()

    with pytest.raises(PublishRefused, match="already"):
        run_promote(fake, stage, version)

    assert [(op, k) for op, _, k, _ in fake.calls] == [("head", key)]


def test_staged_table_that_no_longer_matches_the_manifest_refuses_with_no_writes(
    repo,  # noqa: F811
    tmp_path,
):
    fake, stage, version = stage_mermaid(tmp_path)
    table = version_dir(stage, "mermaid", version) / "points.parquet"
    table.write_bytes(table.read_bytes() + b"x")

    with pytest.raises(ChecksumMismatch, match="points.parquet"):
        run_promote(fake, stage, version)

    assert fake.calls == []


def rewrite_plan(stage: Path, version: str, edit) -> None:
    path = version_dir(stage, "mermaid", version) / "copies.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(line) + "\n" for line in edit(lines)))


@pytest.mark.parametrize("field", ["crc64nvme", "etag"])
@pytest.mark.parametrize("bad", [None, ""])
def test_plan_line_without_crc_or_etag_refuses_at_preflight(repo, tmp_path, field, bad):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)

    def edit(lines):
        lines[0][field] = bad
        return lines

    rewrite_plan(stage, version, edit)

    with pytest.raises(PublishRefused, match=field):
        run_promote(fake, stage, version)

    assert fake.calls == []


def test_plan_missing_a_line_refuses_at_preflight(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    rewrite_plan(stage, version, lambda lines: lines[1:])

    with pytest.raises(PublishRefused, match="keys"):
        run_promote(fake, stage, version)

    assert fake.calls == []


def test_plan_line_disagreeing_with_images_parquet_refuses_at_preflight(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)

    def edit(lines):
        lines[0]["size"] += 1
        return lines

    rewrite_plan(stage, version, edit)

    with pytest.raises(PublishRefused, match="images.parquet"):
        run_promote(fake, stage, version)

    assert fake.calls == []


def test_both_or_neither_plan_file_refuses(repo, tmp_path):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    directory = version_dir(stage, "mermaid", version)
    (directory / "uploads.jsonl").write_text("")
    with pytest.raises(PublishRefused, match="exactly one"):
        run_promote(fake, stage, version)
    (directory / "uploads.jsonl").unlink()
    (directory / "copies.jsonl").unlink()
    with pytest.raises(PublishRefused, match="exactly one"):
        run_promote(fake, stage, version)
    assert fake.calls == []


def test_rerun_after_a_crash_finds_the_images_present_and_writes_the_manifest(repo, tmp_path):  # noqa: F811
    class FailsFirstTable(FakeS3):
        failed = False

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["Key"].endswith(".parquet") and not self.failed:
                self.failed = True
                raise client_error("InternalError", 500, "PutObject")
            return super().put_object(**kwargs)

    fake, stage, version = stage_mermaid(tmp_path)
    flaky = FailsFirstTable(**{k: getattr(fake, k) for k in ("objects", "crcs", "etags")})

    with pytest.raises(Exception, match="InternalError"):
        run_promote(flaky, stage, version)
    manifest_key = (DATASET_BUCKET, f"datasets/mermaid/{version}/manifest.json")
    assert manifest_key not in flaky.objects
    flaky.calls.clear()

    run_promote(flaky, stage, version)

    assert manifest_key in flaky.objects
    assert len(flaky.ops("copy")) == 3
    assert flaky.ops("put")[-1][2] == manifest_key[1]


def test_rerun_logs_images_as_already_present(repo, tmp_path, caplog):  # noqa: F811
    fake, stage, version = stage_mermaid(tmp_path)
    for plan in (version_dir(stage, "mermaid", version) / "copies.jsonl").read_text().splitlines():
        line = json.loads(plan)
        fake.put(
            DATASET_BUCKET, line["dst_key"], fake.objects[(line["src_bucket"], line["src_key"])]
        )

    with caplog.at_level("INFO"):
        run_promote(fake, stage, version)

    assert "3 already present" in caplog.text
    assert "0 copied" in caplog.text


def test_hf_uploads_land_byte_identical_with_the_planned_crc(monkeypatch, tmp_path):
    version_path = build_coralscapes(monkeypatch, tmp_path)
    stage = tmp_path / "stage"
    fake = FakeS3()

    run_promote(fake, stage, version_path.name, dataset="coralscapes_v2")

    expected = {
        "images/coralscapes_v2/train/a1.jpg": IMG_A,
        "images/coralscapes_v2/train/a1_label.png": MASK,
        "images/coralscapes_v2/validation/b1.png": IMG_B,
        "images/coralscapes_v2/validation/b1_label.png": MASK_B,
    }
    for key, data in expected.items():
        assert fake.objects[(DATASET_BUCKET, key)] == data
        assert fake.crcs[(DATASET_BUCKET, key)] == crc64nvme_b64(data)
    log = writes(fake)
    assert [op for op, *_ in log] == ["put"] * len(log)
    assert log[-1][2].endswith("/manifest.json")
    assert not any(k.endswith(("uploads.jsonl", "copies.jsonl")) for _, k in fake.objects)


def test_hf_upload_whose_local_bytes_differ_from_the_plan_refuses_at_preflight(
    monkeypatch, tmp_path
):
    version_path = build_coralscapes(monkeypatch, tmp_path)
    stage = tmp_path / "stage"
    local = stage / DATASET_BUCKET / "images/coralscapes_v2/train/a1.jpg"
    local.write_bytes(jpeg_bytes(4, 3) + b"\x00")
    fake = FakeS3()

    with pytest.raises(PublishRefused, match="a1.jpg"):
        run_promote(fake, stage, version_path.name, dataset="coralscapes_v2")

    assert fake.calls == []


def test_cli_requires_its_flags_and_dispatches(monkeypatch, tmp_path):
    for argv in (
        ["promote", "--dataset", "d", "--version", "v"],
        ["promote", "--stage-dir", str(tmp_path), "--version", "v"],
        ["promote", "--stage-dir", str(tmp_path), "--dataset", "d"],
    ):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(argv)
    seen: dict[str, Any] = {}

    def fake_promote(client: Any, **kwargs: Any) -> None:
        seen.update(kwargs, client=client)

    client = object()
    monkeypatch.setattr(promote, "promote", fake_promote)
    monkeypatch.setattr(cli, "make_s3_client", lambda: client)

    code = cli.main(
        [
            "promote",
            "--stage-dir",
            str(tmp_path),
            "--dataset",
            "d",
            "--version",
            "v",
            "--workers",
            "3",
        ]
    )

    assert code == 0
    assert seen == {
        "client": client,
        "stage_dir": tmp_path,
        "dataset": "d",
        "version": "v",
        "workers": 3,
    }
