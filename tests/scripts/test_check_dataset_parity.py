from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.check_dataset_parity as script
from mermaidseg.data.parity import DenseParityReport, ParityReport
from segdata_spec.resize import ResizeSpec
from tests.mermaidseg.data.conftest import ImageSpec, write_version

_VERSION = "20260101-abc1234"
_SHA = "ab" * 32
_ARGS = [
    "--dataset",
    "catlin_seaview",
    "--split",
    "train",
    "--version",
    _VERSION,
    "--manifest-sha256",
    _SHA,
]


@pytest.mark.parametrize(
    ("report", "code"),
    [(ParityReport(), 0), (ParityReport(old_only_images=["a"]), 1)],
    ids=["clean", "diffs"],
)
def test_main_returns_one_on_any_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    report: ParityReport,
    code: int,
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("data:\n  catlin_seaview:\n    train: {}\n")
    seen: dict[str, Any] = {}

    def _run_parity(data_cfg: Any, **kwargs: Any) -> ParityReport:
        seen.update(data_cfg=data_cfg, **kwargs)
        return report

    monkeypatch.setattr(script, "run_parity", _run_parity)
    monkeypatch.setattr(script, "sync_tables", lambda client, pin, cache_dir: None)
    monkeypatch.setattr(script, "version_kind", lambda cache_dir, pin: "points")
    monkeypatch.setattr(script.boto3, "client", lambda name: ("client", name))

    result = script.main(
        ["--data-config", str(config), "--cache-dir", str(tmp_path / "c"), *_ARGS]
        + ["--annotator", "a1", "--annotator", "a2"]
    )

    assert result == code
    assert seen == {
        "sync": False,
        "data_cfg": {"catlin_seaview": {"train": {}}},
        "dataset": "catlin_seaview",
        "split": "train",
        "version": _VERSION,
        "manifest_sha256": _SHA,
        "cache_dir": tmp_path / "c",
        "client": ("client", "s3"),
        "annotators": ["a1", "a2"],
        "resize": None,
    }
    assert capsys.readouterr().out.rstrip().endswith("PARITY OK" if code == 0 else "PARITY FAILED")


@pytest.mark.parametrize(
    ("report", "code"),
    [(DenseParityReport(matched=1), 0), (DenseParityReport(old_only=["a"]), 1)],
    ids=["clean", "diffs"],
)
def test_no_sync_routes_a_dense_version_to_the_dense_path_without_a_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    report: DenseParityReport,
    code: int,
) -> None:
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="dense",
        name="catlin_seaview",
        images=[ImageSpec("i1", mask=np.ones((2, 2), np.uint8))],
        labels={1: "a"},
    )
    config = tmp_path / "data.yaml"
    config.write_text("data:\n  catlin_seaview:\n    train: {}\n")
    seen: dict[str, Any] = {}

    def _run_dense(data_cfg: Any, **kwargs: Any) -> DenseParityReport:
        seen.update(kwargs)
        return report

    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("points path or S3 used")

    monkeypatch.setattr(script, "run_dense_parity", _run_dense)
    monkeypatch.setattr(script, "run_parity", _unexpected)
    monkeypatch.setattr(script.boto3, "client", _unexpected)

    argv = ["--data-config", str(config), "--cache-dir", str(cache_dir), *_ARGS, "--no-sync"]
    argv[argv.index(_SHA)] = pin.manifest_sha256
    result = script.main(argv)

    assert result == code
    assert seen["sync"] is False
    assert seen["client"] is None
    assert capsys.readouterr().out.rstrip().endswith("PARITY OK" if code == 0 else "PARITY FAILED")


def test_no_sync_builds_no_s3_client_for_a_points_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("data:\n  catlin_seaview:\n    train: {}\n")
    seen: dict[str, Any] = {}

    def _run_parity(data_cfg: Any, **kwargs: Any) -> ParityReport:
        seen.update(kwargs)
        return ParityReport()

    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("S3 used")

    monkeypatch.setattr(script, "run_parity", _run_parity)
    monkeypatch.setattr(script, "sync_tables", _unexpected)
    monkeypatch.setattr(script, "version_kind", lambda cache_dir, pin: "points")
    monkeypatch.setattr(script.boto3, "client", _unexpected)

    result = script.main(
        ["--data-config", str(config), "--cache-dir", str(tmp_path / "c"), "--no-sync", *_ARGS]
    )

    assert result == 0
    assert seen["client"] is None
    assert seen["sync"] is False


def test_main_builds_a_resize_spec_from_the_max_edge_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("data: {}\n")
    seen: dict[str, Any] = {}

    def _run_parity(data_cfg: Any, **kwargs: Any) -> ParityReport:
        seen.update(kwargs)
        return ParityReport()

    monkeypatch.setattr(script, "run_parity", _run_parity)
    monkeypatch.setattr(script, "sync_tables", lambda *a, **k: None)
    monkeypatch.setattr(script, "version_kind", lambda cache_dir, pin: "points")
    monkeypatch.setattr(script.boto3, "client", lambda name: None)

    script.main(
        ["--data-config", str(config), "--cache-dir", str(tmp_path), *_ARGS]
        + ["--resize-max-edge", "2048"]
    )

    assert seen["resize"] == ResizeSpec(max_edge=2048)


def test_main_refuses_resize_max_edge_for_a_dense_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("data: {}\n")

    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("comparison ran")

    monkeypatch.setattr(script, "run_dense_parity", _unexpected)
    monkeypatch.setattr(script, "run_parity", _unexpected)
    monkeypatch.setattr(script, "sync_tables", lambda *a, **k: None)
    monkeypatch.setattr(script, "version_kind", lambda cache_dir, pin: "dense")
    monkeypatch.setattr(script.boto3, "client", lambda name: None)

    with pytest.raises(SystemExit) as exit_info:
        script.main(
            ["--data-config", str(config), "--cache-dir", str(tmp_path), *_ARGS]
            + ["--resize-max-edge", "2048"]
        )

    assert exit_info.value.code == 2


def test_hf_head_sha_returns_none_with_a_warning_when_the_hub_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _fail(self: Any, repo_id: str) -> Any:
        raise OSError("offline")

    monkeypatch.setattr(script.HfApi, "dataset_info", _fail)

    assert script.hf_head_sha("org/repo") is None
    assert "org/repo" in caplog.text
