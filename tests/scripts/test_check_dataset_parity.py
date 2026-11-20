from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.check_dataset_parity as script
from mermaidseg.data.parity import ParityReport

_ARGS = [
    "--dataset",
    "catlin_seaview",
    "--split",
    "train",
    "--version",
    "v1",
    "--manifest-sha256",
    "abcd",
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
    monkeypatch.setattr(script.boto3, "client", lambda name: ("client", name))

    result = script.main(
        ["--data-config", str(config), "--cache-dir", str(tmp_path / "c"), *_ARGS]
        + ["--annotator", "a1", "--annotator", "a2"]
    )

    assert result == code
    assert seen == {
        "data_cfg": {"catlin_seaview": {"train": {}}},
        "dataset": "catlin_seaview",
        "split": "train",
        "version": "v1",
        "manifest_sha256": "abcd",
        "cache_dir": tmp_path / "c",
        "client": ("client", "s3"),
        "annotators": ["a1", "a2"],
    }
    assert capsys.readouterr().out.rstrip().endswith("PARITY OK" if code == 0 else "PARITY FAILED")
