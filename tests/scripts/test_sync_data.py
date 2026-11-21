"""sync_data: main loads the data section, plans, syncs with an S3 client and prints the summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.sync_data as script
from mermaidseg.data.config import SyncPlan
from mermaidseg.io import REPO_ROOT
from segdata_spec.sync import SyncResult

_CONFIG = """
data:
  datasets:
    coralnet:
      version: 20261120-ab12cd3
      manifest_sha256: aaaa
      train: {groups: all}
"""


def test_main_plans_syncs_and_prints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(_CONFIG)
    cache_dir = tmp_path / "cache"
    seen: dict[str, Any] = {}
    plan = SyncPlan(pins={}, taxonomy=None, selection={}, on_missing_images={}, split_files={})
    result = SyncResult(datasets={}, taxonomy=None)

    def _sync_plan(data_cfg: Any, repo_root: Path) -> SyncPlan:
        seen["plan_args"] = (data_cfg, repo_root)
        return plan

    def _run_sync(p: SyncPlan, cache: Path, client: Any) -> SyncResult:
        seen["run_args"] = (p, cache, client)
        return result

    monkeypatch.setattr(script, "sync_plan", _sync_plan)
    monkeypatch.setattr(script, "run_sync", _run_sync)
    monkeypatch.setattr(script.boto3, "client", lambda name: ("client", name))
    monkeypatch.setattr(script, "format_summary", lambda r: f"summary of {r is result}")

    code = script.main(["--data-config", str(config), "--cache-dir", str(cache_dir)])

    assert code == 0
    assert seen["plan_args"] == (
        {
            "datasets": {
                "coralnet": {
                    "version": "20261120-ab12cd3",
                    "manifest_sha256": "aaaa",
                    "train": {"groups": "all"},
                }
            }
        },
        REPO_ROOT,
    )
    assert seen["run_args"] == (plan, cache_dir, ("client", "s3"))
    assert capsys.readouterr().out == "summary of True\n"


def test_main_requires_both_paths() -> None:
    with pytest.raises(SystemExit):
        script.main(["--data-config", "x.yaml"])
