"""sync_data: main loads the data section, plans, syncs with an S3 client and prints the summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.sync_data as script
from mermaidseg.data.config import SyncPlan
from mermaidseg.io import REPO_ROOT
from segdata_spec.manifest import Pin
from segdata_spec.sync import SyncResult

_TAXONOMY_PIN = Pin(name="mermaid-benthic", version="20261121-ab12cd3", manifest_sha256="c" * 64)

_CONFIG = """
data:
  taxonomy:
    version: 20261121-ab12cd3
    manifest_sha256: cccc
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
    plan = SyncPlan(
        pins={},
        taxonomy=_TAXONOMY_PIN,
        selection={},
        split_files={},
    )
    result = SyncResult(datasets={}, taxonomy=plan.taxonomy)

    def _sync_plan(data_cfg: Any, repo_root: Path) -> SyncPlan:
        seen["plan_args"] = (data_cfg, repo_root)
        return plan

    def _run_sync(p: SyncPlan, cache: Path, client: Any, max_workers: int) -> SyncResult:
        seen["run_args"] = (p, cache, client, max_workers)
        return result

    def _client(name: str, config: Any) -> tuple[str, str, int]:
        return ("client", name, config.max_pool_connections)

    monkeypatch.setattr(script, "sync_plan", _sync_plan)
    monkeypatch.setattr(script, "run_sync", _run_sync)
    monkeypatch.setattr(script.boto3, "client", _client)
    monkeypatch.setattr(script, "format_summary", lambda r: f"summary of {r is result}")

    code = script.main(["--data-config", str(config), "--cache-dir", str(cache_dir)])

    assert code == 0
    assert seen["plan_args"] == (
        {
            "taxonomy": {"version": "20261121-ab12cd3", "manifest_sha256": "cccc"},
            "datasets": {
                "coralnet": {
                    "version": "20261120-ab12cd3",
                    "manifest_sha256": "aaaa",
                    "train": {"groups": "all"},
                }
            },
        },
        REPO_ROOT,
    )
    assert seen["run_args"] == (plan, cache_dir, ("client", "s3", 16), 16)
    assert capsys.readouterr().out == "summary of True\n"


@pytest.mark.parametrize(("workers", "pool"), [(4, 10), (32, 32)])
def test_workers_flag_sets_max_workers_and_the_pool_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workers: int, pool: int
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(_CONFIG)
    seen: dict[str, Any] = {}

    def _run_sync(p: SyncPlan, cache: Path, client: Any, max_workers: int) -> SyncResult:
        seen["workers"], seen["client"] = max_workers, client
        return SyncResult(datasets={}, taxonomy=_TAXONOMY_PIN)

    monkeypatch.setattr(script, "run_sync", _run_sync)
    monkeypatch.setattr(script.boto3, "client", lambda name, config: (name, config))

    script.main(
        ["--data-config", str(config), "--cache-dir", str(tmp_path), "--workers", str(workers)]
    )

    assert seen["workers"] == workers
    assert seen["client"][1].max_pool_connections == pool


def test_tables_only_runs_the_tables_path_and_not_the_full_sync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(_CONFIG)
    plan = SyncPlan(pins={}, taxonomy=None, selection={}, split_files={})
    result = SyncResult(datasets={}, taxonomy=None)
    seen: dict[str, Any] = {}

    def _run_table_sync(p: SyncPlan, cache: Path, client: Any) -> SyncResult:
        seen["args"] = (p, cache, client)
        return result

    def _run_sync(*_: Any) -> SyncResult:
        raise AssertionError("full sync must not run")

    monkeypatch.setattr(script, "sync_plan", lambda data_cfg, repo_root: plan)
    monkeypatch.setattr(script, "run_table_sync", _run_table_sync)
    monkeypatch.setattr(script, "run_sync", _run_sync)
    monkeypatch.setattr(script.boto3, "client", lambda name, config: "client")
    monkeypatch.setattr(script, "format_summary", lambda r: f"summary of {r is result}")

    code = script.main(
        ["--data-config", str(config), "--cache-dir", str(tmp_path / "c"), "--tables-only"]
    )

    assert code == 0
    assert seen["args"] == (plan, tmp_path / "c", "client")
    assert capsys.readouterr().out == "summary of True\n"
