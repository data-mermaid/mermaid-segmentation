"""Repository path resolution and the wandb login check."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import pytest
import torch
import yaml

from mermaidseg.io import ConfigDict
from mermaidseg.run_setup import (
    REPO_ROOT,
    check_wandb_auth,
    load_model_weights,
    resolve_repo_path,
    write_resolved_config,
)


def test_relative_path_resolves_to_the_repo_not_the_current_directory(monkeypatch, tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "class_to_concepts.csv").write_text("decoy\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("configs/class_to_concepts.csv") == (
        REPO_ROOT / "configs" / "class_to_concepts.csv"
    )


def _fake_wandb(monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
    fake = types.ModuleType("wandb")
    fake.api = SimpleNamespace(api_key=api_key)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake)


def test_wandb_auth_prints_no_part_of_the_api_key(monkeypatch, capsys):
    key = "wk7Qz9Lm3Xp2Rv8TyHd4"
    _fake_wandb(monkeypatch, key)
    check_wandb_auth()
    out = capsys.readouterr().out
    assert out
    leaked = [key[i : i + 4] for i in range(len(key) - 3) if key[i : i + 4] in out]
    assert leaked == []


def test_wandb_auth_without_a_key_raises(monkeypatch):
    _fake_wandb(monkeypatch, "")
    with pytest.raises(RuntimeError, match="not authenticated"):
        check_wandb_auth()


def test_resolved_config_yaml_holds_the_overridden_values(tmp_path):
    cfg = ConfigDict(
        {
            "run_name": "r1",
            "training": {"grad_accum_steps": 4, "concept_mapping_path": "/abs/map.csv"},
            "model": {"input_size": (56, 56)},
        }
    )
    write_resolved_config(cfg, tmp_path)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == {
        "run_name": "r1",
        "training": {"grad_accum_steps": 4, "concept_mapping_path": "/abs/map.csv"},
        "model": {"input_size": [56, 56]},
    }


def test_load_model_weights_reads_a_legacy_checkpoint_that_pickles_a_compose(tmp_path):
    source = torch.nn.Linear(2, 2)
    path = tmp_path / "legacy.pt"
    legacy = {
        "config": {"transform": A.Compose([A.HorizontalFlip()])},
        "model_state_dict": source.state_dict(),
    }
    torch.save(legacy, path)
    target = torch.nn.Linear(2, 2)
    checkpoint = load_model_weights(SimpleNamespace(model=target), path, "cpu")  # type: ignore[arg-type]
    torch.testing.assert_close(target.weight, source.weight)
    assert isinstance(checkpoint["config"]["transform"], A.Compose)
