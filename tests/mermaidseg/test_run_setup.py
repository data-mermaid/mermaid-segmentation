"""Repository path resolution and the wandb login check."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from mermaidseg.run_setup import REPO_ROOT, check_wandb_auth, resolve_repo_path


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
