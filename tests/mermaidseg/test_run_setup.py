"""Run-setup helpers: checkpoint reads, weight loading and the resolved config."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import pytest
import torch
import yaml

from mermaidseg.io import ConfigDict
from mermaidseg.run_setup import (
    MODEL_CHECKPOINT_KEYS,
    load_model_weights,
    read_checkpoint,
    write_resolved_config,
)


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


def _save_checkpoint(path: Path, state_dict: dict, config: dict | None = None) -> Path:
    torch.save({"config": config or {}, "model_state_dict": state_dict}, path)
    return path


def test_load_model_weights_copies_the_saved_weights(tmp_path):
    source = torch.nn.Linear(2, 2)
    path = _save_checkpoint(tmp_path / "ckpt.pt", source.state_dict(), {"seed": 3})
    target = torch.nn.Linear(2, 2)
    load_model_weights(
        SimpleNamespace(model=target),  # type: ignore[arg-type]
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS),
    )
    torch.testing.assert_close(target.weight, source.weight)


def test_load_model_weights_rejects_a_state_dict_missing_a_model_key(tmp_path):
    state = torch.nn.Linear(2, 2).state_dict()
    del state["bias"]
    checkpoint = read_checkpoint(
        _save_checkpoint(tmp_path / "ckpt.pt", state), ("model_state_dict",)
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_weights(SimpleNamespace(model=torch.nn.Linear(2, 2)), checkpoint)  # type: ignore[arg-type]


def test_load_model_weights_rejects_a_state_dict_with_an_unexpected_key(tmp_path):
    state = {**torch.nn.Linear(2, 2).state_dict(), "extra": torch.zeros(1)}
    checkpoint = read_checkpoint(
        _save_checkpoint(tmp_path / "ckpt.pt", state), ("model_state_dict",)
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_model_weights(SimpleNamespace(model=torch.nn.Linear(2, 2)), checkpoint)  # type: ignore[arg-type]


def test_read_checkpoint_refuses_a_checkpoint_that_pickles_a_compose_naming_the_file(tmp_path):
    path = _save_checkpoint(
        tmp_path / "ckpt.pt",
        torch.nn.Linear(2, 2).state_dict(),
        {"transform": A.Compose([A.HorizontalFlip()])},
    )
    with pytest.raises(RuntimeError, match="ckpt.pt") as excinfo:
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS)
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)


def test_read_checkpoint_names_the_file_and_the_missing_key(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"config": {}}, path)
    with pytest.raises(RuntimeError, match="ckpt.pt.*model_state_dict"):
        read_checkpoint(path, MODEL_CHECKPOINT_KEYS)


def test_read_checkpoint_returns_the_dict_with_tensors_on_the_cpu(tmp_path):
    source = torch.nn.Linear(2, 2)
    path = _save_checkpoint(tmp_path / "ckpt.pt", source.state_dict(), {"seed": 3})
    checkpoint = read_checkpoint(path, MODEL_CHECKPOINT_KEYS)
    assert checkpoint["config"] == {"seed": 3}
    assert checkpoint["model_state_dict"]["weight"].device.type == "cpu"
    torch.testing.assert_close(checkpoint["model_state_dict"]["weight"], source.weight)
