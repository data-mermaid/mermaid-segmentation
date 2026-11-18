"""Artifact and checkpoint loading for evaluation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import albumentations as A
import pytest
import torch
import yaml

from mermaidseg.inference import loading
from mermaidseg.inference.loading import DemoArtifacts, build_model, load_artifacts
from mermaidseg.run_setup import resolve_repo_path


def _write_layouts(tmp_path: Path) -> tuple[Path, Path]:
    id2label = tmp_path / "id2label.json"
    id2label.write_text(json.dumps({"0": "ignore", "1": "sand"}))
    concepts = tmp_path / "concept_id2name.json"
    concepts.write_text(json.dumps({"0": "genus:Acropora"}))
    return id2label, concepts


def test_model_yaml_without_the_model_wrapper_raises_naming_the_file(tmp_path):
    section = yaml.safe_load(
        resolve_repo_path("configs/model_config_cbm_dpt_lora_vitl.yaml").read_text()
    )["model"]
    model_yaml = tmp_path / "unwrapped_model.yaml"
    model_yaml.write_text(yaml.safe_dump(section))
    id2label, concepts = _write_layouts(tmp_path)
    with pytest.raises(ValueError, match="unwrapped_model.yaml"):
        load_artifacts("ckpt.pt", model_yaml, id2label, concepts)


def _model_section() -> dict:
    return yaml.safe_load(
        resolve_repo_path("configs/model_config_cbm_dpt_lora_vitl.yaml").read_text()
    )["model"]


@pytest.mark.parametrize(
    "document",
    [
        {"model": {**_model_section(), "unknown_key": 1}},
        {"model": {k: v for k, v in _model_section().items() if k != "lora_r"}},
        {"model": _model_section(), "training": {}},
    ],
    ids=["unknown-key", "missing-key", "extra-top-level-key"],
)
def test_wrapped_model_yaml_that_fails_the_schema_raises_naming_the_file(tmp_path, document):
    model_yaml = tmp_path / "bad_model.yaml"
    model_yaml.write_text(yaml.safe_dump(document))
    id2label, concepts = _write_layouts(tmp_path)
    with pytest.raises(ValueError, match="bad_model.yaml"):
        load_artifacts("ckpt.pt", model_yaml, id2label, concepts)


def test_build_model_refuses_a_checkpoint_that_pickles_an_albumentations_compose(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"config": {"transform": A.Compose([A.HorizontalFlip()])}, "model_state_dict": {}},
        checkpoint,
    )
    artifacts = DemoArtifacts(
        model_cfg={},
        id2label={0: "ignore", 1: "sand"},
        concept_id2name={0: "genus:Acropora"},
        checkpoint_path=str(checkpoint),
    )
    with pytest.raises(pickle.UnpicklingError):
        build_model(artifacts, "cpu")


def test_build_model_rejects_a_state_dict_missing_a_model_key(tmp_path, monkeypatch):
    class _Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    monkeypatch.setitem(loading.MODEL_CLASSES, "stub", _Model)
    monkeypatch.setattr(loading, "load_pretrained_encoder", lambda _name: object())
    state = _Model().state_dict()
    del state["linear.bias"]
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": state}, checkpoint)
    artifacts = DemoArtifacts(
        model_cfg={"name": "stub", "encoder_name": "stub/encoder"},
        id2label={0: "ignore", 1: "sand"},
        concept_id2name={0: "genus:Acropora"},
        checkpoint_path=str(checkpoint),
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        build_model(artifacts, "cpu")
