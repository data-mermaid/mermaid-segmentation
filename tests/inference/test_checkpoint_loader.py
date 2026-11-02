"""load_checkpoint_model builds a research checkpoint's model offline and strict-loads it."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModel

import mermaidseg.model.models as mm_models
from mermaidseg.inference import derive_concept_value2id, load_checkpoint_model
from tests.inference._tiny import (
    INPUT_SIZE,
    concept_id2name,
    concept_value2id,
    id2label,
)


@pytest.fixture(autouse=True)
def offline_encoder(monkeypatch):
    """Build the encoder from its saved config instead of downloading pretrained weights."""

    def from_pretrained(name, token=None, **kwargs):
        return AutoModel.from_config(AutoConfig.from_pretrained(name))

    monkeypatch.setattr(
        mm_models,
        "AutoModel",
        SimpleNamespace(from_pretrained=from_pretrained, from_config=AutoModel.from_config),
    )


def _image() -> torch.Tensor:
    return torch.randn(1, 3, *INPUT_SIZE, generator=torch.Generator().manual_seed(1))


def test_derive_concept_value2id_matches_the_label_map_layout():
    derived = derive_concept_value2id(concept_id2name())
    assert {r: len(next(iter(v.values()))) for r, v in derived.items()} == {
        r: len(next(iter(v.values()))) for r, v in concept_value2id().items()
    }
    assert derived.keys() == concept_value2id().keys()


def test_loaded_model_reproduces_the_trained_model(trained_run, checkpoint_path):
    model = load_checkpoint_model(
        checkpoint_path, trained_run["model_cfg"], id2label(), concept_id2name()
    )
    assert not model.training
    with torch.no_grad():
        expected = trained_run["model"](_image())
        got = model(_image())
    assert torch.allclose(got.logits, expected.logits, atol=1e-6, rtol=0)


def test_bare_state_dict_checkpoint_loads(trained_run, tmp_path):
    path = tmp_path / "bare.pt"
    torch.save(trained_run["model"].state_dict(), path)
    model = load_checkpoint_model(path, trained_run["model_cfg"], id2label(), concept_id2name())
    assert model.concept_classifier.out_channels == len(id2label())


def test_model_config_is_not_mutated(trained_run, checkpoint_path):
    cfg = dict(trained_run["model_cfg"])
    load_checkpoint_model(checkpoint_path, cfg, id2label(), concept_id2name())
    assert cfg == trained_run["model_cfg"]


def test_class_count_mismatch_is_refused(trained_run, checkpoint_path):
    labels = {**id2label(), 5: "extra"}
    with pytest.raises(ValueError, match="classes"):
        load_checkpoint_model(checkpoint_path, trained_run["model_cfg"], labels, concept_id2name())


def test_concept_count_mismatch_is_refused(trained_run, checkpoint_path):
    names = {k: v for k, v in concept_id2name().items() if k != 1}
    with pytest.raises(ValueError, match="concept"):
        load_checkpoint_model(checkpoint_path, trained_run["model_cfg"], id2label(), names)


def test_missing_weight_fails_the_strict_load(trained_run, tmp_path):
    state = dict(trained_run["model"].state_dict())
    state.pop(next(k for k in state if k.startswith("dpt_head.")))
    path = tmp_path / "missing.pt"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint_model(path, trained_run["model_cfg"], id2label(), concept_id2name())


def test_batchnorm_head_is_refused(trained_run, tmp_path):
    state = dict(trained_run["model"].state_dict())
    state["dpt_head.neck.bn.running_mean"] = torch.zeros(3)
    path = tmp_path / "bn.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="BatchNorm"):
        load_checkpoint_model(path, trained_run["model_cfg"], id2label(), concept_id2name())


def test_unsupported_model_name_is_refused(trained_run, checkpoint_path):
    cfg = {**trained_run["model_cfg"], "name": "SomethingElse"}
    with pytest.raises(ValueError, match="Unsupported model"):
        load_checkpoint_model(checkpoint_path, cfg, id2label(), concept_id2name())


def test_loader_imports_without_the_research_stack():
    script = (
        "import sys\n"
        "for n in ('albumentations', 'pandas'):\n"
        "    sys.modules[n] = None\n"
        "from mermaidseg.inference import load_checkpoint_model\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
