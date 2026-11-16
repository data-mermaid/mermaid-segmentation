"""Tests for StandardTrainingConfig: the fail-fast guard on standard-mode
training_kwargs."""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from torch import nn

import mermaidseg.model.models as models_mod
from mermaidseg.io import ConfigDict
from mermaidseg.model.training_mode import StandardTrainingConfig


def _valid_kwargs() -> dict:
    return {
        "training_mode": "standard",
        "freeze_encoder": True,
        "mixed_precision": True,
        "mixed_precision_dtype": "bfloat16",
        "max_grad_norm": 1.0,
        "iterations_per_train_epoch": 1000,
        "iterations_per_val_epoch": 200,
        "optimizer": {"type": "AdamW", "lr": 0.001},
        "loss": {"type": "CrossEntropyLoss", "ignore_index": 0},
        "scheduler": {"type": "PolynomialLR", "power": 1, "total_iters": 200},
    }


def test_valid_standard_config_validates():
    StandardTrainingConfig.model_validate(_valid_kwargs())


def test_minimal_config_uses_defaults():
    cfg = StandardTrainingConfig.model_validate(
        {"training_mode": "standard", "optimizer": {"type": "AdamW", "lr": 0.001}}
    )
    assert cfg.freeze_encoder is False
    assert cfg.mixed_precision is False
    assert cfg.max_grad_norm == 1.0
    assert cfg.loss is None
    assert cfg.scheduler is None


def test_wrong_training_mode_literal_raises():
    kwargs = {**_valid_kwargs(), "training_mode": "concept-bottleneck"}
    with pytest.raises(ValidationError):
        StandardTrainingConfig.model_validate(kwargs)


def test_missing_training_mode_raises():
    kwargs = _valid_kwargs()
    del kwargs["training_mode"]
    with pytest.raises(ValidationError):
        StandardTrainingConfig.model_validate(kwargs)


def test_missing_optimizer_raises():
    kwargs = _valid_kwargs()
    del kwargs["optimizer"]
    with pytest.raises(ValidationError):
        StandardTrainingConfig.model_validate(kwargs)


def test_unknown_extra_keys_are_tolerated():
    kwargs = {**_valid_kwargs(), "detach_concepts": False, "class_subset": ["a", "b"]}
    StandardTrainingConfig.model_validate(kwargs)


class _TinyConceptLogits(nn.Module):
    """Concept-mode stub: forward(x) -> obj with ``.logits`` (B, C, H, W); no HF
    download."""

    def __init__(self, num_classes: int = 2, **_kwargs):
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        return types.SimpleNamespace(logits=self.head(x))


def test_non_standard_kwargs_never_reach_the_guard(monkeypatch):
    """MetaModel must only validate through StandardTrainingConfig in standard mode."""
    from mermaidseg.model import meta as meta_module

    monkeypatch.setattr(models_mod, "TinyConceptLogits", _TinyConceptLogits, raising=False)

    with patch.object(
        meta_module.StandardTrainingConfig,
        "model_validate",
        wraps=StandardTrainingConfig.model_validate,
    ) as spy:
        meta_module.MetaModel(
            run_name="test",
            num_classes=2,
            num_concepts=2,
            model_kwargs=ConfigDict({"name": "TinyConceptLogits"}),
            device="cpu",
            training_kwargs=ConfigDict(
                {
                    "training_mode": "concept",
                    "optimizer": {"type": "AdamW", "lr": 0.001},
                }
            ),
        )
        spy.assert_not_called()
