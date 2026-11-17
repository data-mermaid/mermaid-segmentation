"""CBMPredictor: the forward pass autocasts in the configured dtype, or not at all."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mermaidseg.evaluation.predictor import CBMPredictor


class _StubModel(nn.Module):
    """CBM-shaped stub that records the CPU autocast state during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.concept_classifier = nn.Conv2d(3, 2, 1)
        self.seen: dict[str, object] = {}

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        self.seen["enabled"] = torch.is_autocast_enabled("cpu")
        self.seen["dtype"] = torch.get_autocast_dtype("cpu")
        return SimpleNamespace(logits=self.concept_classifier(x), hidden_states=x)


def test_amp_dtype_runs_forward_under_cpu_autocast():
    model = _StubModel()
    predictor = CBMPredictor(model, (8, 8), device="cpu", amp_dtype=torch.bfloat16)
    predictor.forward_probs(torch.zeros(1, 3, 8, 8))
    assert model.seen == {"enabled": True, "dtype": torch.bfloat16}


def test_no_amp_dtype_runs_forward_without_autocast():
    model = _StubModel()
    predictor = CBMPredictor(model, (8, 8), device="cpu")
    predictor.forward_probs(torch.zeros(1, 3, 8, 8))
    assert model.seen["enabled"] is False


def test_device_is_required():
    with pytest.raises(TypeError, match="device"):
        CBMPredictor(_StubModel(), (8, 8))  # type: ignore[call-arg]
