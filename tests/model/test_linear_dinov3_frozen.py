"""Frozen-encoder no_grad gating for LinearDINOv3.

Freezing a probe should not just stop encoder gradients (requires_grad=False) — it
should also run the backbone forward under ``torch.no_grad`` so no autograd graph /
activations are held. These tests use a tiny mock encoder with a real parameter so we
can prove the gate: with the frozen gate on, a gradient must NOT reach the encoder
parameter even if its ``requires_grad`` is left True.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mermaidseg.model import models as models_mod


class _ParamEncoder(torch.nn.Module):
    """Minimal stand-in for the DINOv3 encoder whose output depends on a trainable
    parameter."""

    def __init__(self, hidden_size: int = 64, patch_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, patch_size=patch_size)
        self.scale = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, pixel_values: torch.Tensor, **_):
        b, _c, h, w = pixel_values.shape
        seq = 5 + (h // self.config.patch_size) * (w // self.config.patch_size)  # 5 prefix tokens
        base = torch.ones(
            b, seq, self.config.hidden_size, device=pixel_values.device, dtype=pixel_values.dtype
        )
        return SimpleNamespace(last_hidden_state=base * self.scale)


@pytest.fixture
def model(monkeypatch):
    """LinearDINOv3 backed by the param encoder, sized so a 32x32 input yields a 2x2
    token grid."""
    encoder = _ParamEncoder(hidden_size=64, patch_size=16)
    mock_auto = MagicMock()
    mock_auto.from_pretrained = lambda *_a, **_k: encoder
    monkeypatch.setattr(models_mod, "AutoModel", mock_auto)
    return models_mod.LinearDINOv3(num_classes=3, input_size=(32, 32))


def _forward(model):
    return model(torch.randn(2, 3, 32, 32)).logits


def test_freeze_unfreeze_toggles_flag_and_requires_grad(model):
    assert model._encoder_frozen is False
    model.freeze_encoder()
    assert model._encoder_frozen is True
    assert all(not p.requires_grad for p in model.encoder.parameters())
    model.unfreeze_encoder()
    assert model._encoder_frozen is False
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_frozen_forward_shape(model):
    model.freeze_encoder()
    logits = _forward(model)
    assert logits.shape == (2, 3, 32, 32)


def test_frozen_gate_keeps_backbone_out_of_the_graph(model):
    """With the frozen gate on, no gradient reaches the encoder param even if
    requires_grad=True."""
    model.freeze_encoder()
    # Isolate the no_grad gate from the requires_grad=False freeze: re-enable grad on the param.
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    _forward(model).sum().backward()
    assert model.encoder.scale.grad is None  # no_grad gate skipped the backbone graph
    assert model.head.classifier.weight.grad is not None  # the head still trains


def test_unfrozen_forward_tracks_the_backbone(model):
    """The contrast: unfrozen, the same forward builds the graph through the encoder
    param."""
    assert model._encoder_frozen is False
    _forward(model).sum().backward()
    assert model.encoder.scale.grad is not None
    assert model.head.classifier.weight.grad is not None
