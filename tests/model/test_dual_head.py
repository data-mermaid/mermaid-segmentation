"""Tests for dual-head morphology wiring (no HuggingFace download)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from mermaidseg.dataset_reconciliation.morphology import (
    growth_form_to_morphology_vector,
    source_labels_to_morphology,
)
from mermaidseg.model.loss import DualTaxonomicalLoss
from mermaidseg.model.models import LinearDualDINOv3


class _FakeConfig:
    hidden_size = 8
    patch_size = 16


class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x, **kwargs):
        b, _, h, w = x.shape
        # 5 prefix + (h/16)*(w/16) patches at hidden=8
        n_tokens = 5 + (h // 16) * (w // 16)
        hidden = torch.randn(b, n_tokens, 8, device=x.device, dtype=x.dtype)
        return SimpleNamespace(last_hidden_state=hidden)

    def parameters(self):
        return self.linear.parameters()


def test_growth_form_to_morphology_vector_sets_active_channel():
    vec = growth_form_to_morphology_vector("plating", ["plating", "branching", "massive"])
    assert vec == [2, 1, 1]
    assert growth_form_to_morphology_vector(None, ["plating"]) == [0]
    assert growth_form_to_morphology_vector("unknown-form", ["plating"]) == [0]


def test_source_labels_to_morphology_gather_shape():
    lookup = torch.tensor(
        [
            [0, 0],
            [2, 1],
            [1, 2],
        ],
        dtype=torch.float32,
    )
    source = torch.tensor([[[1, 2], [0, 1]]])
    morph = source_labels_to_morphology(source, lookup)
    assert morph.shape == (1, 2, 2, 2)
    assert morph[0, 0, 0, 0].item() == 2.0
    assert morph[0, 1, 0, 1].item() == 2.0


def test_linear_dual_forward_shapes():
    fake = _FakeEncoder()
    with patch("mermaidseg.model.models.AutoModel.from_pretrained", return_value=fake):
        model = LinearDualDINOv3(
            num_classes=4,
            input_size=(32, 32),
            use_lora=False,
            morphology_names=["plating", "branching"],
        )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.logits.shape == (2, 4, 32, 32)
    assert out.morphology_logits is not None
    assert out.morphology_logits.shape == (2, 2, 32, 32)


def test_dual_loss_with_partial_morph_mask():
    id2label = {0: "ignore", 1: "Acropora", 2: "Sand"}
    hierarchy = {"acropora": "hard coral", "hard coral": None, "sand": None}
    loss_fn = DualTaxonomicalLoss(id2label, hierarchy, alpha=0.0, beta=0.0, gamma=1.0)
    logits = torch.randn(1, 3, 2, 2)
    targets = torch.ones(1, 2, 2, dtype=torch.long)
    morph_logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    morph_targets = torch.zeros(1, 2, 2, 2)
    morph_targets[0, 0, 0, 0] = 2  # one known plating pixel
    total, components = loss_fn(logits, targets, morph_logits, morph_targets)
    assert components["morphology"] > 0.0
    total.backward()
    assert morph_logits.grad is not None
