"""Masked taxonomic and multi-hot concept losses."""

from __future__ import annotations

import math

import pytest
import torch

from mermaidseg.model.concept_metrics import (
    calculate_multi_hot_concept_loss,
    calculate_taxonomic_rank_loss,
)

WEIGHTS = {"per_pixel_loss_weight": 0.5, "per_image_loss_weight": 0.5}
LN2 = math.log(2.0)
LN3 = math.log(3.0)
LN4 = math.log(4.0)


def test_taxonomic_rank_loss_of_a_partial_mask_blends_the_masked_means():
    """Two classes, target class 1; a pixel scores ln 2 (logits 0, 0) or ln 4 (ln 3, 0).

    Image 0: ln 2 and ln 4. Image 1: ln 4 and an unlabelled pixel. Image 2: unlabelled.
    Per-pixel mean (5/3) ln 2, per-image mean (1.5 + 2) / 2 ln 2 = 1.75 ln 2.
    """
    logits = torch.zeros(3, 2, 1, 2)
    logits[0, 0, 0, 1] = LN3
    logits[1, 0, 0, 0] = LN3
    labels = torch.zeros(3, 2, 1, 2)
    labels[0, :, 0, :] = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    labels[1, :, 0, 0] = torch.tensor([1.0, 2.0])

    loss = calculate_taxonomic_rank_loss(logits, labels, from_logits=True, **WEIGHTS)

    assert loss.item() == pytest.approx(0.5 * (5 / 3) * LN2 + 0.5 * 1.75 * LN2, rel=1e-6)


def test_multi_hot_loss_of_a_partial_mask_blends_the_masked_means():
    """BCE is ln 2 at logit 0; at logit ln 3 with label 1 (target 0.01) it is ln 4 - 0.01 ln 3.

    Image 0: ln 2 and v. Image 1: v and an unlabelled pixel. Image 2: unlabelled.
    """
    v = LN4 - 0.01 * LN3
    logits = torch.zeros(3, 1, 1, 2)
    logits[0, 0, 0, 1] = LN3
    logits[1, 0, 0, 0] = LN3
    logits[1, 0, 0, 1] = 50.0
    labels = torch.tensor([[[[2.0, 1.0]]], [[[1.0, 0.0]]], [[[0.0, 0.0]]]])

    loss = calculate_multi_hot_concept_loss(logits, labels, **WEIGHTS)

    per_pixel = (LN2 + 2 * v) / 3
    per_image = ((LN2 + v) / 2 + v) / 2
    assert loss.item() == pytest.approx(0.5 * per_pixel + 0.5 * per_image, rel=1e-6)


def test_multi_hot_loss_of_an_empty_mask_is_zero_with_finite_gradients():
    logits = torch.randn(2, 3, 2, 2, requires_grad=True)
    labels = torch.zeros(2, 3, 2, 2)

    loss = calculate_multi_hot_concept_loss(logits, labels, **WEIGHTS)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))
