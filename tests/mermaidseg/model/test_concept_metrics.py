"""Masked taxonomic and multi-hot concept losses."""

from __future__ import annotations

import math

import pytest
import torch

from mermaidseg.model.concept_metrics import (
    calculate_multi_hot_concept_loss,
    calculate_taxonomic_rank_loss,
    map_taxonomy_predictions_to_dense,
    map_taxonomy_to_dense,
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

    loss = calculate_taxonomic_rank_loss(logits, labels, **WEIGHTS)

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


def test_taxonomic_rank_loss_rejects_labels_with_a_different_channel_count():
    with pytest.raises(ValueError, match="labels 3 vs outputs 2"):
        calculate_taxonomic_rank_loss(torch.zeros(1, 2, 1, 1), torch.zeros(1, 3, 1, 1))


def test_taxonomic_rank_loss_of_a_rank_with_no_channels_is_a_zero_of_the_output_dtype():
    outputs = torch.zeros(2, 0, 3, 3, dtype=torch.float64)

    loss = calculate_taxonomic_rank_loss(outputs, torch.zeros(2, 0, 3, 3))

    assert loss.dtype == torch.float64
    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_map_taxonomy_to_dense_maps_all_zero_pixels_to_zero_and_others_to_argmax_plus_one():
    # Shape (B=1, C=3, H=1, W=3): pixel 0 is not_given, pixel 1 is channel 0, pixel 2 is channel 2.
    labels = torch.tensor([[[[0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 2.0]]]])

    assert map_taxonomy_to_dense(labels).tolist() == [[[0, 1, 3]]]


def test_map_taxonomy_predictions_to_dense_is_the_argmax_channel_plus_one():
    scores = torch.tensor([[[[0.1, 0.9]], [[0.7, 0.2]], [[0.3, 0.4]]]])

    assert map_taxonomy_predictions_to_dense(scores).tolist() == [[[2, 1]]]
