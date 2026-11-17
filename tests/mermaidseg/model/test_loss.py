"""Classification and concept-bottleneck losses."""

from __future__ import annotations

import math

import pytest
import torch

from mermaidseg.model.concept_metrics import calculate_multi_hot_concept_loss
from mermaidseg.model.loss import (
    ConceptBottleneckLoss,
    _masked_cross_entropy,
    binary_concept_losses,
)

NAMES = ["bleached", "anthropogenic", "sand"]
WEIGHTS = {"per_pixel_loss_weight": 0.5, "per_image_loss_weight": 0.5}
LN2 = math.log(2.0)


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, len(NAMES), 4, 4, generator=generator)
    labels = torch.randint(0, 3, (3, len(NAMES), 4, 4), generator=generator)
    labels[0, 0] = 0  # image 0 has no bleached labels
    labels[:, 2] = 0  # no sand labels anywhere
    return logits, labels


def test_concept_bottleneck_loss_requires_lambda_weight():
    with pytest.raises(TypeError, match="lambda_weight"):
        ConceptBottleneckLoss(
            {}, ignore_index=0, per_pixel_loss_weight=0.5, per_image_loss_weight=0.5
        )


def test_each_channel_matches_the_multi_hot_loss_on_that_channel():
    logits, labels = _batch()
    losses = binary_concept_losses(logits, labels, NAMES, **WEIGHTS)
    assert set(losses) == {"multi_hot/bleached", "multi_hot/anthropogenic"}
    for channel, name in enumerate(NAMES[:2]):
        expected = calculate_multi_hot_concept_loss(
            logits[:, channel : channel + 1],
            labels[:, channel : channel + 1],
            **WEIGHTS,
        )
        assert losses[f"multi_hot/{name}"] == pytest.approx(expected.item(), rel=1e-5)


def test_name_count_must_match_the_channels():
    logits, labels = _batch()
    with pytest.raises(ValueError, match="2 binary concept names for 3 channels"):
        binary_concept_losses(logits, labels, NAMES[:2], **WEIGHTS)


def test_concept_bottleneck_loss_reports_the_breakdown_only_when_named():
    logits, labels = _batch()
    outputs = torch.randn(3, 5, 4, 4)
    targets = torch.randint(0, 5, (3, 4, 4))
    plain = ConceptBottleneckLoss({}, ignore_index=0, lambda_weight=1.0, **WEIGHTS)
    named = ConceptBottleneckLoss(
        {}, ignore_index=0, lambda_weight=1.0, binary_concept_names=NAMES, **WEIGHTS
    )
    total_plain, plain_components = plain(outputs, targets, logits, labels)
    total_named, named_components = named(outputs, targets, logits, labels)

    assert not any(key.startswith("multi_hot/") for key in plain_components)
    assert {"multi_hot/bleached", "multi_hot/anthropogenic"} <= named_components.keys()
    assert "multi_hot/sand" not in named_components
    assert total_named.item() == pytest.approx(total_plain.item())
    assert float(named_components["multi_hot"]) == pytest.approx(
        float(plain_components["multi_hot"])
    )


def _cross_entropy_case() -> tuple[torch.Tensor, torch.Tensor]:
    """Two classes, class 0 ignored; a pixel scores ln 2 (logits 0, 0) or ln 4 (ln 3, 0).

    Image 0: ln 2 and ln 4. Image 1: ln 4 and an ignored pixel. Image 2: all ignored.
    Per-pixel mean (5/3) ln 2, per-image mean (1.5 + 2) / 2 ln 2 = 1.75 ln 2.
    """
    outputs = torch.zeros(3, 2, 1, 2)
    outputs[0, 0, 0, 1] = math.log(3.0)
    outputs[1, 0, 0, 0] = math.log(3.0)
    targets = torch.tensor([[[1, 1]], [[1, 0]], [[0, 0]]])
    return outputs, targets


def test_masked_cross_entropy_of_a_partial_mask_blends_the_masked_means():
    outputs, targets = _cross_entropy_case()
    loss = _masked_cross_entropy(
        outputs, targets, weight=None, ignore_index=0, label_smoothing=0.0, **WEIGHTS
    )
    expected = 0.5 * (5 / 3) * LN2 + 0.5 * 1.75 * LN2
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_masked_cross_entropy_of_an_empty_mask_is_zero_with_finite_gradients():
    outputs = torch.randn(2, 3, 2, 2, requires_grad=True)
    targets = torch.zeros(2, 2, 2, dtype=torch.long)
    loss = _masked_cross_entropy(
        outputs, targets, weight=None, ignore_index=0, label_smoothing=0.0, **WEIGHTS
    )
    loss.backward()
    assert loss.item() == 0.0
    assert outputs.grad is not None
    assert torch.equal(outputs.grad, torch.zeros_like(outputs))
