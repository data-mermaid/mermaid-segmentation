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
    iter_concept_slices,
)

NAMES = ["bleached", "anthropogenic", "sand"]
WEIGHTS = {"per_pixel_loss_weight": 0.5, "per_image_loss_weight": 0.5}
LN2 = math.log(2.0)
LN3 = math.log(3.0)
LN4 = math.log(4.0)


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, len(NAMES), 4, 4, generator=generator)
    labels = torch.randint(0, 3, (3, len(NAMES), 4, 4), generator=generator)
    labels[0, 0] = 0  # image 0 has no bleached labels
    labels[:, 2] = 0  # no sand labels anywhere
    return logits, labels


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


LN4_3 = math.log(4.0 / 3.0)
# Concept layout: kingdom (2 channels), class (2 channels), then one binary channel.
RANKS = {
    "kingdom": {"not_given": [0, 0], "a": [2, 1], "b": [1, 2]},
    "class": {"not_given": [0, 0], "a": [2, 1], "b": [1, 2]},
}
CONCEPT_VALUE2ID = {**RANKS, "bleached": {"not_given": 0, "FALSE": 1, "TRUE": 2}}


def test_iter_concept_slices_offsets_skip_absent_ranks_and_end_with_the_binary_tail():
    value2id = {
        "kingdom": {"not_given": [0, 0], "a": [2, 1], "b": [1, 2]},
        "class": {"not_given": [0, 0, 0], "a": [2, 1, 1], "b": [1, 2, 1], "c": [1, 1, 2]},
        "bleached": {"not_given": 0, "FALSE": 1, "TRUE": 2},
    }
    # Channel c holds the constant c, so a slice shows which channels it covers.
    labels = torch.arange(7.0).reshape(1, 7, 1, 1)
    outputs = labels + 100

    slices = list(iter_concept_slices(value2id, labels, outputs))

    assert [name for name, _, _ in slices] == ["kingdom", "class", "multi_hot"]
    assert [labels_slice.flatten().tolist() for _, labels_slice, _ in slices] == [
        [0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0],
    ]
    assert [outputs_slice.flatten().tolist() for _, _, outputs_slice in slices] == [
        [100.0, 101.0],
        [102.0, 103.0, 104.0],
        [105.0, 106.0],
    ]


def _one_pixel_case() -> dict[str, torch.Tensor]:
    """One valid pixel: class logits 0 (ln 2), kingdom/class logits (ln 3, 0) on the active channel.

    The binary channel scores label 1 (target 0.01) at logit ln 3: ln 4 - 0.01 ln 3.
    """
    taxonomic = torch.tensor([LN3, 0.0]).reshape(1, 2, 1, 1)
    return {
        "outputs": torch.zeros(1, 2, 1, 1),
        "target_labels": torch.ones(1, 1, 1, dtype=torch.long),
        "concept_logits": torch.cat([taxonomic, taxonomic, torch.tensor([[[[LN3]]]])], dim=1),
        "concept_labels": torch.tensor([[[[2.0]], [[1.0]], [[2.0]], [[1.0]], [[1.0]]]]),
    }


@pytest.mark.parametrize(
    ("smoothing_kwargs", "rank_ce"),
    [
        pytest.param({"taxonomic_label_smoothing": 0.0}, LN4_3, id="no_smoothing"),
        # CE with smoothing e is (1 - e) nll + e * mean_c(-log p_c); p = (3/4, 1/4), target 0.
        pytest.param(
            {"taxonomic_label_smoothing": 0.2},
            0.8 * LN4_3 + 0.2 * (LN4_3 + LN4) / 2,
            id="smoothing_0.2",
        ),
        pytest.param({}, 0.99 * LN4_3 + 0.01 * (LN4_3 + LN4) / 2, id="default_0.01"),
    ],
)
def test_concept_bottleneck_loss_scales_ranks_by_rank_count_and_concepts_by_lambda(
    smoothing_kwargs, rank_ce
):
    binary = LN4 - 0.01 * LN3
    criterion = ConceptBottleneckLoss(
        CONCEPT_VALUE2ID, ignore_index=0, lambda_weight=0.5, **WEIGHTS, **smoothing_kwargs
    )

    total, components = criterion(**_one_pixel_case())

    concepts = 2 * rank_ce / 6 + binary
    assert float(components["classification"]) == pytest.approx(LN2, rel=1e-6)
    assert float(components["kingdom"]) == pytest.approx(rank_ce / 6, rel=1e-6)
    assert float(components["class"]) == pytest.approx(rank_ce / 6, rel=1e-6)
    assert float(components["multi_hot"]) == pytest.approx(binary, rel=1e-6)
    assert float(components["concepts"]) == pytest.approx(concepts, rel=1e-6)
    assert total.item() == pytest.approx(LN2 + 0.5 * concepts, rel=1e-6)


def test_concept_bottleneck_loss_breakdown_names_each_present_rank_and_binary_channel():
    one_pixel = _one_pixel_case()
    one_pixel["concept_logits"] = torch.zeros(1, 5, 1, 1)
    one_pixel["concept_labels"] = torch.tensor([[[[2.0]], [[1.0]], [[2.0]], [[1.0]], [[2.0]]]])
    criterion = ConceptBottleneckLoss(
        CONCEPT_VALUE2ID,
        ignore_index=0,
        lambda_weight=1.0,
        binary_concept_names=["bleached"],
        **WEIGHTS,
    )

    total, components = criterion(**one_pixel)

    # Zero logits score ln 2 everywhere; each rank is divided by the 6 taxonomic ranks.
    assert set(components) == {
        "classification",
        "kingdom",
        "class",
        "multi_hot",
        "multi_hot/bleached",
        "concepts",
    }
    assert float(components["kingdom"]) == pytest.approx(LN2 / 6, rel=1e-6)
    assert float(components["class"]) == pytest.approx(LN2 / 6, rel=1e-6)
    assert components["multi_hot/bleached"] == pytest.approx(LN2, rel=1e-6)
    assert float(components["concepts"]) == pytest.approx(LN2 * 4 / 3, rel=1e-6)
    assert total.item() == pytest.approx(LN2 * 7 / 3, rel=1e-6)
