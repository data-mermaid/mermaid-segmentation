"""Tests for taxonomic vs multi-hot concept accuracy and masking."""

from __future__ import annotations

import torch
from torchmetrics.classification import Accuracy

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.model.concept_metrics import (
    calculate_taxonomic_rank_loss,
    map_taxonomy_predictions_to_dense,
    map_taxonomy_to_dense,
    taxonomic_valid_mask,
)
from mermaidseg.model.eval import Evaluator


def _phylum_labels_and_preds(
    *,
    active_idx: int,
    peak_prob: float,
    num_classes: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = [1] * num_classes
    enc[active_idx] = 2
    labels = torch.tensor(enc, dtype=torch.float32).view(1, num_classes, 1, 1)

    preds = torch.full((1, num_classes, 1, 1), (1.0 - peak_prob) / (num_classes - 1))
    preds[0, active_idx, 0, 0] = peak_prob
    return labels, preds


def test_taxonomic_accuracy_matches_below_half_threshold() -> None:
    """Regression: argmax prediction must count even when max softmax prob is below 0.5."""
    labels, preds = _phylum_labels_and_preds(active_idx=6, peak_prob=0.35)
    dense_labels = map_taxonomy_to_dense(labels)
    dense_preds = map_taxonomy_predictions_to_dense(preds)

    assert dense_labels.item() == 8
    assert dense_preds.item() == 8

    acc = Accuracy(task="multiclass", num_classes=18, ignore_index=0)
    acc.update(dense_preds, dense_labels)
    assert acc.compute().item() == 1.0


def test_taxonomic_valid_mask_excludes_not_given() -> None:
    not_given = torch.zeros(1, 4, 2, 2)
    concrete = torch.tensor([2, 1, 1, 1], dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 2, 2)

    assert not taxonomic_valid_mask(not_given).any()
    assert taxonomic_valid_mask(concrete).all()


def test_taxonomic_nll_zero_when_all_masked() -> None:
    not_given = torch.zeros(1, 4, 2, 2)
    probs = torch.softmax(torch.randn(1, 4, 2, 2), dim=1)
    loss = calculate_taxonomic_rank_loss(probs, not_given)
    assert loss.item() == 0.0


def test_taxonomic_nll_works_with_float_labels() -> None:
    """Regression: lookup tensors are float; loss must not use float gather indices."""
    labels, probs = _phylum_labels_and_preds(active_idx=2, peak_prob=0.6, num_classes=8)
    labels = labels.float()
    loss = calculate_taxonomic_rank_loss(probs, labels)
    assert loss.item() > 0.0


def test_taxonomic_nll_decreases_with_correct_class_prob() -> None:
    labels, _ = _phylum_labels_and_preds(active_idx=3, peak_prob=0.9, num_classes=8)
    active_idx = 3

    weak = torch.full((1, 8, 1, 1), 0.05)
    weak[0, active_idx, 0, 0] = 0.35
    weak = weak / weak.sum(dim=1, keepdim=True)

    strong = torch.full((1, 8, 1, 1), 0.01)
    strong[0, active_idx, 0, 0] = 0.9
    strong = strong / strong.sum(dim=1, keepdim=True)

    weak_loss = calculate_taxonomic_rank_loss(weak, labels)
    strong_loss = calculate_taxonomic_rank_loss(strong, labels)

    assert strong_loss.item() < weak_loss.item()


def test_evaluator_multi_hot_uses_threshold_not_argmax() -> None:
    concept_value2id = {rank: {"a": [2, 1]} for rank in TAXONOMIC_CONCEPTS}
    concept_value2id["branching"] = {"true": 1, "false": 0}

    evaluator = Evaluator(
        num_classes=5,
        device="cpu",
        calculate_concept_metrics=True,
        concept_value2id=concept_value2id,
        include_classification=False,
    )

    num_tax = sum(len(next(iter(concept_value2id[r].values()))) for r in TAXONOMIC_CONCEPTS)
    num_concepts = num_tax + 1

    labels = torch.zeros(1, num_concepts, 2, 2)
    labels[:, num_tax:, :, :] = 2  # multi-hot true

    outputs = torch.zeros(1, num_concepts, 2, 2)
    outputs[:, num_tax:, :, :] = 0.8  # sigmoid-like activation > 0.5

    evaluator.evaluate_concepts(outputs, labels)
    results = evaluator.compute_concept_metric_results()
    assert results["accuracy/multi_hot"] == 1.0
