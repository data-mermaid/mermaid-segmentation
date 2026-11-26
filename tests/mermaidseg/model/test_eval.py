"""Concept accuracy metrics accumulated by the Evaluator."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mermaidseg.model.eval import Evaluator

# Channels: kingdom (2), then the binary concepts bleached and sand. Taxonomic values are
# per-level id lists; binary values are ints keyed by bool, which the annotation does not model.
CONCEPT_VALUE2ID: dict[str, dict[Any, Any]] = {
    "kingdom": {"not_given": [0, 0], "a": [2, 1]},
    "bleached": {"not_given": 0, False: 1, True: 2},
    "sand": {"not_given": 0, False: 1, True: 2},
}


def _evaluator() -> Evaluator:
    return Evaluator(num_classes=3, device="cpu", concept_value2id=CONCEPT_VALUE2ID)


def _batch(
    kingdom_labels: list[list[float]],
    kingdom_scores: list[list[float]],
    bleached: tuple[list[float], list[float]],
    sand: tuple[list[float], list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concept outputs and labels of shape (1, 4, 1, W): two kingdom channels, then two binary."""
    labels = torch.tensor([*kingdom_labels, bleached[0], sand[0]]).reshape(1, 4, 1, -1)
    outputs = torch.tensor([*kingdom_scores, bleached[1], sand[1]]).reshape(1, 4, 1, -1)
    return outputs, labels


def _mixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Four pixels; kingdom labels 1, 2, not_given, 2 against predictions 1, 1, -, 2.

    Bleached labels 2, 1, 0, 1 against thresholded predictions 2, 2, -, 2; sand labels 1, 2, 0, 2
    against 1, 2, -, 1. Ignored pixels (label 0) score nothing.
    """
    return _batch(
        kingdom_labels=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        kingdom_scores=[[0.9, 0.9, 0.5, 0.2], [0.1, 0.1, 0.5, 0.8]],
        bleached=([2.0, 1.0, 0.0, 1.0], [0.9, 0.8, 0.1, 0.9]),
        sand=([1.0, 2.0, 0.0, 2.0], [0.1, 0.9, 0.1, 0.1]),
    )


def test_concept_accuracy_scores_each_rank_and_averages_the_binary_concepts():
    evaluator = _evaluator()
    outputs, labels = _mixed_batch()

    evaluator.evaluate_concepts(outputs, labels)
    results = evaluator.compute_concept_metric_results()

    # Kingdom: 2 of the 3 labelled pixels are right. Bleached 1/3 and sand 2/3 average to 0.5.
    assert results == {
        "accuracy/kingdom": pytest.approx(2 / 3),
        "accuracy/multi_hot": pytest.approx(0.5),
    }


def test_computing_the_concept_metrics_resets_them():
    evaluator = _evaluator()
    outputs, labels = _mixed_batch()
    evaluator.evaluate_concepts(outputs, labels)
    evaluator.compute_concept_metric_results()
    perfect_outputs, perfect_labels = _batch(
        kingdom_labels=[[1.0, 0.0], [0.0, 1.0]],
        kingdom_scores=[[0.9, 0.1], [0.1, 0.9]],
        bleached=([2.0, 1.0], [0.9, 0.1]),
        sand=([2.0, 1.0], [0.9, 0.1]),
    )

    evaluator.evaluate_concepts(perfect_outputs, perfect_labels)

    assert evaluator.compute_concept_metric_results() == {
        "accuracy/kingdom": pytest.approx(1.0),
        "accuracy/multi_hot": pytest.approx(1.0),
    }
