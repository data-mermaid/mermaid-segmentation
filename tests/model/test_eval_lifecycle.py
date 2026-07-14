"""Tests for the Evaluator metric lifecycle (accumulate / compute_and_reset).

Seam 1: the metric update/compute/reset lifecycle now lives on Evaluator (was duplicated
inline in MetaModel.train_epoch/validation_epoch and Evaluator.evaluate_model). These
lock the extracted behavior: logits-vs- class-id handling, scalar results, and that
compute_and_reset actually resets so metrics don't bleed across epochs.
"""

from __future__ import annotations

import torch

from mermaidseg.model.eval import Evaluator

NUM_CLASSES = 3  # ids: 0 = ignore, 1..2 = classes


def _logits_predicting(class_id: int, *, bs=1, size=4) -> torch.Tensor:
    logits = torch.zeros(bs, NUM_CLASSES, size, size)
    logits[:, class_id] = 10.0  # argmax -> class_id everywhere
    return logits


def _targets(class_id: int, *, bs=1, size=4) -> torch.Tensor:
    return torch.full((bs, size, size), class_id, dtype=torch.long)


def test_accumulate_then_compute_and_reset_returns_scalar_metrics():
    ev = Evaluator(num_classes=NUM_CLASSES, device="cpu")
    ev.accumulate(_logits_predicting(1), _targets(1))  # perfect predictions of class 1
    results = ev.compute_and_reset()
    assert set(results) >= {"accuracy", "miou"}
    assert isinstance(results["accuracy"], float) and results["accuracy"] == 1.0
    assert isinstance(results["miou"], float) and 0.0 <= results["miou"] <= 1.0


def test_compute_and_reset_actually_resets_between_epochs():
    ev = Evaluator(num_classes=NUM_CLASSES, device="cpu")
    ev.accumulate(_logits_predicting(1), _targets(1))
    assert ev.compute_and_reset()["accuracy"] == 1.0
    # Fresh epoch: all-wrong predictions. If reset didn't happen, the prior perfect
    # batch would still count and accuracy would be > 0.
    ev.accumulate(_logits_predicting(2), _targets(1))
    assert ev.compute_and_reset()["accuracy"] == 0.0


def test_accumulate_accepts_class_id_maps_equivalently_to_logits():
    ev_logits = Evaluator(num_classes=NUM_CLASSES, device="cpu")
    ev_logits.accumulate(_logits_predicting(1), _targets(1))  # (B, C, H, W)

    ev_ids = Evaluator(num_classes=NUM_CLASSES, device="cpu")
    ev_ids.accumulate(_targets(1), _targets(1))  # (B, H, W) class-id map, no argmax

    assert (
        ev_logits.compute_and_reset()["accuracy"] == ev_ids.compute_and_reset()["accuracy"] == 1.0
    )


def test_empty_bank_is_a_noop():
    ev = Evaluator(num_classes=NUM_CLASSES, device="cpu", include_classification=False)
    assert ev.metric_dict == {}
    ev.accumulate(_logits_predicting(1), _targets(1))  # no-op, must not raise
    assert ev.compute_and_reset() == {}
