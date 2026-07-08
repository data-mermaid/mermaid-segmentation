"""Tests for Evaluator's per_class_metrics toggle (f1_per_class / iou_per_class)."""

from __future__ import annotations

import numpy as np
import torch

from mermaidseg.model.eval import Evaluator

NUM_CLASSES = 4


class _StubMetaModel:
    """Minimal meta-model stand-in: identity label mapping, scripted predictions."""

    def __init__(self, batches: list[torch.Tensor], training_mode: str = "standard"):
        self.training_mode = training_mode
        self.model = torch.nn.Identity()
        self._batches = batches
        self._call = 0

    def _to_target_labels(self, source_labels: torch.Tensor) -> torch.Tensor:
        return source_labels

    def batch_predict(self, _inputs):
        preds = self._batches[self._call]
        self._call += 1
        return preds, None


def _make_dataloader(preds: torch.Tensor, targets: torch.Tensor):
    """Single-batch dataloader stand-in; inputs are unused by _StubMetaModel."""
    return [(torch.zeros(1), targets)]


class TestPerClassMetricsToggle:
    def test_disabled_by_default(self):
        evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")
        assert "f1_per_class" not in evaluator.metric_dict
        assert "iou_per_class" not in evaluator.metric_dict
        assert set(evaluator.metric_dict) == {"accuracy", "miou"}

    def test_enabled_adds_per_class_metrics(self):
        evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu", per_class_metrics=True)
        assert "f1_per_class" in evaluator.metric_dict
        assert "iou_per_class" in evaluator.metric_dict

    def test_no_op_when_classification_disabled(self):
        """per_class_metrics should not resurrect classification metrics on its own."""
        evaluator = Evaluator(
            num_classes=NUM_CLASSES,
            device="cpu",
            per_class_metrics=True,
            include_classification=False,
        )
        assert evaluator.metric_dict == {}


class TestPerClassMetricsComputation:
    def test_per_class_results_are_arrays_sized_to_num_classes(self):
        # ignore_index=-1 (no pixels match) so all 4 classes have support and a defined score.
        evaluator = Evaluator(
            num_classes=NUM_CLASSES, device="cpu", per_class_metrics=True, ignore_index=-1
        )
        targets = torch.tensor([[0, 1], [2, 3]])
        preds = targets.clone()  # perfect predictions
        meta_model = _StubMetaModel([preds])
        dataloader = _make_dataloader(preds, targets)

        results = evaluator.evaluate_model(dataloader, meta_model)

        assert isinstance(results["f1_per_class"], np.ndarray)
        assert results["f1_per_class"].shape == (NUM_CLASSES,)
        assert isinstance(results["iou_per_class"], np.ndarray)
        assert results["iou_per_class"].shape == (NUM_CLASSES,)
        # Scalar metrics remain scalar even when per-class metrics are also enabled.
        assert isinstance(results["accuracy"], float)
        assert isinstance(results["miou"], float)

    def test_perfect_predictions_score_one_per_class(self):
        evaluator = Evaluator(
            num_classes=NUM_CLASSES, device="cpu", per_class_metrics=True, ignore_index=-1
        )
        targets = torch.tensor([[0, 1, 2, 3]])
        preds = targets.clone()
        meta_model = _StubMetaModel([preds])
        dataloader = _make_dataloader(preds, targets)

        results = evaluator.evaluate_model(dataloader, meta_model)

        np.testing.assert_allclose(results["f1_per_class"], np.ones(NUM_CLASSES))
        np.testing.assert_allclose(results["iou_per_class"], np.ones(NUM_CLASSES))
