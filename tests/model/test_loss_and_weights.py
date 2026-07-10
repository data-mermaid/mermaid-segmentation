"""Tests for class-weight helpers and advanced segmentation losses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mermaidseg.model.class_weights import (
    build_target_name_map,
    build_weight_artifact,
    counts_from_label_series,
    inverse_sqrt_weights,
    load_class_weight_tensor,
    save_weight_artifact,
)
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.loss import (
    DomainWeightedFocalLoss,
    EdgeAwareSegmentationLoss,
    label_edge_mask,
)


class TestClassWeights:
    def test_inverse_sqrt_zeros_ignore_and_upweights_rare(self):
        counts = np.array([1000, 1000, 10], dtype=np.float64)
        weights = inverse_sqrt_weights(counts, const=0.0, ignore_index=0)
        assert weights[0] == 0.0
        assert weights[2] > weights[1]

    def test_counts_from_label_series(self):
        series = pd.Series(["Acropora", "Porites", "Acropora", "unknown"])
        name_to_target = {"Acropora": 1, "Porites": 2}
        counts = counts_from_label_series(series, name_to_target, num_classes=3)
        assert counts.tolist() == [1, 2, 1]

    def test_artifact_roundtrip(self, tmp_path: Path):
        class_subset = ["Acropora", "Porites"]
        target_name_to_id = build_target_name_map(class_subset)
        target_id2label = {i: n for n, i in target_name_to_id.items()}
        counts = np.array([5, 100, 10], dtype=np.int64)
        artifact = build_weight_artifact(
            counts=counts,
            target_id2label=target_id2label,
            class_subset=class_subset,
            const=1.0,
        )
        path = save_weight_artifact(artifact, tmp_path / "weights.json")
        loaded = load_class_weight_tensor(path)
        assert loaded.shape == (3,)
        assert loaded[0].item() == 0.0
        assert "mapping_hash" in json.loads(path.read_text())


class TestDomainWeightedFocalLoss:
    def test_shapes_ignore_and_finite_grads(self):
        loss_fn = DomainWeightedFocalLoss(
            ignore_index=0,
            gamma=2.0,
            damping_denominator=1.0,
            weight=[0.0, 1.0, 2.0],
        )
        outputs = torch.randn(2, 3, 8, 8, requires_grad=True)
        labels = torch.randint(0, 3, (2, 8, 8))
        loss, comps = loss_fn(outputs, labels)
        assert torch.isfinite(loss)
        assert "classification" in comps
        loss.backward()
        assert outputs.grad is not None

    def test_all_ignore_returns_zero(self):
        loss_fn = DomainWeightedFocalLoss(ignore_index=0, gamma=2.0)
        outputs = torch.randn(1, 3, 4, 4, requires_grad=True)
        labels = torch.zeros(1, 4, 4, dtype=torch.long)
        loss, comps = loss_fn(outputs, labels)
        assert loss.item() == 0.0
        assert comps["classification"] == 0.0

    def test_weight_path_loads(self, tmp_path: Path):
        artifact = {
            "weights": [0.0, 1.0, 3.0],
            "schema_version": 1,
        }
        path = tmp_path / "w.json"
        path.write_text(json.dumps(artifact))
        loss_fn = DomainWeightedFocalLoss(weight_path=str(path), gamma=2.0)
        assert loss_fn.weight is not None
        assert loss_fn.weight[2].item() == 3.0


class TestEdgeAwareLoss:
    def test_edge_mask_detects_boundary(self):
        labels = torch.zeros(1, 4, 4, dtype=torch.long)
        labels[0, :, 2:] = 1
        edge = label_edge_mask(labels, ignore_index=-1)
        assert edge.any()

    def test_composite_finite(self):
        loss_fn = EdgeAwareSegmentationLoss(
            base_loss_type="DomainWeightedFocalLoss",
            edge_weight=0.5,
            gamma=2.0,
            damping_denominator=1.0,
            weight=[0.0, 1.0, 1.0],
        )
        outputs = torch.randn(1, 3, 8, 8, requires_grad=True)
        labels = torch.randint(1, 3, (1, 8, 8))
        loss, comps = loss_fn(outputs, labels)
        assert torch.isfinite(loss)
        assert "edge" in comps
        loss.backward()


class TestEvaluatorMiou:
    def test_default_metrics_include_miou(self):
        evaluator = Evaluator(num_classes=4, device="cpu")
        assert "accuracy" in evaluator.metric_dict
        assert "miou" in evaluator.metric_dict

    def test_per_class_metrics_optional(self):
        evaluator = Evaluator(num_classes=4, device="cpu", per_class_metrics=True)
        assert "iou_per_class" in evaluator.metric_dict
        assert "f1_per_class" in evaluator.metric_dict
