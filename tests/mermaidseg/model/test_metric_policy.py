"""Unit tests for mermaidseg.model.metric_policy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mermaidseg.model.metric_policy import (
    canonical_metric_name,
    extract_metric_value,
    metric_direction,
    to_scalar_metric,
)


class TestCanonicalMetricName:
    @pytest.mark.parametrize("name", ["loss", "accuracy", "miou", "f1-score"])
    def test_accepts_known_metrics(self, name):
        assert canonical_metric_name(name) == name

    def test_case_insensitive(self):
        assert canonical_metric_name("LOSS") == "loss"
        assert canonical_metric_name("Accuracy") == "accuracy"

    def test_rejects_unknown_metric(self):
        with pytest.raises(ValueError, match="Unsupported metric_of_interest"):
            canonical_metric_name("precision")


class TestMetricDirection:
    @pytest.mark.parametrize(
        ("name", "direction"),
        [("loss", "min"), ("accuracy", "max"), ("miou", "max"), ("f1-score", "max")],
    )
    def test_direction(self, name, direction):
        assert metric_direction(name) == direction


class TestToScalarMetric:
    @pytest.mark.parametrize(
        "value",
        [torch.tensor(0.75), np.array(0.75), 0.75],
        ids=["torch_tensor", "np_array", "float"],
    )
    def test_accepts_scalars(self, value):
        assert to_scalar_metric("accuracy", value) == pytest.approx(0.75)

    @pytest.mark.parametrize(
        "value",
        [torch.tensor([0.1, 0.2]), np.array([0.1, 0.2, 0.3]), [0.1, 0.2]],
        ids=["torch_tensor", "np_array", "list"],
    )
    def test_rejects_non_scalars(self, value):
        with pytest.raises(ValueError, match="must be scalar"):
            to_scalar_metric("accuracy", value)


class TestExtractMetricValue:
    def test_loss_uses_val_loss(self):
        assert extract_metric_value("loss", 0.85, {"accuracy": 0.7}) == pytest.approx(0.85)

    @pytest.mark.parametrize("metric", ["accuracy", "miou", "f1-score"])
    def test_extracts_from_results(self, metric):
        result = extract_metric_value(metric, 0.5, {metric: 0.82})
        assert result == pytest.approx(0.82)

    def test_empty_results_shows_none(self):
        with pytest.raises(ValueError, match="\\(none\\)"):
            extract_metric_value("accuracy", 0.5, {})
