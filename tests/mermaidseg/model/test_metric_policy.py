"""Unit tests for mermaidseg.model.metric_policy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mermaidseg.model.metric_policy import (
    SUPPORTED_METRIC_NAMES,
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

    def test_supported_names_are_canonical_only(self):
        assert SUPPORTED_METRIC_NAMES == ("accuracy", "f1-score", "loss", "miou")


class TestMetricDirection:
    def test_loss_is_min(self):
        assert metric_direction("loss") == "min"

    @pytest.mark.parametrize("name", ["accuracy", "miou", "f1-score"])
    def test_maximize_metrics(self, name):
        assert metric_direction(name) == "max"


class TestToScalarMetric:
    @pytest.mark.parametrize(
        "value",
        [torch.tensor(0.75), np.array(0.75), 0.75, np.float32(0.75)],
        ids=["torch_tensor", "np_array", "float", "np_scalar"],
    )
    def test_accepts_scalars(self, value):
        assert to_scalar_metric("accuracy", value) == pytest.approx(0.75)

    def test_rejects_non_scalar_tensor(self):
        with pytest.raises(ValueError, match="must be scalar"):
            to_scalar_metric("accuracy", torch.tensor([0.1, 0.2]))

    def test_rejects_non_scalar_array(self):
        with pytest.raises(ValueError, match="must be scalar"):
            to_scalar_metric("miou", np.array([0.1, 0.2, 0.3]))

    def test_rejects_list(self):
        with pytest.raises(ValueError, match="must be scalar"):
            to_scalar_metric("loss", [0.1, 0.2])


class TestExtractMetricValue:
    def test_loss_uses_val_loss(self):
        assert extract_metric_value("loss", 0.85, {"accuracy": 0.7}) == pytest.approx(0.85)

    @pytest.mark.parametrize("metric", ["accuracy", "miou", "f1-score"])
    def test_extracts_from_results(self, metric):
        result = extract_metric_value(metric, 0.5, {metric: 0.82})
        assert result == pytest.approx(0.82)

    def test_raises_when_metric_missing(self):
        with pytest.raises(ValueError, match="is not present"):
            extract_metric_value("accuracy", 0.5, {"other": 0.5})

    def test_empty_results_shows_none(self):
        with pytest.raises(ValueError, match="\\(none\\)"):
            extract_metric_value("accuracy", 0.5, {})
