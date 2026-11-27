"""Metric logging helpers of the training loop."""

from __future__ import annotations

import numpy as np

from mermaidseg.model.train import _log_metric_dict


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int]] = []

    def log(self, metrics: dict[str, object], step: int) -> None:
        self.calls.append((metrics, step))


class TestLogMetricDict:
    def test_logs_each_metric_under_the_prefix_at_the_step(self):
        logger = _RecordingLogger()
        miou = np.array([0.5, 0.25])

        _log_metric_dict(logger, "validation", {"accuracy": 0.75, "miou": miou}, step=40)

        assert logger.calls == [({"validation/accuracy": 0.75, "validation/miou": miou}, 40)]
