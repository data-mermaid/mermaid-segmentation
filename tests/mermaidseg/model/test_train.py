"""Load-failure guard and metric logging helpers of the training loop."""

from __future__ import annotations

import numpy as np
import pytest

from mermaidseg.model.train import (
    _enforce_load_failure_rate,
    _loader_load_failure_count,
    _log_metric_dict,
)


class _TrackedDataset:
    def __init__(self, size: int, failures: int) -> None:
        self.size = size
        self.failures = failures

    def __len__(self) -> int:
        return self.size

    def num_load_failures(self) -> int:
        return self.failures


class _Loader:
    def __init__(self, dataset: object) -> None:
        self.dataset = dataset


class _BrokenDataset:
    def num_load_failures(self) -> int:
        raise OSError("failure report unreadable")


class _UntrackedDataset:
    def __len__(self) -> int:
        return 10


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int]] = []

    def log(self, metrics: dict[str, object], step: int) -> None:
        self.calls.append((metrics, step))


class TestLoaderLoadFailureCount:
    def test_reads_the_cumulative_count_as_an_int(self):
        assert _loader_load_failure_count(_Loader(_TrackedDataset(size=10, failures=3))) == 3

    @pytest.mark.parametrize(
        "loader",
        [object(), _Loader(None), _Loader(_UntrackedDataset()), _Loader(_BrokenDataset())],
        ids=["no_dataset_attr", "dataset_none", "untracked", "count_raises"],
    )
    def test_is_none_when_the_count_is_unavailable(self, loader):
        assert _loader_load_failure_count(loader) is None


class TestEnforceLoadFailureRate:
    @pytest.mark.parametrize(
        ("failures_now", "raises"),
        [(15, False), (16, True)],
        ids=["at_the_rate", "just_over"],
    )
    def test_raises_only_when_the_epoch_rate_exceeds_the_limit(self, failures_now, raises):
        # 10 failures predate the epoch; the limit is 5 of 100 samples.
        loader = _Loader(_TrackedDataset(size=100, failures=failures_now))

        if not raises:
            _enforce_load_failure_rate(loader, 10, 0.05, epoch=3)
            return
        with pytest.raises(RuntimeError) as excinfo:
            _enforce_load_failure_rate(loader, 10, 0.05, epoch=3, split="validation")
        message = str(excinfo.value)
        assert "Epoch 3: validation load-failure rate 6.0% (6/100)" in message
        assert "max_load_failure_rate=5.0%" in message

    def test_split_defaults_to_train(self):
        loader = _Loader(_TrackedDataset(size=10, failures=10))

        with pytest.raises(RuntimeError, match="Epoch 0: train load-failure rate 100.0% .10/10."):
            _enforce_load_failure_rate(loader, 0, 0.5, epoch=0)

    @pytest.mark.parametrize(
        ("loader", "failures_before", "max_rate"),
        [
            (_Loader(_TrackedDataset(size=10, failures=10)), 0, None),
            (_Loader(_TrackedDataset(size=10, failures=10)), None, 0.0),
            (_Loader(_TrackedDataset(size=0, failures=10)), 0, 0.0),
            (_Loader(None), 0, 0.0),
        ],
        ids=["disabled", "untracked", "empty_dataset", "no_dataset"],
    )
    def test_is_a_no_op_when_it_cannot_judge_a_rate(self, loader, failures_before, max_rate):
        _enforce_load_failure_rate(loader, failures_before, max_rate, epoch=0)


class TestLogMetricDict:
    def test_logs_each_metric_under_the_prefix_at_the_step(self):
        logger = _RecordingLogger()
        miou = np.array([0.5, 0.25])

        _log_metric_dict(logger, "validation", {"accuracy": 0.75, "miou": miou}, step=40)

        assert logger.calls == [({"validation/accuracy": 0.75, "validation/miou": miou}, 40)]

    def test_logs_nothing_for_empty_metrics(self):
        logger = _RecordingLogger()

        _log_metric_dict(logger, "train", {}, step=1)

        assert logger.calls == []

    def test_accepts_no_logger(self):
        _log_metric_dict(None, "train", {"accuracy": 0.5}, step=1)
