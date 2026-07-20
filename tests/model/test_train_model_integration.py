"""Integration-style tests for train_model wiring and metric policy."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import mermaidseg.model.train as train_module
from mermaidseg.model.train import (
    _enforce_load_failure_rate,
    _loader_load_failure_count,
    train_model,
)
from scripts.train import (
    _build_parser,
    _configure_third_party_loggers,
    _save_failure_report_if_available,
    _save_failure_reports_if_available,
    _take_first_non_empty_batch,
)


class StubLogger:
    """Minimal logger stub for train_model tests."""

    def __init__(self, log_epochs: int = 1, log_checkpoint: int | None = None):
        self.log_epochs = log_epochs
        self.log_checkpoint = log_checkpoint
        self.logged: list[tuple[dict[str, float], int]] = []
        self.checkpoint_epochs: list[int] = []
        self.checkpoint_is_best: list[bool] = []

    def log(self, payload: dict[str, float], step: int) -> None:
        self.logged.append((payload, step))

    def save_model_checkpoint(
        self, _meta_model, epoch: int, _metrics: dict[str, float], is_best: bool = True
    ) -> None:
        self.checkpoint_epochs.append(epoch)
        self.checkpoint_is_best.append(is_best)


class FakeMetaModel:
    """Small, deterministic MetaModel substitute for loop-control testing."""

    def __init__(
        self,
        *,
        epochs: int = 3,
        train_loss: float = 1.0,
        train_metrics: dict[str, float] | None = None,
        val_losses: list[float] | None = None,
        val_metrics_seq: list[dict[str, float | np.ndarray]] | None = None,
    ):
        self.training_kwargs = SimpleNamespace(epochs=epochs)
        self.model = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self._train_loss = train_loss
        self._train_metrics = train_metrics or {"accuracy": 0.5}
        self._val_losses = val_losses or [0.9] * epochs
        self._val_metrics_seq = val_metrics_seq or [{"accuracy": 0.5}] * epochs
        self._val_idx = 0

    def train_epoch(self, _loader, _evaluator):
        timing = {
            "data_loading_sec": 0.0,
            "forward_sec": 0.0,
            "backward_sec": 0.0,
            "num_samples": 2,
        }
        return self._train_loss, dict(self._train_metrics), timing

    def validation_epoch(self, _loader, _evaluator):
        idx = min(self._val_idx, len(self._val_metrics_seq) - 1)
        self._val_idx += 1
        return self._val_losses[idx], dict(self._val_metrics_seq[idx])


class RecordingPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """ReduceLROnPlateau that records received metrics."""

    def __init__(self, optimizer):
        super().__init__(optimizer, mode="max", factor=0.5, patience=1)
        self.metric_history: list[float] = []

    def step(self, metrics, epoch=None):  # type: ignore[override]
        self.metric_history.append(float(metrics))
        return super().step(metrics, epoch)


def _tiny_loader():
    images = torch.randn(2, 3, 8, 8)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    return [(images, labels)]


def test_metric_of_interest_allowlist_rejects_unknown_metric() -> None:
    meta = FakeMetaModel(epochs=1)
    with pytest.raises(ValueError, match="Unsupported metric_of_interest"):
        train_model(
            meta_model=meta,
            evaluator=object(),
            train_loader=_tiny_loader(),
            metric_of_interest="precision",
        )


def test_train_model_logs_per_dataset_validation_metrics() -> None:
    meta = FakeMetaModel(
        epochs=1,
        val_losses=[0.5, 0.4, 0.6],
        val_metrics_seq=[{"miou": 0.5}, {"miou": 0.4}, {"miou": 0.6}],
    )
    logger = StubLogger()
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        dataset_val_loaders={"coralnet": _tiny_loader(), "mermaid": _tiny_loader()},
        logger=logger,
        metric_of_interest="miou",
    )
    logged_keys = {k for payload, _ in logger.logged for k in payload}
    assert "validation/mermaid/miou" in logged_keys
    assert "validation/coralnet/miou" in logged_keys
    assert "validation/mermaid/loss" in logged_keys
    assert "validation/coralnet/loss" in logged_keys


@pytest.mark.parametrize("metric_name", ["accuracy", "miou", "f1-score", "loss"])
def test_metric_of_interest_allowlist_accepts_known_metrics(metric_name: str) -> None:
    val_metrics = {
        "accuracy": {"accuracy": 0.7},
        "miou": {"miou": 0.4},
        "f1-score": {"f1-score": 0.6},
        "loss": {"accuracy": 0.5},
    }[metric_name]
    val_losses = [0.8]
    meta = FakeMetaModel(epochs=1, val_losses=val_losses, val_metrics_seq=[val_metrics])
    logger = StubLogger()

    metrics = train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest=metric_name,
    )

    assert len(metrics) == 1
    assert logger.checkpoint_epochs == [0]


def test_metric_direction_accuracy_maximize_and_loss_minimize() -> None:
    logger_accuracy = StubLogger()
    meta_accuracy = FakeMetaModel(
        epochs=2,
        val_losses=[0.9, 0.9],
        val_metrics_seq=[{"accuracy": 0.7}, {"accuracy": 0.6}],
    )
    train_model(
        meta_model=meta_accuracy,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger_accuracy,
        metric_of_interest="accuracy",
    )
    assert logger_accuracy.checkpoint_epochs == [0]

    logger_loss = StubLogger()
    meta_loss = FakeMetaModel(
        epochs=2,
        val_losses=[1.0, 0.8],
        val_metrics_seq=[{"accuracy": 0.5}, {"accuracy": 0.5}],
    )
    train_model(
        meta_model=meta_loss,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger_loss,
        metric_of_interest="loss",
    )
    assert logger_loss.checkpoint_epochs == [0, 1]


def test_log_checkpoint_triggers_periodic_non_improvement_checkpoint() -> None:
    """A flat validation metric never improves — log_checkpoint must still
    checkpoint."""
    meta = FakeMetaModel(
        epochs=4,
        val_losses=[0.9, 0.9, 0.9, 0.9],
        val_metrics_seq=[{"accuracy": 0.5}] * 4,
    )
    logger = StubLogger(log_checkpoint=2)
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest="accuracy",
    )
    # Epoch 0 improves (first result always beats -inf/inf); epochs 2 are periodic-only.
    assert logger.checkpoint_epochs == [0, 2]
    assert logger.checkpoint_is_best == [True, False]


def test_log_checkpoint_does_not_duplicate_improvement_epoch() -> None:
    """When improvement and the periodic interval land on the same epoch, save once."""
    meta = FakeMetaModel(
        epochs=3,
        val_losses=[0.9, 0.8, 0.9],
        val_metrics_seq=[{"accuracy": 0.5}, {"accuracy": 0.6}, {"accuracy": 0.5}],
    )
    logger = StubLogger(log_checkpoint=1)
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest="accuracy",
    )
    assert logger.checkpoint_epochs == [0, 1, 2]
    # Every epoch improves except epoch 2 — should stay best=True for 0,1 and
    # fall back to periodic (best=False) only for the non-improving epoch 2.
    assert logger.checkpoint_is_best == [True, True, False]


def test_no_log_checkpoint_attribute_disables_periodic_checkpointing() -> None:
    """Loggers without a log_checkpoint attribute must not trigger periodic saves."""
    meta = FakeMetaModel(
        epochs=3,
        val_losses=[0.9, 0.9, 0.9],
        val_metrics_seq=[{"accuracy": 0.5}] * 3,
    )
    logger = StubLogger()
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest="accuracy",
    )
    assert logger.checkpoint_epochs == [0]


def test_end_epoch_is_honored_when_start_epoch_default() -> None:
    meta = FakeMetaModel(epochs=5)
    metrics = train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        start_epoch=-1,
        end_epoch=2,
    )
    assert list(metrics.keys()) == [0, 1]


def test_reduce_on_plateau_receives_validation_metric() -> None:
    meta = FakeMetaModel(
        epochs=2,
        val_losses=[0.9, 0.9],
        val_metrics_seq=[{"accuracy": 0.4}, {"accuracy": 0.5}],
    )
    scheduler = RecordingPlateau(meta.optimizer)
    meta.scheduler = scheduler

    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        metric_of_interest="accuracy",
    )

    assert scheduler.metric_history == [0.4, 0.5]


def test_non_scalar_metric_of_interest_raises_value_error() -> None:
    meta = FakeMetaModel(
        epochs=1,
        val_losses=[0.9],
        val_metrics_seq=[{"miou": np.array([0.1, 0.2])}],
    )
    with pytest.raises(ValueError, match="must be scalar"):
        train_model(
            meta_model=meta,
            evaluator=object(),
            train_loader=_tiny_loader(),
            val_loader=_tiny_loader(),
            metric_of_interest="miou",
        )


def test_early_stopping_triggers_with_patience_and_min_delta() -> None:
    meta = FakeMetaModel(
        epochs=5,
        val_losses=[0.9, 0.9, 0.9, 0.9, 0.9],
        val_metrics_seq=[
            {"accuracy": 0.5},
            {"accuracy": 0.5},
            {"accuracy": 0.5},
            {"accuracy": 0.5},
            {"accuracy": 0.5},
        ],
    )
    metrics = train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        metric_of_interest="accuracy",
        early_stopping=True,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
    )

    assert len(metrics) == 2


def test_early_stopping_runs_final_test_eval_on_stop_epoch(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def _fake_evaluate_and_log(
        evaluator,
        loader,
        meta_model,
        logger,
        epoch,
        split="train",
    ):
        calls.append((epoch, split))
        return {"accuracy": 0.5}

    monkeypatch.setattr(train_module, "evaluate_and_log", _fake_evaluate_and_log)

    meta = FakeMetaModel(
        epochs=5,
        val_losses=[0.9, 0.9, 0.9, 0.9, 0.9],
        val_metrics_seq=[
            {"accuracy": 0.5},
            {"accuracy": 0.5},
            {"accuracy": 0.5},
            {"accuracy": 0.5},
            {"accuracy": 0.5},
        ],
    )
    logger = StubLogger(log_epochs=100)
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        test_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest="accuracy",
        early_stopping=True,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
    )

    assert calls[-1] == (1, "test")
    assert len(calls) >= 2


def test_cli_metric_of_interest_argument_is_supported() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/data_config.yaml",
            "--metric-of-interest",
            "miou",
        ]
    )
    assert args.metric_of_interest == "miou"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "configs/data_config.yaml",
                "--metric-of-interest",
                "precision",
            ]
        )


def test_configure_third_party_loggers_suppresses_botocore_token_info() -> None:
    botocore_logger = logging.getLogger("botocore.tokens")
    original_level = botocore_logger.level
    try:
        botocore_logger.setLevel(logging.INFO)
        _configure_third_party_loggers()
        assert botocore_logger.level == logging.WARNING
    finally:
        botocore_logger.setLevel(original_level)


def test_take_first_non_empty_batch_skips_empty_batch() -> None:
    empty = (torch.tensor([]), torch.tensor([]))
    valid = (
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 8, 8, dtype=torch.long),
    )
    picked = _take_first_non_empty_batch([empty, valid], "train")
    assert picked[0].shape[0] == 1


def test_take_first_non_empty_batch_raises_if_none_found() -> None:
    with pytest.raises(RuntimeError, match="non-empty batch"):
        _take_first_non_empty_batch([(torch.tensor([]), torch.tensor([]))], "train")


def test_save_failure_report_if_available_returns_none_without_failures(tmp_path: Path) -> None:
    class _Dataset:
        @staticmethod
        def num_load_failures() -> int:
            return 0

        @staticmethod
        def save_load_failures(_path):
            raise AssertionError(
                "save_load_failures should not be called when there are no failures"
            )

    assert _save_failure_report_if_available(_Dataset(), tmp_path) is None


def test_save_failure_report_if_available_writes_parquet(tmp_path: Path) -> None:
    class _Dataset:
        @staticmethod
        def num_load_failures() -> int:
            return 2

        @staticmethod
        def save_load_failures(path):
            pd.DataFrame([{"image_id": "a"}, {"image_id": "b"}]).to_parquet(path, index=False)
            return Path(path)

    report_path = _save_failure_report_if_available(_Dataset(), tmp_path)
    assert report_path is not None
    assert report_path.exists()
    assert report_path.suffix == ".parquet"


class _FailingDataset:
    def __init__(self, n_failures: int) -> None:
        self._n_failures = n_failures

    def num_load_failures(self) -> int:
        return self._n_failures

    def save_load_failures(self, path):
        pd.DataFrame([{"image_id": "a"}] * self._n_failures).to_parquet(path, index=False)
        return Path(path)


def test_save_failure_reports_if_available_writes_one_report_per_failing_dataset(
    tmp_path: Path,
) -> None:
    dataset_dict = {
        ("mermaid", "train"): _FailingDataset(3),
        ("mermaid", "val"): _FailingDataset(0),
        ("coralnet", "train"): _FailingDataset(1),
    }
    paths = _save_failure_reports_if_available(dataset_dict, tmp_path)
    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.suffix == ".parquet"


def test_save_failure_reports_if_available_returns_empty_without_any_failures(
    tmp_path: Path,
) -> None:
    dataset_dict = {
        ("mermaid", "train"): _FailingDataset(0),
        ("coralnet", "train"): _FailingDataset(0),
    }
    assert _save_failure_reports_if_available(dataset_dict, tmp_path) == []


def test_save_failure_reports_if_available_suffixes_explicit_output_path(tmp_path: Path) -> None:
    dataset_dict = {
        ("mermaid", "train"): _FailingDataset(2),
        ("coralnet", "train"): _FailingDataset(1),
    }
    explicit = tmp_path / "failures.parquet"
    paths = _save_failure_reports_if_available(dataset_dict, tmp_path, str(explicit))
    names = {p.name for p in paths}
    assert names == {"failures_mermaid_train.parquet", "failures_coralnet_train.parquet"}


class _FakeFailDataset:
    """Dataset stub exposing the load-failure tracking API used by the rate guard."""

    def __init__(self, size: int, failures: int) -> None:
        self._size = size
        self._failures = failures

    def __len__(self) -> int:
        return self._size

    def num_load_failures(self) -> int:
        return self._failures


class _FakeLoader:
    def __init__(self, dataset: object) -> None:
        self.dataset = dataset


class _FakeConcat:
    """Mimics torch ConcatDataset: no num_load_failures of its own, exposes
    .datasets."""

    def __init__(self, datasets: list) -> None:
        self.datasets = list(datasets)

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)


def test_loader_load_failure_count_returns_none_for_untracked_dataset() -> None:
    assert _loader_load_failure_count(_FakeLoader(object())) is None
    assert _loader_load_failure_count(_FakeLoader(_FakeFailDataset(100, 7))) == 7


def test_loader_load_failure_count_sums_concat_dataset_children() -> None:
    # The real training path wraps per-source datasets in a ConcatDataset, which has no
    # num_load_failures of its own; the count must sum the tracked children (else the guard
    # silently no-ops on every real run).
    concat = _FakeConcat([_FakeFailDataset(50, 3), _FakeFailDataset(50, 4)])
    assert _loader_load_failure_count(_FakeLoader(concat)) == 7


def test_enforce_load_failure_rate_raises_above_threshold() -> None:
    # 10 new failures out of 100 attempts (90 processed + 10 failed) = 10% > 5% -> raise.
    loader = _FakeLoader(_FakeFailDataset(size=1000, failures=10))
    with pytest.raises(RuntimeError, match="load-failure rate"):
        _enforce_load_failure_rate(
            loader, failures_before=0, max_rate=0.05, epoch=0, samples_processed=90
        )


def test_enforce_load_failure_rate_uses_per_epoch_delta() -> None:
    # 100 cumulative failures but only 2 new this epoch; 2/(98+2)=2% < 5% -> no raise.
    loader = _FakeLoader(_FakeFailDataset(size=1000, failures=100))
    _enforce_load_failure_rate(
        loader, failures_before=98, max_rate=0.05, epoch=3, samples_processed=98
    )


def test_enforce_load_failure_rate_rate_is_over_epoch_attempts_not_dataset_size() -> None:
    # Regression: denominator is per-epoch attempts, NOT len(dataset). 10 failures over a
    # huge 1e6-item dataset but only 90 processed this epoch -> 10% -> must still raise.
    loader = _FakeLoader(_FakeFailDataset(size=1_000_000, failures=10))
    with pytest.raises(RuntimeError, match="load-failure rate"):
        _enforce_load_failure_rate(
            loader, failures_before=0, max_rate=0.05, epoch=0, samples_processed=90
        )


def test_enforce_load_failure_rate_allows_below_threshold() -> None:
    loader = _FakeLoader(_FakeFailDataset(size=1000, failures=3))
    _enforce_load_failure_rate(
        loader, failures_before=0, max_rate=0.05, epoch=0, samples_processed=97
    )


def test_enforce_load_failure_rate_disabled_when_none() -> None:
    loader = _FakeLoader(_FakeFailDataset(size=1000, failures=99))
    _enforce_load_failure_rate(
        loader, failures_before=0, max_rate=None, epoch=0, samples_processed=1
    )


def test_enforce_load_failure_rate_noop_for_untracked_dataset() -> None:
    # Untracked dataset -> failures_before is None -> guard is a no-op (does not raise).
    _enforce_load_failure_rate(
        _FakeLoader(object()), failures_before=None, max_rate=0.05, epoch=0, samples_processed=100
    )


def test_train_model_logs_main_metric_set_to_logger() -> None:
    """Regression guard: train/val metrics AND per-epoch timing/loss reach the logger.

    The branch had replaced unfiltered logging with a filter that dropped
    timing/gpu/raw-loss metrics; this asserts main's metric set is tracked again.
    """
    meta = FakeMetaModel(epochs=1, val_losses=[0.8], val_metrics_seq=[{"accuracy": 0.7}])
    logger = StubLogger()
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest="accuracy",
    )
    logged_keys = {key for payload, _ in logger.logged for key in payload}
    for key in (
        "train/accuracy",
        "validation/accuracy",
        "train/loss",
        "validation/loss",
        "train/time_taken",
        "train/samples_per_sec",
    ):
        assert key in logged_keys, f"{key!r} was not logged (logged: {sorted(logged_keys)})"


def test_train_model_logs_test_time_taken_to_logger(monkeypatch) -> None:
    """Regression guard (#PR2 review): test/time_taken must reach the logger.

    epoch_loss_dict is logged before the test split is evaluated, so test/time_taken is
    added after that log and must be logged separately — otherwise it silently never
    reaches MLflow.
    """
    monkeypatch.setattr(train_module, "evaluate_and_log", lambda *a, **k: {"accuracy": 0.5})
    meta = FakeMetaModel(epochs=1, val_losses=[0.8], val_metrics_seq=[{"accuracy": 0.7}])
    logger = StubLogger(log_epochs=1)
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        test_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest="accuracy",
    )
    logged_keys = {key for payload, _ in logger.logged for key in payload}
    assert "test/time_taken" in logged_keys, (
        f"test/time_taken not logged (got {sorted(logged_keys)})"
    )
