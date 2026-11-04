"""Integration-style tests for train_model wiring and metric policy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mermaidseg.model.train as train_module
from mermaidseg.model.train import (
    _enforce_load_failure_rate,
    _loader_load_failure_count,
    train_model,
)


class StubLogger:
    """Minimal logger stub for train_model tests."""

    def __init__(self, log_epochs: int = 1):
        self.log_epochs = log_epochs
        self.logged: list[tuple[dict[str, float], int]] = []
        self.checkpoint_epochs: list[int] = []

    def log(self, payload: dict[str, float], step: int) -> None:
        self.logged.append((payload, step))

    def save_model_checkpoint(self, _meta_model, epoch: int, _metrics: dict[str, float]) -> None:
        self.checkpoint_epochs.append(epoch)


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
        self.global_step = 0

    def train_epoch(
        self,
        _loader,
        _evaluator,
        run_logger=None,
        log_interval=1,
        epoch=None,
        on_optimizer_step=None,
    ):
        # Advance the global step like the real MetaModel so train_model logs against it.
        self.global_step += 1
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


def test_save_best_checkpoint_can_be_disabled() -> None:
    meta = FakeMetaModel(epochs=1, val_metrics_seq=[{"accuracy": 0.9}])
    logger = StubLogger()
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
        save_best_checkpoint=False,
    )
    assert logger.checkpoint_epochs == []


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


@pytest.mark.parametrize(
    ("accuracies", "epochs_run"),
    [
        # 0.55 beats 0.50 by 0.05 < 0.1: no improvement, patience 1 stops after epoch 1.
        ([0.5, 0.55, 0.9, 0.9, 0.9], 2),
        # 0.65 beats 0.50 by 0.15 > 0.1: improvement; epoch 2 repeats it and stops there.
        ([0.5, 0.65, 0.65, 0.9, 0.9], 3),
    ],
)
def test_early_stopping_triggers_with_patience_and_min_delta(accuracies, epochs_run) -> None:
    meta = FakeMetaModel(
        epochs=5,
        val_losses=[0.9] * 5,
        val_metrics_seq=[{"accuracy": a} for a in accuracies],
    )
    metrics = train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        metric_of_interest="accuracy",
        early_stopping=True,
        early_stopping_patience=1,
        early_stopping_min_delta=0.1,
    )

    assert len(metrics) == epochs_run


def test_early_stopping_runs_final_test_eval_on_stop_epoch(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def _fake_evaluate_and_log(
        evaluator,
        loader,
        meta_model,
        logger,
        epoch,
        split="train",
        step=None,
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


def test_loader_load_failure_count_returns_none_for_untracked_dataset() -> None:
    assert _loader_load_failure_count(_FakeLoader(object())) is None
    assert _loader_load_failure_count(_FakeLoader(_FakeFailDataset(100, 7))) == 7


def test_enforce_load_failure_rate_raises_above_threshold() -> None:
    loader = _FakeLoader(_FakeFailDataset(size=100, failures=10))
    with pytest.raises(RuntimeError, match="load-failure rate"):
        _enforce_load_failure_rate(loader, failures_before=0, max_rate=0.05, epoch=0)


def test_enforce_load_failure_rate_uses_per_epoch_delta() -> None:
    # 100 cumulative failures but only 2 new this epoch on a 100-item set -> 2% < 5%, no raise.
    loader = _FakeLoader(_FakeFailDataset(size=100, failures=100))
    _enforce_load_failure_rate(loader, failures_before=98, max_rate=0.05, epoch=3)


def test_enforce_load_failure_rate_allows_below_threshold() -> None:
    loader = _FakeLoader(_FakeFailDataset(size=100, failures=3))
    _enforce_load_failure_rate(loader, failures_before=0, max_rate=0.05, epoch=0)


def test_enforce_load_failure_rate_disabled_when_none() -> None:
    loader = _FakeLoader(_FakeFailDataset(size=100, failures=99))
    _enforce_load_failure_rate(loader, failures_before=0, max_rate=None, epoch=0)


def test_enforce_load_failure_rate_noop_for_untracked_dataset() -> None:
    # Untracked dataset -> failures_before is None -> guard is a no-op (does not raise).
    _enforce_load_failure_rate(_FakeLoader(object()), failures_before=None, max_rate=0.05, epoch=0)


def test_train_model_logs_main_metric_set_to_logger() -> None:
    """Regression guard: train/val metrics AND per-epoch timing/loss reach the logger.

    The branch had replaced unfiltered logging with a filter that dropped timing/gpu/raw-loss
    metrics; this asserts main's metric set is tracked again.
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


def test_train_model_logs_against_global_step_axis() -> None:
    """Epoch/val metrics are logged at meta_model.global_step, not the epoch index.

    FakeMetaModel.train_epoch advances global_step by 1 per epoch, so across two epochs
    the logged steps should be [1, 2] (global step) rather than [0, 1] (epoch).
    """
    meta = FakeMetaModel(
        epochs=2,
        val_losses=[0.9, 0.9],
        val_metrics_seq=[{"accuracy": 0.5}, {"accuracy": 0.5}],
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
    logged_steps = sorted({step for _, step in logger.logged})
    assert logged_steps == [1, 2], f"expected global-step axis [1, 2], got {logged_steps}"


def test_train_model_logs_test_time_taken_to_logger(monkeypatch) -> None:
    """Regression guard (#PR2 review): test/time_taken must reach the logger.

    epoch_loss_dict is logged before the test split is evaluated, so test/time_taken is added after
    that log and must be logged separately — otherwise it silently never reaches MLflow.
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


@pytest.mark.parametrize(("warmup_done", "expected_lr"), [(5, 0.1), (10, 0.05)])
def test_epoch_scheduler_waits_for_warmup(warmup_done, expected_lr) -> None:
    meta = FakeMetaModel(epochs=1)
    meta.scheduler = torch.optim.lr_scheduler.StepLR(meta.optimizer, step_size=1, gamma=0.5)
    meta.warmup_iters = 10
    meta._warmup_iters_completed = warmup_done
    train_model(meta_model=meta, evaluator=object(), train_loader=_tiny_loader())
    assert meta.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_enforce_load_failure_rate_allows_rate_equal_to_threshold() -> None:
    loader = _FakeLoader(_FakeFailDataset(size=100, failures=5))
    _enforce_load_failure_rate(loader, failures_before=0, max_rate=0.05, epoch=0)


def test_enforce_load_failure_rate_noop_for_empty_dataset() -> None:
    loader = _FakeLoader(_FakeFailDataset(size=0, failures=3))
    _enforce_load_failure_rate(loader, failures_before=0, max_rate=0.05, epoch=0)


def test_min_delta_applies_to_minimised_metric() -> None:
    # 0.95 is below the best 1.0 by less than 0.1, so only 0.85 counts as an improvement.
    meta = FakeMetaModel(epochs=3, val_losses=[1.0, 0.95, 0.85])
    logger = StubLogger()
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
        metric_of_interest="loss",
        early_stopping_min_delta=0.1,
    )
    assert logger.checkpoint_epochs == [0, 2]
