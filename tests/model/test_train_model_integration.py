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
from mermaidseg.model.train import train_model
from scripts.train import (
    _build_parser,
    _configure_third_party_loggers,
    _save_failure_report_if_available,
    _take_first_non_empty_batch,
)


class StubLogger:
    """Minimal logger stub for train_model tests."""

    def __init__(self, log_epochs: int = 1, checkpoint_dir: str = "."):
        self.log_epochs = log_epochs
        self.checkpoint_dir = checkpoint_dir
        self.logged: list[tuple[dict[str, float], int]] = []
        self.checkpoint_epochs: list[int] = []

    def log(self, payload: dict[str, float], step: int) -> None:
        self.logged.append((payload, step))

    def log_training_metrics(self, payload: dict[str, float], step: int) -> None:
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
        self.run_name = "fake-run"
        self.model = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self._train_loss = train_loss
        self._train_metrics = train_metrics or {"accuracy/classification": 0.5}
        self._val_losses = val_losses or [0.9] * epochs
        self._val_metrics_seq = val_metrics_seq or [{"accuracy/classification": 0.5}] * epochs
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


@pytest.mark.parametrize("metric_name", ["accuracy", "loss"])
def test_metric_of_interest_allowlist_accepts_known_metrics(metric_name: str) -> None:
    val_metrics = {
        "accuracy": {"accuracy/classification": 0.7},
        "loss": {"accuracy/classification": 0.5},
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
        val_metrics_seq=[{"accuracy/classification": 0.7}, {"accuracy/classification": 0.6}],
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
        val_metrics_seq=[{"accuracy/classification": 0.5}, {"accuracy/classification": 0.5}],
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
        val_metrics_seq=[{"accuracy/classification": 0.4}, {"accuracy/classification": 0.5}],
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
        val_metrics_seq=[{"accuracy/classification": np.array([0.1, 0.2])}],
    )
    with pytest.raises(ValueError, match="must be scalar"):
        train_model(
            meta_model=meta,
            evaluator=object(),
            train_loader=_tiny_loader(),
            val_loader=_tiny_loader(),
            metric_of_interest="accuracy",
        )


def test_early_stopping_triggers_with_patience_and_min_delta() -> None:
    meta = FakeMetaModel(
        epochs=5,
        val_losses=[0.9, 0.9, 0.9, 0.9, 0.9],
        val_metrics_seq=[
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
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
        epoch,
        split="train",
    ):
        calls.append((epoch, split))
        return {"accuracy/classification": 0.5}

    monkeypatch.setattr(train_module, "evaluate_and_log", _fake_evaluate_and_log)

    meta = FakeMetaModel(
        epochs=5,
        val_losses=[0.9, 0.9, 0.9, 0.9, 0.9],
        val_metrics_seq=[
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
            {"accuracy/classification": 0.5},
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
            "configs/linear-dinov3-base.yaml",
            "--metric-of-interest",
            "loss",
        ]
    )
    assert args.metric_of_interest == "loss"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "configs/linear-dinov3-base.yaml",
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


def test_build_epoch_metrics_prefixes_loss_and_accuracy() -> None:
    metrics = train_module.build_epoch_metrics(
        "train",
        1.25,
        {
            "loss/classification": 0.5,
            "loss/kingdom": 0.1,
            "accuracy/classification": 0.8,
            "accuracy/kingdom": 0.7,
        },
    )
    assert metrics["train/loss/total"] == pytest.approx(1.25)
    assert metrics["train/loss/classification"] == pytest.approx(0.5)
    assert metrics["train/accuracy/classification"] == pytest.approx(0.8)


def test_local_metrics_csv_is_written(tmp_path: Path) -> None:
    logger = StubLogger(checkpoint_dir=str(tmp_path))
    meta = FakeMetaModel(
        epochs=2,
        train_metrics={
            "loss/classification": 0.4,
            "accuracy/classification": 0.6,
        },
        val_metrics_seq=[
            {
                "loss/classification": 0.3,
                "accuracy/classification": 0.7,
            },
            {
                "loss/classification": 0.2,
                "accuracy/classification": 0.75,
            },
        ],
    )
    train_model(
        meta_model=meta,
        evaluator=object(),
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        logger=logger,
    )

    csv_path = tmp_path / "model_checkpoints" / "fake-run" / "metrics.csv"
    assert csv_path.exists()
    frame = pd.read_csv(csv_path)
    assert len(frame) == 2
    assert "train/loss/total" in frame.columns
    assert "validation/accuracy/classification" in frame.columns
