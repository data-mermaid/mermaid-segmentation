"""Characterization tests for MetaModel's standard-mode train/val loops.

Locks the current observable behavior of ``MetaModel.train_epoch`` /
``validation_epoch`` / ``batch_predict_loss`` before the planned Trainer/ModelWrapper
decomposition, so that refactor can be shown behavior-preserving. Runs fully offline on
CPU with a tiny stub model (no HF download, no S3, no CUDA).
"""

from __future__ import annotations

import types

import torch
from torch import nn

import mermaidseg.model.models as models_mod
from mermaidseg.io import ConfigDict
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel

NUM_CLASSES = 3


class _TinyLogitsModel(nn.Module):
    """Standard-mode output contract: forward(x) -> obj with ``.logits`` (B, C, H,
    W)."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_ch: int = 3):
        super().__init__()
        self.head = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        return types.SimpleNamespace(logits=self.head(x))


def _make_meta(monkeypatch, *, iters_train=2, iters_val=2) -> MetaModel:
    monkeypatch.setattr(models_mod, "TinyLogitsModel", _TinyLogitsModel, raising=False)
    training_kwargs = ConfigDict(
        {
            "training_mode": "standard",
            "iterations_per_train_epoch": iters_train,
            "iterations_per_val_epoch": iters_val,
            "loss": {"type": "CrossEntropyLoss", "ignore_index": 0},
            "optimizer": {"type": "AdamW", "lr": 1e-2},
        }
    )
    return MetaModel(
        run_name="char-test",
        num_classes=NUM_CLASSES,
        model_kwargs=ConfigDict({"name": "TinyLogitsModel"}),
        device="cpu",
        training_kwargs=training_kwargs,
    )


def _loader(*, n_batches=2, bs=2, size=8, empty_at=None):
    batches = []
    for i in range(n_batches):
        if empty_at is not None and i == empty_at:
            batches.append((torch.tensor([]), torch.tensor([])))
            continue
        img = torch.randn(bs, 3, size, size)
        lbl = torch.randint(0, NUM_CLASSES, (bs, size, size))
        batches.append((img, lbl))
    return batches


def _golden_loader(*, n_batches=2, bs=2, size=8):
    """Fully deterministic loader (own seed) so loss/metric values are reproducible
    regardless of any RNG consumed during model construction."""
    torch.manual_seed(123)
    return [
        (torch.randn(bs, 3, size, size), torch.randint(0, NUM_CLASSES, (bs, size, size)))
        for _ in range(n_batches)
    ]


# Golden values captured from the pre-Seam-3 train_epoch/validation_epoch (seed 0 for model
# init, seed 123 for data). They lock the mode-independent loop scaffolding — loader
# iteration, loss averaging, metric compute/reset, and the train-vs-val differences — across
# the epoch-unification refactor. If these change, the refactor altered observable behavior.
_GOLDEN = {
    "train_loss": 1.3080165386,
    "val_loss": 1.3142812848,
    "accuracy": 0.2195121944,
    "miou": 0.1632783860,
    "num_samples": 4,
}
_TOL = 1e-6


def test_train_epoch_returns_loss_metrics_timing_and_steps_optimizer(monkeypatch):
    torch.manual_seed(0)
    meta = _make_meta(monkeypatch, iters_train=2)
    evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")
    before = meta.model.head.weight.detach().clone()

    loss, metrics, timing = meta.train_epoch(_loader(n_batches=2, bs=2), evaluator)

    assert isinstance(loss, float) and torch.isfinite(torch.tensor(loss)) and loss > 0
    # metric bank produces scalar accuracy + miou, plus averaged loss components.
    assert "accuracy" in metrics and "miou" in metrics
    assert any(k.startswith("loss/") for k in metrics)
    # timing reports successfully-processed samples = iterations * batch_size.
    assert timing["num_samples"] == 2 * 2
    # optimizer actually stepped (weights moved).
    assert not torch.allclose(before, meta.model.head.weight)


def test_validation_epoch_computes_metrics_without_updating_weights(monkeypatch):
    torch.manual_seed(0)
    meta = _make_meta(monkeypatch, iters_val=2)
    evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")
    before = meta.model.head.weight.detach().clone()

    loss, metrics = meta.validation_epoch(_loader(n_batches=2, bs=2), evaluator)

    assert isinstance(loss, float) and loss > 0
    assert "accuracy" in metrics and "miou" in metrics
    # validation must not mutate weights (no optimizer step, no_grad).
    assert torch.allclose(before, meta.model.head.weight)


def test_train_epoch_skips_empty_batches(monkeypatch):
    """An empty (all-failed-load) batch is skipped, not counted in num_samples."""
    torch.manual_seed(0)
    meta = _make_meta(monkeypatch, iters_train=2)
    evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")

    # 2 iterations over [empty, full] -> only the full batch (bs=2) contributes.
    _, _, timing = meta.train_epoch(_loader(n_batches=2, bs=2, empty_at=0), evaluator)

    assert timing["num_samples"] == 2


def test_train_epoch_golden_values(monkeypatch):
    """Lock exact train loss / metrics / num_samples across the epoch-unification
    refactor."""
    torch.manual_seed(0)
    meta = _make_meta(monkeypatch, iters_train=2)
    evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")

    loss, metrics, timing = meta.train_epoch(_golden_loader(n_batches=2, bs=2), evaluator)

    assert abs(loss - _GOLDEN["train_loss"]) < _TOL, loss
    assert abs(metrics["accuracy"] - _GOLDEN["accuracy"]) < _TOL, metrics["accuracy"]
    assert abs(metrics["miou"] - _GOLDEN["miou"]) < _TOL, metrics["miou"]
    assert abs(metrics["loss/classification"] - _GOLDEN["train_loss"]) < _TOL
    assert timing["num_samples"] == _GOLDEN["num_samples"]


def test_validation_epoch_golden_values(monkeypatch):
    """Lock exact val loss / metrics across the epoch-unification refactor."""
    torch.manual_seed(0)
    meta = _make_meta(monkeypatch, iters_val=2)
    evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")

    loss, metrics = meta.validation_epoch(_golden_loader(n_batches=2, bs=2), evaluator)

    assert abs(loss - _GOLDEN["val_loss"]) < _TOL, loss
    assert abs(metrics["accuracy"] - _GOLDEN["accuracy"]) < _TOL, metrics["accuracy"]
    assert abs(metrics["miou"] - _GOLDEN["miou"]) < _TOL, metrics["miou"]


def test_train_epoch_skips_nonfinite_loss_batch(monkeypatch):
    """A non-finite (NaN) loss batch is skipped: no optimizer step, not counted in
    num_samples, and excluded from metric accumulation (train-only skip path)."""
    torch.manual_seed(0)
    meta = _make_meta(monkeypatch, iters_train=2)
    evaluator = Evaluator(num_classes=NUM_CLASSES, device="cpu")

    real_loss = meta.loss

    class _NaNFirstLoss:
        """Wraps the real loss but poisons the first batch with NaN (still graph-
        connected)."""

        def __init__(self):
            self.calls = 0

        def __call__(self, outputs, targets):
            self.calls += 1
            loss, comps = real_loss(outputs, targets)
            return (loss * float("nan"), comps) if self.calls == 1 else (loss, comps)

    meta.loss = _NaNFirstLoss()
    before = meta.model.head.weight.detach().clone()

    _, _, timing = meta.train_epoch(_golden_loader(n_batches=2, bs=2), evaluator)

    # Only the second (finite) batch contributes to num_samples.
    assert timing["num_samples"] == 2
    # Weights still moved (the finite batch stepped the optimizer).
    assert not torch.allclose(before, meta.model.head.weight)
