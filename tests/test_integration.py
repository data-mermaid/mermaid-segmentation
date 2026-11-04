"""Lightweight integration tests for pipeline wiring.

Tests verify that components wire together correctly:
- Config loading → MetaModel creation
- Dataset batch shapes
- One-step train/validation without errors
- Fast: single steps, no full epochs
- Offline: no network, no real S3, no MLflow server

Marks: integration (can be deselected with '-m "not integration"')
"""

from __future__ import annotations

import copy

import pytest
import torch

from mermaidseg.io import ConfigDict
from mermaidseg.model.eval import EvaluatorSemanticSegmentation

from .conftest import IMAGE_SIZE, NUM_CLASSES

# Zero encoder features and a zero-weight head leave the logits equal to the head bias, so every
# pixel predicts class 2 with logits [0, 0, 5, 0].
_BIAS = [0.0, 0.0, 5.0, 0.0]


def _constant_prediction_setup(meta):
    """Pin the head to a constant prediction and build a batch whose masks are hand-countable.

    Each 56x56 mask has four 28x28 quadrants: class 2, class 1, class 3, and class 0 (ignored).
    """
    conv = meta.model.head.classifier
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.copy_(torch.tensor(_BIAS))
    masks = torch.zeros(2, *IMAGE_SIZE, dtype=torch.long)
    masks[:, :28, :28] = 2
    masks[:, :28, 28:] = 1
    masks[:, 28:, :28] = 3
    images = torch.randn(2, 3, *IMAGE_SIZE)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, masks), batch_size=2, num_workers=0
    )


# Of the non-ignored pixels one third are class 2, the predicted class.
EXPECTED_ACCURACY = 1 / 3
# Cross-entropy over the non-ignored pixels: one third are class 2 at ln(1 + 3e^-5) = 0.0200, and
# two thirds are class 1 or 3 at 5 + 0.0200.
EXPECTED_LOSS = 0.0200 + 2 / 3 * 5.0


@pytest.mark.integration
def test_metamodel_instantiates_from_config(minimal_config, make_meta_model):
    """MetaModel created from ConfigDict has correct attributes, optimizer, and loss."""
    assert minimal_config.model.name == "LinearDINOv3"
    meta = make_meta_model(minimal_config, run_name="test-run")
    assert meta.run_name == "test-run"
    assert meta.num_classes == NUM_CLASSES
    assert meta.model_name == "LinearDINOv3"
    assert meta.device == "cpu"
    assert isinstance(meta.optimizer, torch.optim.SGD)
    assert meta.optimizer.param_groups[0]["lr"] == 0.01
    assert meta.loss is not None


@pytest.mark.integration
def test_one_train_step_updates_model_weights(minimal_config, tiny_loader, make_meta_model):
    """After train step, model weights have changed."""
    meta = make_meta_model(minimal_config, run_name="test-weights")
    meta.model.train()

    images, masks = next(iter(tiny_loader))

    pre_weights = {}
    for name, param in meta.model.head.named_parameters():
        pre_weights[name] = param.clone()

    outputs = meta.model(images)
    loss, _ = meta.loss(outputs.logits, masks)
    loss.backward()
    meta.optimizer.step()
    meta.optimizer.zero_grad()

    changed = False
    for name, param in meta.model.head.named_parameters():
        if not torch.equal(param, pre_weights[name]):
            changed = True
            break
    assert changed, "Model weights did not change after train step"


@pytest.mark.integration
def test_one_val_step_produces_predictions(minimal_config, tiny_loader, make_meta_model):
    """Eval mode produces predictions without gradients."""
    meta = make_meta_model(minimal_config, run_name="test-val")
    meta.model.eval()

    images, masks = next(iter(tiny_loader))

    with torch.no_grad():
        outputs = meta.model(images)

    assert outputs.logits.shape[0] == images.shape[0]
    assert outputs.logits.shape[1] == NUM_CLASSES
    assert outputs.logits.shape[-2:] == masks.shape[-2:]


@pytest.mark.integration
def test_evaluator_returns_metrics(minimal_config, make_meta_model):
    """Accuracy ignores class 0 and counts the one predicted class among the rest."""
    meta = make_meta_model(minimal_config, run_name="test-eval")
    loader = _constant_prediction_setup(meta)
    evaluator = EvaluatorSemanticSegmentation(num_classes=NUM_CLASSES, device="cpu")

    metrics = evaluator.evaluate_model(loader, meta)

    assert set(metrics) == {"accuracy"}
    assert metrics["accuracy"] == pytest.approx(EXPECTED_ACCURACY, abs=1e-6)


class _RecordingLogger:
    """Captures (payload, step) pairs passed to Logger.log."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int]] = []

    def log(self, payload: dict[str, float], step: int) -> None:
        self.logged.append((dict(payload), step))


@pytest.mark.integration
def test_train_epoch_logs_per_batch_loss_on_global_step(
    minimal_config, tiny_loader, make_meta_model
):
    """train_epoch emits train/batch_loss against the monotonic global_step each batch."""
    cfg = copy.deepcopy(minimal_config)
    cfg.training = ConfigDict({**dict(cfg.training), "training_mode": "standard"})
    meta = make_meta_model(cfg, run_name="test-batch-logging")
    evaluator = EvaluatorSemanticSegmentation(num_classes=NUM_CLASSES, device="cpu")
    run_logger = _RecordingLogger()

    assert meta.global_step == 0
    # tiny_loader yields a single batch; iterations_per_train_epoch defaults to len(loader).
    meta.train_epoch(tiny_loader, evaluator, run_logger=run_logger, log_interval=1, epoch=0)
    assert meta.global_step == 1
    batch_logs = [(p, s) for p, s in run_logger.logged if "train/batch_loss" in p]
    assert batch_logs, "train/batch_loss was not logged per batch"
    assert batch_logs[0][1] == 1, "first batch should log at global_step == 1"

    # A second epoch continues advancing the same global-step axis.
    meta.train_epoch(tiny_loader, evaluator, run_logger=run_logger, log_interval=1, epoch=1)
    assert meta.global_step == 2
    steps = [s for p, s in run_logger.logged if "train/batch_loss" in p]
    assert steps == sorted(steps) and steps[-1] == 2


@pytest.mark.integration
def test_metamodel_train_epoch_returns_loss_and_metrics(minimal_config, make_meta_model):
    """train_epoch reports the pre-update loss and moves the head bias against the error."""
    meta = make_meta_model(minimal_config, run_name="test-epoch")
    loader = _constant_prediction_setup(meta)
    evaluator = EvaluatorSemanticSegmentation(num_classes=NUM_CLASSES, device="cpu")
    bias_before = meta.model.head.classifier.bias.detach().clone()

    loss, metrics, timing = meta.train_epoch(loader, evaluator)

    assert loss == pytest.approx(EXPECTED_LOSS, abs=1e-3)
    assert metrics["accuracy"] == pytest.approx(EXPECTED_ACCURACY, abs=1e-6)
    bias_after = meta.model.head.classifier.bias.detach()
    assert not torch.equal(bias_after, bias_before)
    # Class 2 is over-predicted and classes 1 and 3 are under-predicted, so SGD lowers bias[2]
    # and raises bias[1] and bias[3].
    assert bias_after[2] < bias_before[2]
    assert bias_after[1] > bias_before[1]
    assert bias_after[3] > bias_before[3]
    assert timing["num_samples"] == 2


@pytest.mark.integration
def test_metamodel_validation_epoch_returns_loss_and_metrics(minimal_config, make_meta_model):
    """validation_epoch returns the batch loss and accuracy without touching the weights."""
    meta = make_meta_model(minimal_config, run_name="test-val-epoch")
    loader = _constant_prediction_setup(meta)
    evaluator = EvaluatorSemanticSegmentation(num_classes=NUM_CLASSES, device="cpu")
    bias_before = meta.model.head.classifier.bias.detach().clone()

    val_loss, val_metrics = meta.validation_epoch(loader, evaluator)

    assert val_loss == pytest.approx(EXPECTED_LOSS, abs=1e-3)
    assert val_metrics["accuracy"] == pytest.approx(EXPECTED_ACCURACY, abs=1e-6)
    assert val_metrics["loss/classification"] == pytest.approx(EXPECTED_LOSS, abs=1e-3)
    assert torch.equal(meta.model.head.classifier.bias, bias_before)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_metamodel_runs_on_cuda_when_available(minimal_config, make_meta_model):
    """MetaModel can be instantiated on CUDA device."""
    meta = make_meta_model(minimal_config, run_name="test-cuda", device="cuda")
    assert meta.device == "cuda"
    assert next(meta.model.parameters()).device.type == "cuda"
