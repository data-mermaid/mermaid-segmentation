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

import pytest
import torch

from mermaidseg.model.eval import EvaluatorSemanticSegmentation

from .conftest import IMAGE_SIZE, NUM_CLASSES


@pytest.mark.integration
def test_metamodel_instantiates_from_config(minimal_config, make_meta_model):
    """MetaModel created from ConfigDict has correct attributes, optimizer, and loss."""
    assert minimal_config.model.name == "LinearDINOv3"
    meta = make_meta_model(minimal_config, run_name="test-run")
    assert meta.run_name == "test-run"
    assert meta.num_classes == NUM_CLASSES
    assert meta.model_name == "LinearDINOv3"
    assert meta.device == "cpu"
    assert meta.optimizer is not None
    assert isinstance(meta.optimizer, torch.optim.Optimizer)
    assert meta.loss is not None


@pytest.mark.integration
def test_dataloader_yields_correct_batch(tiny_loader):
    """DataLoader produces a batch with correct shapes and dtypes."""
    images, masks = next(iter(tiny_loader))
    assert images.shape[0] <= 2
    assert images.shape[1:] == (3, *IMAGE_SIZE)
    assert masks.shape[1:] == IMAGE_SIZE
    assert images.dtype == torch.float32
    assert masks.dtype == torch.int64


@pytest.mark.integration
def test_one_train_step_forward_backward(minimal_config, tiny_loader, make_meta_model):
    """Single train step: forward, loss compute, backward, optimizer step."""
    meta = make_meta_model(minimal_config, run_name="test-step")
    meta.model.train()

    images, masks = next(iter(tiny_loader))

    outputs = meta.model(images)
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[-2:] == IMAGE_SIZE

    loss = meta.loss(outputs.logits, masks)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    loss.backward()
    meta.optimizer.step()
    meta.optimizer.zero_grad()


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
    loss = meta.loss(outputs.logits, masks)
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
def test_evaluator_returns_metrics(minimal_config, tiny_loader, make_meta_model):
    """Evaluator produces metric dict with expected keys."""
    meta = make_meta_model(minimal_config, run_name="test-eval")
    evaluator = EvaluatorSemanticSegmentation(num_classes=NUM_CLASSES, device="cpu")

    metrics = evaluator.evaluate_model(tiny_loader, meta)

    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    assert "accuracy" in metrics or "miou" in metrics


@pytest.mark.integration
def test_metamodel_train_epoch_returns_loss_and_metrics(minimal_config, tiny_loader, make_meta_model):
    """MetaModel.train_epoch returns loss and metric dict."""
    meta = make_meta_model(minimal_config, run_name="test-epoch")
    evaluator = EvaluatorSemanticSegmentation(num_classes=NUM_CLASSES, device="cpu")

    loss, metrics, timing = meta.train_epoch(tiny_loader, evaluator)

    assert isinstance(loss, float)
    assert loss >= 0
    assert isinstance(metrics, dict)
    assert isinstance(timing, dict)
    assert "data_loading_sec" in timing
    assert "forward_sec" in timing
    assert "backward_sec" in timing
    assert "num_samples" in timing


@pytest.mark.integration
def test_metamodel_validation_epoch_returns_loss_and_metrics(minimal_config, tiny_loader, make_meta_model):
    """MetaModel.validation_epoch returns loss and metric dict in eval mode."""
    meta = make_meta_model(minimal_config, run_name="test-val-epoch")
    evaluator = EvaluatorSemanticSegmentation(num_classes=NUM_CLASSES, device="cpu")

    val_loss, val_metrics = meta.validation_epoch(tiny_loader, evaluator)

    assert isinstance(val_loss, float)
    assert isinstance(val_metrics, dict)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_metamodel_runs_on_cuda_when_available(minimal_config, make_meta_model):
    """MetaModel can be instantiated on CUDA device."""
    meta = make_meta_model(minimal_config, run_name="test-cuda", device="cuda")
    assert meta.device == "cuda"
    assert next(meta.model.parameters()).device.type == "cuda"
