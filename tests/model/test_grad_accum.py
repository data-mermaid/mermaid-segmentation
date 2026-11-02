"""Gradient accumulation counts optimizer steps and averages the loss in fp32."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mermaidseg.model.meta import next_epoch_from_checkpoint, restore_warmup_progress
from tests.conftest import IMAGE_SIZE


def _standard_cfg(minimal_config, **training):
    cfg = copy.deepcopy(minimal_config)
    cfg.training.training_mode = "standard"
    cfg.training.loss = {
        "type": "CrossEntropyLoss",
        "ignore_index": 0,
        "per_pixel_loss_weight": 0.0,
        "per_image_loss_weight": 1.0,
    }
    for key, value in training.items():
        cfg.training[key] = value
    return cfg


def _images_labels(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(n, 3, *IMAGE_SIZE, generator=generator)
    labels = torch.ones(n, *IMAGE_SIZE, dtype=torch.long)
    return images, labels


def test_next_epoch_from_checkpoint_keeps_old_epoch_boundaries():
    assert next_epoch_from_checkpoint({"epoch": 3}) == 4
    assert next_epoch_from_checkpoint({"epoch": 0, "epoch_complete": True}) == 1
    assert next_epoch_from_checkpoint({"epoch": 0, "epoch_complete": False}) == 0
    assert next_epoch_from_checkpoint({}) == 0


def test_restore_warmup_marks_old_checkpoints_complete():
    class _Meta:
        warmup_iters = 4000
        _warmup_iters_completed = 0

    meta = _Meta()
    restore_warmup_progress(meta, {})
    assert meta._warmup_iters_completed == 4000
    restore_warmup_progress(meta, {"warmup_iters_completed": 10})
    assert meta._warmup_iters_completed == 10


def test_bf16_does_not_enable_grad_scaler(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, mixed_precision=True, mixed_precision_dtype="bfloat16")
    meta = make_meta_model(cfg, run_name="bf16-scaler")
    assert meta.amp_dtype == torch.bfloat16
    assert meta.scaler.is_enabled() is False
    # The constructor gate, isolated from the CPU "no CUDA" short-circuit.
    enabled = torch.amp.GradScaler(enabled=True and meta.amp_dtype == torch.float16)
    assert enabled.is_enabled() is False


def test_accumulate_loss_averages_in_fp32(minimal_config, make_meta_model):
    meta = make_meta_model(_standard_cfg(minimal_config), run_name="scale")
    meta.grad_accum_steps = 2
    param = torch.nn.Parameter(torch.tensor(1.0))
    loss = param * 4
    meta._accumulate_loss(loss)
    assert param.grad is not None
    assert param.grad.dtype == torch.float32
    assert param.grad.item() == pytest.approx(2.0)


def test_two_microbatches_are_one_optimizer_step(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=2)
    meta = make_meta_model(cfg, run_name="accum-count")
    images, labels = _images_labels(4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    calls = {"n": 0}
    real = meta.batch_predict_loss

    def _count(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    meta.batch_predict_loss = _count
    with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
        meta.train_epoch(loader, evaluator=None, epoch=0)
    assert calls["n"] == 2
    assert meta.global_step == 1
    assert clip.call_count == 1


def test_microbatches_match_a_single_batch(minimal_config, make_meta_model):
    """Per-image mean of two equal microbatches matches one combined batch."""
    images, labels = _images_labels(4)
    full_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    micro_loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    full = make_meta_model(
        _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=1),
        run_name="full-batch",
    )
    micro = make_meta_model(
        _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=2),
        run_name="micro-batch",
    )
    micro.model.load_state_dict(full.model.state_dict())
    before = {name: tensor.detach().clone() for name, tensor in full.model.named_parameters()}

    full.train_epoch(full_loader, evaluator=None, epoch=0)
    micro.train_epoch(micro_loader, evaluator=None, epoch=0)

    moved = False
    for name, start in before.items():
        full_param = dict(full.model.named_parameters())[name]
        micro_param = dict(micro.model.named_parameters())[name]
        assert torch.allclose(full_param, micro_param, atol=1e-5, rtol=1e-5)
        moved = moved or not torch.allclose(full_param, start)
    assert moved, "the optimizer step did not change any parameter"


def test_nonfinite_microbatch_drops_the_window(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=1, grad_accum_steps=2)
    meta = make_meta_model(cfg, run_name="nonfinite")
    before = {name: tensor.detach().clone() for name, tensor in meta.model.state_dict().items()}
    calls = {"n": 0}
    real = meta.batch_predict_loss

    def _poison(*args, **kwargs):
        calls["n"] += 1
        loss, outputs, concepts, components = real(*args, **kwargs)
        if calls["n"] == 2:
            loss = torch.tensor(float("nan"))
        return loss, outputs, concepts, components

    meta.batch_predict_loss = _poison
    images, labels = _images_labels(4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    meta.train_epoch(loader, evaluator=None, epoch=0)

    assert meta.global_step == 0
    after = meta.model.state_dict()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor)


def test_resume_continues_the_same_epoch(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, iterations_per_train_epoch=4, grad_accum_steps=1)
    meta = make_meta_model(cfg, run_name="resume-epoch")
    meta.global_step = 2
    images, labels = _images_labels(8)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    calls = {"n": 0}
    real = meta.batch_predict_loss

    def _count(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    meta.batch_predict_loss = _count
    meta.train_epoch(loader, evaluator=None, epoch=0)
    assert calls["n"] == 2
    assert meta.global_step == 4
