"""MetaModel: run settings from the training config, optimizer wiring, gradient accumulation."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import mermaidseg.model.meta as meta_module
from mermaidseg.io import ConfigDict
from mermaidseg.model.meta import next_epoch_from_checkpoint, restore_warmup_progress
from tests.conftest import IMAGE_SIZE


@pytest.mark.parametrize(("raw", "expected"), [(180, 180.0), (45, 45.0), (0, 0.0), (-1, 0.0)])
def test_stall_dump_seconds_reads_training_config(raw, expected):
    assert meta_module.stall_dump_seconds(ConfigDict({"stall_dump_sec": raw})) == expected


def test_resolve_amp_dtype_rejects_a_missing_dtype():
    with pytest.raises(ValueError, match="mixed_precision_dtype"):
        meta_module._resolve_amp_dtype(None)


def test_meta_model_requires_max_grad_norm(minimal_config, make_meta_model):
    del minimal_config.training["max_grad_norm"]
    with pytest.raises(KeyError, match="max_grad_norm"):
        make_meta_model(minimal_config)


def test_optimizer_gives_lora_params_lora_lr_and_the_rest_dpt_lr(minimal_config, make_meta_model):
    minimal_config.training.optimizer.lora_lr = 1e-5
    minimal_config.training.optimizer.dpt_lr = 5e-5
    meta = make_meta_model(minimal_config)
    initial_lr = {
        id(param): group["initial_lr"]
        for group in meta.optimizer.param_groups
        for param in group["params"]
    }
    trainable = [(n, p) for n, p in meta.model.named_parameters() if p.requires_grad]
    lora = {n for n, _ in trainable if "lora_" in n}
    assert lora and len(lora) < len(trainable)
    for name, param in trainable:
        assert initial_lr[id(param)] == (1e-5 if name in lora else 5e-5), name


def _standard_cfg(minimal_config, **training):
    cfg = copy.deepcopy(minimal_config)
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


@pytest.mark.parametrize("name", ["float16", "fp16", "half"])
def test_resolve_amp_dtype_rejects_fp16_aliases(name):
    with pytest.raises(ValueError, match="mixed_precision_dtype"):
        meta_module._resolve_amp_dtype(name)


def test_resolve_amp_dtype_accepts_bfloat16():
    assert meta_module._resolve_amp_dtype("bfloat16") is torch.bfloat16


def _record_encoder_autocast(meta) -> dict[str, object]:
    """Record the CPU autocast state seen inside the mocked encoder's forward."""
    seen: dict[str, object] = {}

    def _record(_module, _args):
        seen["enabled"] = torch.is_autocast_enabled("cpu")
        seen["dtype"] = torch.get_autocast_dtype("cpu")

    encoder = next(m for m in meta.model.modules() if hasattr(m, "embed"))
    encoder.register_forward_pre_hook(_record)
    return seen


def _train_one_step(meta) -> None:
    images, labels = _images_labels(2)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    meta.train_epoch(loader, evaluator=None, epoch=0)


def test_cpu_train_step_runs_encoder_under_cpu_bf16_autocast(minimal_config, make_meta_model):
    cfg = _standard_cfg(
        minimal_config,
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",
        iterations_per_train_epoch=1,
    )
    meta = make_meta_model(cfg, run_name="cpu-amp")
    seen = _record_encoder_autocast(meta)
    _train_one_step(meta)
    assert seen == {"enabled": True, "dtype": torch.bfloat16}


def test_cpu_train_step_without_mixed_precision_does_not_autocast(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, mixed_precision=False, iterations_per_train_epoch=1)
    meta = make_meta_model(cfg, run_name="cpu-no-amp")
    seen = _record_encoder_autocast(meta)
    _train_one_step(meta)
    assert seen["enabled"] is False


def test_cuda_bf16_unsupported_raises_before_building_the_model(minimal_config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    cfg = _standard_cfg(minimal_config, mixed_precision=True, mixed_precision_dtype="bfloat16")
    with pytest.raises(RuntimeError, match="mixed_precision_dtype"):
        meta_module.MetaModel(
            run_name="no-bf16",
            num_classes=4,
            model_kwargs=cfg.model,
            training_kwargs=cfg.training,
            device=torch.device("cuda"),
        )


def test_cpu_bf16_autocast_keeps_logits_in_fp32(minimal_config, make_meta_model):
    cfg = _standard_cfg(minimal_config, mixed_precision=True, mixed_precision_dtype="bfloat16")
    meta = make_meta_model(cfg, run_name="fp32-logits")
    images, _ = _images_labels(2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = meta.model(images)
    assert out.logits.dtype == torch.float32
    assert out.concept_logits.dtype == torch.float32


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


def _capture_step_grads(meta) -> dict[str, torch.Tensor]:
    """Gradients each parameter holds when ``meta`` applies its optimizer step."""
    captured: dict[str, torch.Tensor] = {}
    real = meta._apply_accumulated_step

    def _capture() -> bool:
        for name, param in meta.model.named_parameters():
            if param.grad is not None:
                captured[name] = param.grad.detach().clone()
        return real()

    meta._apply_accumulated_step = _capture
    return captured


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
    # Dropout off, so both runs see the same forward.
    full.model.eval()
    micro.model.eval()
    before = {name: param.detach().clone() for name, param in full.model.named_parameters()}
    full_grads = _capture_step_grads(full)
    micro_grads = _capture_step_grads(micro)

    full.train_epoch(full_loader, evaluator=None, epoch=0)
    micro.train_epoch(micro_loader, evaluator=None, epoch=0)

    assert full_grads.keys() == micro_grads.keys()
    for name, grad in full_grads.items():
        assert torch.allclose(grad, micro_grads[name], atol=1e-5, rtol=1e-5), name
    assert any(grad.abs().sum() > 0 for grad in full_grads.values())
    assert any(not torch.equal(before[n], p) for n, p in full.model.named_parameters())


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


def test_meta_model_requires_training_kwargs(minimal_config):
    with pytest.raises(TypeError, match="training_kwargs"):
        meta_module.MetaModel(
            run_name="no-training", num_classes=4, model_kwargs=minimal_config.model, device="cpu"
        )
