"""Round-trip tests for the checkpoint format module (slim / normalize / load /
resume)."""

from __future__ import annotations

import pytest
import torch

from mermaidseg.model.checkpoint import (
    frozen_param_names,
    load_into,
    normalize_state_dict_keys,
    restore_training_state,
    slim_state_dict,
)


class _Tiny(torch.nn.Module):
    """A frozen 'backbone' + a trainable 'head', mirroring the frozen-probe shape."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)
        for param in self.encoder.parameters():
            param.requires_grad_(False)


def test_frozen_param_names_lists_only_backbone():
    assert frozen_param_names(_Tiny()) == {"encoder.weight", "encoder.bias"}


def test_slim_drops_frozen_and_sets_flag():
    sd, excludes = slim_state_dict(_Tiny())
    assert excludes is True
    assert not any(k.startswith("encoder.") for k in sd)
    assert {"head.weight", "head.bias"} <= set(sd)


def test_slim_full_keeps_everything():
    sd, excludes = slim_state_dict(_Tiny(), save_full=True)
    assert excludes is False
    assert any(k.startswith("encoder.") for k in sd)


def test_slim_does_not_move_the_model():
    model = _Tiny()
    slim_state_dict(model)
    assert model.encoder.weight.device.type == "cpu"  # unchanged, no hidden .to() side effect


def test_normalize_reconciles_peft_nesting():
    out = normalize_state_dict_keys(
        {
            "encoder.base_model.model.model.q_proj.weight": 1,
            "encoder.model.cls_token": 2,
            "head.weight": 3,
        }
    )
    assert set(out) == {
        "encoder.base_model.model.q_proj.weight",  # collapsed one nesting level
        "encoder.cls_token",  # encoder.model. -> encoder.
        "head.weight",  # untouched
    }


def test_load_into_roundtrip_slim_restores_head_and_flags_no_unexpected():
    src = _Tiny()
    with torch.no_grad():
        src.head.weight.add_(1.0)  # make the head distinguishable
    sd, excludes = slim_state_dict(src)
    ckpt = {"model_state_dict": sd, "excludes_frozen_params": excludes}

    dst = _Tiny()  # fresh: frozen backbone re-initialised
    result = load_into(dst, ckpt)

    assert result.unexpected_keys == []  # every saved key resolved
    assert set(result.missing_keys) == frozen_param_names(dst)  # only frozen missing
    assert torch.equal(dst.head.weight, src.head.weight)  # head restored
    assert not dst.encoder.weight.requires_grad  # frozen state from construction, not load


def test_load_into_bare_state_dict_is_strict():
    result = load_into(_Tiny(), _Tiny().state_dict())
    assert result.missing_keys == [] and result.unexpected_keys == []


# --- restore_training_state (Managed-Spot resume) ---


def _trained_source() -> tuple[_Tiny, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """A _Tiny plus optimizer+scheduler taken a few steps, so their state is non-
    trivial."""
    model = _Tiny()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(3):
        optimizer.zero_grad()
        model.head(torch.ones(1, 4)).sum().backward()
        optimizer.step()
        scheduler.step()
    return model, optimizer, scheduler


def _full_checkpoint(model, optimizer, scheduler, epoch: int) -> dict:
    sd, excludes = slim_state_dict(model)
    ckpt = {
        "model_state_dict": sd,
        "excludes_frozen_params": excludes,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    return ckpt


def test_restore_training_state_restores_weights_optimizer_scheduler_and_epoch():
    src_model, src_opt, src_sched = _trained_source()
    checkpoint = _full_checkpoint(src_model, src_opt, src_sched, epoch=7)

    dst_model = _Tiny()
    dst_opt = torch.optim.SGD(
        [p for p in dst_model.parameters() if p.requires_grad], lr=0.1, momentum=0.9
    )
    dst_sched = torch.optim.lr_scheduler.StepLR(dst_opt, step_size=1, gamma=0.5)

    start_epoch = restore_training_state(dst_model, dst_opt, checkpoint, scheduler=dst_sched)

    assert start_epoch == 8  # resume FROM the epoch after the last completed one
    assert torch.equal(dst_model.head.weight, src_model.head.weight)  # weights restored
    # Scheduler position restored → same last_epoch and same current LR as the source.
    assert dst_sched.state_dict()["last_epoch"] == src_sched.state_dict()["last_epoch"] == 3
    assert dst_opt.param_groups[0]["lr"] == src_opt.param_groups[0]["lr"]
    # Optimizer momentum buffers restored (proves optimizer state, not just LR, was loaded).
    src_state = src_opt.state_dict()["state"]
    dst_state = dst_opt.state_dict()["state"]
    assert dst_state.keys() == src_state.keys() and src_state
    for key in src_state:
        assert torch.equal(dst_state[key]["momentum_buffer"], src_state[key]["momentum_buffer"])


def test_restore_training_state_without_scheduler_is_fine():
    src_model, src_opt, _ = _trained_source()
    checkpoint = _full_checkpoint(src_model, src_opt, scheduler=None, epoch=0)

    dst_model = _Tiny()
    dst_opt = torch.optim.SGD(
        [p for p in dst_model.parameters() if p.requires_grad], lr=0.1, momentum=0.9
    )

    start_epoch = restore_training_state(dst_model, dst_opt, checkpoint)  # scheduler defaults None

    assert start_epoch == 1
    assert torch.equal(dst_model.head.weight, src_model.head.weight)


def test_restore_training_state_requires_optimizer_state():
    """The lean best-model snapshot (no optimizer_state_dict) cannot resume — must fail
    loud."""
    model = _Tiny()
    sd, excludes = slim_state_dict(model)
    lean = {"model_state_dict": sd, "excludes_frozen_params": excludes, "epoch": 5}  # no optimizer
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)

    with pytest.raises(KeyError):
        restore_training_state(_Tiny(), opt, lean)
