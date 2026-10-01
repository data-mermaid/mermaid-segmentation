"""LR schedule wiring for the duration ablation.

Constant peak LR after linear warmup, and a per-iteration linear cooldown to 0.
"""

from __future__ import annotations

import copy

import pytest

from mermaidseg.model.train import train_model


def _with_scheduler(minimal_config, **scheduler):
    cfg = copy.deepcopy(minimal_config)
    cfg.training.training_mode = "standard"
    cfg.training.epochs = 1
    cfg.training.scheduler = scheduler
    return cfg


@pytest.mark.integration
def test_iteration_linear_cooldown_reaches_zero_without_extra_epoch_step(
    minimal_config, tiny_loader, make_meta_model
):
    """25 iteration steps land on LR 0, and the epoch-end step does not fire."""
    total = 25
    cfg = _with_scheduler(
        minimal_config,
        type="LinearLR",
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total,
        step_every="iteration",
        warmup_iters=0,
    )
    cfg.training.iterations_per_train_epoch = total
    meta = make_meta_model(cfg, run_name="cooldown")
    assert meta.scheduler_step_every == "iteration"
    # Constructor leaves the LR at the base (start_factor=1).
    base_lr = 0.01
    assert meta.optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

    train_model(
        meta,
        evaluator=None,
        train_loader=tiny_loader,
        start_epoch=0,
        end_epoch=1,
        max_load_failure_rate=None,
    )

    assert meta.scheduler.last_epoch == total
    assert meta.optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.integration
def test_constant_lr_stays_at_base_after_warmup(minimal_config, tiny_loader, make_meta_model):
    """Warmup returns to the configured base LR, and ConstantLR factor 1 holds it."""
    base_lr = 0.02
    warmup_iters = 4
    cfg = copy.deepcopy(minimal_config)
    cfg.training.training_mode = "standard"
    cfg.training.epochs = 1
    cfg.training.iterations_per_train_epoch = warmup_iters
    cfg.training.optimizer.lr = base_lr
    cfg.training.scheduler = {
        "type": "ConstantLR",
        "factor": 1.0,
        "total_iters": 1000,
        "warmup_iters": warmup_iters,
        "warmup_start_factor": 0.01,
    }
    meta = make_meta_model(cfg, run_name="constant")
    assert meta.scheduler_step_every == "epoch"
    # Warmup's initial step drops the LR; training must bring it back to base
    # and the epoch-level ConstantLR step must not move it.
    assert meta.optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * 0.01)

    train_model(
        meta,
        evaluator=None,
        train_loader=tiny_loader,
        start_epoch=0,
        end_epoch=1,
        max_load_failure_rate=None,
    )

    assert meta._warmup_iters_completed == warmup_iters
    assert meta.optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)
    # One epoch-end step on top of the constructor's initial step.
    assert meta.scheduler.last_epoch == 1


@pytest.mark.integration
def test_rebuild_scheduler_binds_to_restored_lr(minimal_config, make_meta_model):
    """Cooldown rebuild uses the loaded group LR, not the source schedule's initial_lr."""
    cfg = _with_scheduler(
        minimal_config,
        type="LinearLR",
        start_factor=1.0,
        end_factor=0.0,
        total_iters=10,
        step_every="iteration",
        warmup_iters=0,
    )
    meta = make_meta_model(cfg, run_name="rebuild")
    restored = 0.02
    for group in meta.optimizer.param_groups:
        group["lr"] = restored
        group["initial_lr"] = 0.01
    meta.rebuild_scheduler()
    assert meta.scheduler.base_lrs[0] == pytest.approx(restored)
    assert meta.optimizer.param_groups[0]["lr"] == pytest.approx(restored)
    assert meta.scheduler.last_epoch == 0
    assert meta.scheduler_step_every == "iteration"


@pytest.mark.integration
def test_step_every_rejects_unknown_value(minimal_config, make_meta_model):
    cfg = _with_scheduler(
        minimal_config,
        type="ConstantLR",
        factor=1.0,
        total_iters=1,
        step_every="batch",
    )
    with pytest.raises(ValueError, match="step_every"):
        make_meta_model(cfg, run_name="bad-step")
