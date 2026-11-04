"""MetaModel.train_epoch: resume offset, skipped updates, and the returned average loss."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from mermaidseg.model.meta import MAX_CONSECUTIVE_EMPTY_BATCHES
from tests.conftest import IMAGE_SIZE

# Zero encoder features and a zero-weight head leave the logits equal to this bias everywhere.
_BIAS = [0.0, 0.0, 5.0, 0.0]
# Cross-entropy of a pixel labelled with the predicted class 2, and with any other class.
CE_PREDICTED = math.log(1 + 3 * math.exp(-5))
CE_OTHER = 5 + math.log(1 + 3 * math.exp(-5))


def _cfg(minimal_config, *, iterations: int, lr: float = 0.01):
    cfg = copy.deepcopy(minimal_config)
    cfg.training.iterations_per_train_epoch = iterations
    cfg.training.optimizer = {"type": "SGD", "lr": lr}
    return cfg


def _pin_head(meta) -> None:
    conv = meta.model.head.classifier
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.copy_(torch.tensor(_BIAS))


def _batch(label: int) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(2, 3, *IMAGE_SIZE)
    return images, torch.full((2, *IMAGE_SIZE), label, dtype=torch.long)


def _empty_batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(0, 3, *IMAGE_SIZE), torch.empty(0, *IMAGE_SIZE, dtype=torch.long)


@pytest.mark.parametrize(
    ("global_step", "epoch", "updates_run"),
    [
        (5, 0, 2),  # offset 5 is past the 2 updates of epoch 0: restart the epoch at 0
        (0, 1, 2),  # offset -2 is before epoch 1: restart the epoch at 0
        (2, 0, 0),  # offset 2 == iterations: epoch 0 already finished, nothing to run
    ],
)
def test_resume_offset_outside_the_epoch_restarts_it(
    minimal_config, make_meta_model, global_step, epoch, updates_run
):
    meta = make_meta_model(_cfg(minimal_config, iterations=2), run_name="resume-offset")
    meta.global_step = global_step
    meta.train_epoch([_batch(1)], evaluator=None, epoch=epoch)
    assert meta.global_step == global_step + updates_run


def _count_optimizer_steps(meta) -> list[int]:
    steps: list[int] = []
    meta.optimizer.register_step_post_hook(lambda *_: steps.append(1))
    return steps


def _inf_grad_on_calls(meta, calls: set[int]) -> None:
    """Make the head bias gradient infinite on the given backward calls (0-based)."""
    seen = {"n": 0}

    def _hook(grad):
        index = seen["n"]
        seen["n"] += 1
        return torch.full_like(grad, math.inf) if index in calls else grad

    meta.model.head.classifier.bias.register_hook(_hook)


def test_empty_batch_is_replaced_by_the_next_batch(minimal_config, make_meta_model):
    meta = make_meta_model(_cfg(minimal_config, iterations=3, lr=0.0), run_name="empty-batch")
    _pin_head(meta)
    steps = _count_optimizer_steps(meta)
    loader = [_batch(2), _empty_batch(), _batch(1), _batch(1)]

    loss, _, timing = meta.train_epoch(loader, evaluator=None, epoch=0)

    assert meta.global_step == 3
    assert len(steps) == 3
    assert loss == pytest.approx((CE_PREDICTED + 2 * CE_OTHER) / 3, abs=1e-5)
    assert timing["skipped_updates"] == 0
    assert timing["num_samples"] == 6


def test_too_many_consecutive_empty_batches_raise(minimal_config, make_meta_model):
    meta = make_meta_model(_cfg(minimal_config, iterations=1), run_name="empty-cap")
    loader = [_empty_batch()] * MAX_CONSECUTIVE_EMPTY_BATCHES + [_batch(1)]
    with pytest.raises(
        RuntimeError, match=rf"{MAX_CONSECUTIVE_EMPTY_BATCHES} consecutive.*every item failed"
    ):
        meta.train_epoch(loader, evaluator=None, epoch=0)


def test_empty_batches_below_the_cap_do_not_raise(minimal_config, make_meta_model):
    meta = make_meta_model(_cfg(minimal_config, iterations=1), run_name="empty-below-cap")
    loader = [_empty_batch()] * (MAX_CONSECUTIVE_EMPTY_BATCHES - 1) + [_batch(1)]
    _, _, timing = meta.train_epoch(loader, evaluator=None, epoch=0)
    assert meta.global_step == 1
    assert timing["num_samples"] == 2


def test_empty_batch_inside_an_accumulation_window(minimal_config, make_meta_model):
    cfg = _cfg(minimal_config, iterations=1, lr=0.0)
    cfg.training.grad_accum_steps = 2
    meta = make_meta_model(cfg, run_name="empty-accum")
    _pin_head(meta)
    loader = [_batch(2), _empty_batch(), _batch(1)]

    loss, _, timing = meta.train_epoch(loader, evaluator=None, epoch=0)

    assert meta.global_step == 1
    assert timing["num_samples"] == 4
    assert loss == pytest.approx((CE_PREDICTED + CE_OTHER) / 2, abs=1e-5)


def test_non_finite_grad_norm_skips_the_step(minimal_config, make_meta_model):
    meta = make_meta_model(_cfg(minimal_config, iterations=1), run_name="inf-grad")
    _pin_head(meta)
    # The forward loss stays finite; only the gradient reaching the optimizer is infinite.
    meta.model.head.classifier.bias.register_hook(lambda grad: torch.full_like(grad, math.inf))
    before = {name: p.detach().clone() for name, p in meta.model.head.named_parameters()}

    meta.train_epoch([_batch(1)], evaluator=None, epoch=0)

    assert meta.global_step == 0
    for name, param in meta.model.head.named_parameters():
        assert torch.equal(param, before[name]), name


def test_returned_loss_is_the_mean_over_the_updates_run(minimal_config, make_meta_model):
    # lr=0 keeps the head fixed, so each update's loss is known in advance. Resuming at offset 1
    # of 3 runs two updates: one batch labelled with the predicted class, one with another class.
    meta = make_meta_model(_cfg(minimal_config, iterations=3, lr=0.0), run_name="avg-loss")
    _pin_head(meta)
    meta.global_step = 1

    loss, _, _ = meta.train_epoch([_batch(2), _batch(1)], evaluator=None, epoch=0)

    assert meta.global_step == 3
    assert loss == pytest.approx((CE_PREDICTED + CE_OTHER) / 2, abs=1e-5)


def test_skipped_update_is_left_out_of_the_average(minimal_config, make_meta_model):
    meta = make_meta_model(_cfg(minimal_config, iterations=2, lr=0.0), run_name="skip-avg")
    _pin_head(meta)
    _inf_grad_on_calls(meta, {1})

    loss, _, timing = meta.train_epoch([_batch(2), _batch(1)], evaluator=None, epoch=0)

    assert meta.global_step == 1
    assert timing["skipped_updates"] == 1
    assert loss == pytest.approx(CE_PREDICTED, abs=1e-5)


def test_average_loss_is_nan_when_every_update_is_skipped(minimal_config, make_meta_model):
    meta = make_meta_model(_cfg(minimal_config, iterations=2), run_name="all-skipped")
    _inf_grad_on_calls(meta, {0, 1})

    loss, _, timing = meta.train_epoch([_batch(1)], evaluator=None, epoch=0)

    assert math.isnan(loss)
    assert timing["skipped_updates"] == 2
    assert meta.global_step == 0
