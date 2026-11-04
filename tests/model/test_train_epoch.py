"""MetaModel.train_epoch: resume offset, skipped updates, and the returned average loss."""

from __future__ import annotations

import copy
import math

import pytest
import torch

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


def test_empty_batch_skips_its_update(minimal_config, make_meta_model):
    meta = make_meta_model(_cfg(minimal_config, iterations=2), run_name="empty-batch")
    _, _, timing = meta.train_epoch([_empty_batch(), _batch(1)], evaluator=None, epoch=0)
    assert meta.global_step == 1
    assert timing["num_samples"] == 2


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
