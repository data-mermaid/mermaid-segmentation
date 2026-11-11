"""Checks for the 4-GPU shard reduction.

The 768px job averages four contiguous shards of 4, twice, in place of eight
serial microbatches of 4. These tests lock that reduction, the per-shard label
lookup, the persistent replicas' sync and gradient reduction (on CPU copies),
and the launch guard. The last test runs only with 4 visible GPUs.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.dataset_reconciliation.label_mapping import source_labels_to_target_labels
from mermaidseg.io import ConfigDict
from mermaidseg.model.data_parallel import (
    ShardReplicas,
    ShardStep,
    apply_concept_bottleneck_loss,
    build_shard_replicas,
    contiguous_shard_bounds,
    mean_shard_losses,
    parallel_forward_loss,
    require_data_parallel_gpus,
    require_divisible_batch,
    shard_targets,
)
from mermaidseg.model.meta import records_train_metrics

_NUM_SOURCE = 9
_NUM_CONCEPTS = 5
_NUM_CLASSES = 4


def _lookups(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.randint(0, _NUM_CLASSES, (_NUM_SOURCE + 1,), generator=generator)
    target[0] = 0
    concepts = torch.rand(_NUM_SOURCE + 1, _NUM_CONCEPTS, generator=generator)
    concepts[0] = 0
    return target, concepts


class _TinyCBM(nn.Module):
    """Stand-in for the concept-bottleneck model: a frozen stem and two trainable heads."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 1)
        self.concept_head = nn.Conv2d(6, _NUM_CONCEPTS, 1)
        self.classifier = nn.Conv2d(_NUM_CONCEPTS, _NUM_CLASSES, 1)
        self.register_buffer("scale", torch.ones(1))
        self._encoder_frozen = False
        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        concept_logits = self.concept_head(torch.relu(self.stem(x))) * self.scale
        concept_outputs = torch.sigmoid(concept_logits)
        return SimpleNamespace(
            logits=self.classifier(concept_outputs),
            concept_logits=concept_logits,
            hidden_states=concept_outputs,
        )


class _TinyLoss(nn.Module):
    def forward(self, outputs, target_labels, concept_logits, target_concepts):
        class_loss = nn.functional.cross_entropy(outputs, target_labels)
        concept_loss = nn.functional.binary_cross_entropy_with_logits(
            concept_logits, target_concepts
        )
        loss = class_loss + concept_loss
        return loss, {"concepts": float(concept_loss.detach())}


def _cpu_replicas(n: int = 4) -> ShardReplicas:
    target, concepts = _lookups()
    step = ShardStep(_TinyCBM(), _TinyLoss(), None, None, target, concepts, False, torch.bfloat16)
    return ShardReplicas(step, ["cpu"] * n)


def _batch(batch_size: int = 12, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(batch_size, 3, 8, 8, generator=generator)
    source_labels = torch.randint(0, _NUM_SOURCE + 1, (batch_size, 8, 8), generator=generator)
    return images, source_labels


def test_data_parallel_rejects_other_gpu_counts():
    require_data_parallel_gpus(4)
    for device_count in (0, 1, 3, 8):
        with pytest.raises(RuntimeError, match="4 visible GPUs"):
            require_data_parallel_gpus(device_count)


def test_batch_must_split_evenly_across_four_gpus():
    assert require_divisible_batch(16) == 4
    assert require_divisible_batch(8) == 2
    assert contiguous_shard_bounds(16) == [(0, 4), (4, 8), (8, 12), (12, 16)]
    with pytest.raises(RuntimeError, match="not divisible"):
        require_divisible_batch(15)
    with pytest.raises(RuntimeError, match="not divisible"):
        contiguous_shard_bounds(2)


def test_mean_shard_losses_averages_values_and_grads():
    losses = [torch.tensor(value, requires_grad=True) for value in (1.0, 3.0, 5.0, 7.0)]
    mean = mean_shard_losses(losses)
    assert mean.item() == pytest.approx(4.0)
    mean.backward()
    for loss in losses:
        assert loss.grad is not None
        assert loss.grad.item() == pytest.approx(0.25)


def test_two_minibatches_of_four_shards_match_eight_serial_microbatches():
    """Minibatch 16 accumulated twice matches microbatch 4 accumulated 8 times."""
    values = torch.arange(32, dtype=torch.float32)

    serial = torch.nn.Parameter(torch.tensor(1.0))
    for start in range(0, 32, 4):
        batch = values[start : start + 4]
        ((serial * batch).mean() / 8).backward()

    parallel = torch.nn.Parameter(torch.tensor(1.0))
    for start in (0, 16):
        minibatch = values[start : start + 16]
        shard_losses = [
            (parallel * minibatch[shard_start:shard_end]).mean()
            for shard_start, shard_end in contiguous_shard_bounds(16)
        ]
        (mean_shard_losses(shard_losses) / 2).backward()

    assert serial.grad is not None
    assert parallel.grad is not None
    assert parallel.grad.item() == pytest.approx(serial.grad.item())


def test_data_parallel_init_rejects_fewer_than_four_gpus(minimal_config, make_meta_model):
    if torch.cuda.is_available() and torch.cuda.device_count() == 4:
        pytest.skip("this machine has 4 visible GPUs")
    cfg = copy.deepcopy(minimal_config)
    cfg.training.training_mode = "standard"
    cfg.training.data_parallel = True
    with pytest.raises(RuntimeError, match="4 visible GPUs"):
        make_meta_model(cfg)


def test_shard_targets_match_the_whole_batch_lookup():
    target, concepts = _lookups()
    _, source_labels = _batch()
    whole_labels = source_labels_to_target_labels(source_labels, target)
    whole_concepts = source_labels_to_concepts(source_labels, concepts)
    for start, end in contiguous_shard_bounds(12):
        labels, shard_concepts = shard_targets(source_labels[start:end], target, concepts)
        assert torch.equal(labels, whole_labels[start:end])
        assert torch.equal(shard_concepts, whole_concepts[start:end])

    identity_labels, _ = shard_targets(source_labels, None, concepts)
    assert torch.equal(identity_labels, source_labels)


def test_shard_step_matches_the_single_gpu_loss():
    replicas = _cpu_replicas(n=1)
    step = replicas.modules[0]
    images, source_labels = _batch()
    loss, outputs, concept_outputs, labels, concepts, _ = step(images, source_labels)

    target, concept_lookup = _lookups()
    expected_labels = source_labels_to_target_labels(source_labels, target)
    expected_concepts = source_labels_to_concepts(source_labels, concept_lookup)
    expected_loss, expected_outputs, expected_concept_outputs, _ = apply_concept_bottleneck_loss(
        step.model(images),
        step.loss_fn,
        None,
        None,
        expected_labels,
        expected_concepts,
    )
    assert torch.equal(loss, expected_loss)
    assert torch.equal(outputs, expected_outputs)
    assert torch.equal(concept_outputs, expected_concept_outputs)
    assert torch.equal(labels, expected_labels)
    assert torch.equal(concepts, expected_concepts)

    no_maps = step(images, source_labels, return_maps=False)
    assert torch.equal(no_maps[0], expected_loss)
    assert no_maps[1:5] == (None, None, None, None)


def _assert_replicas_match_primary(replicas: ShardReplicas) -> None:
    primary = dict(replicas.modules[0].state_dict(keep_vars=True))
    primary.update(dict(replicas.modules[0].named_buffers()))
    for replica in replicas.modules[1:]:
        state = dict(replica.state_dict(keep_vars=True))
        state.update(dict(replica.named_buffers()))
        assert state.keys() == primary.keys()
        for name, tensor in primary.items():
            assert torch.equal(state[name], tensor), name
            assert state[name].data_ptr() != tensor.data_ptr(), name


def test_replicas_are_independent_copies_that_follow_optimizer_steps():
    replicas = _cpu_replicas()
    _assert_replicas_match_primary(replicas)

    primary = replicas.modules[0]
    trainable = [p for p in primary.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.1)
    images, source_labels = _batch()
    primary(images, source_labels)[0].backward()
    optimizer.step()
    replicas.invalidate()
    replicas.sync()
    _assert_replicas_match_primary(replicas)


def test_invalidate_recopies_trainable_params_without_a_version_bump():
    replicas = _cpu_replicas()
    head = replicas.modules[0].model.concept_head.weight
    head.data.add_(1.0)
    replicas.sync()
    assert not torch.equal(replicas.modules[1].model.concept_head.weight, head)
    replicas.invalidate()
    replicas.sync()
    assert torch.equal(replicas.modules[1].model.concept_head.weight, head)


def test_frozen_param_loads_and_buffer_writes_are_copied():
    replicas = _cpu_replicas()
    model = replicas.modules[0].model
    new_state = {key: torch.randn_like(value) for key, value in model.state_dict().items()}
    model.load_state_dict(new_state)
    model.scale.mul_(3.0)
    replicas.sync()
    _assert_replicas_match_primary(replicas)

    model.stem.weight.data = torch.zeros_like(model.stem.weight)
    replicas.sync()
    _assert_replicas_match_primary(replicas)


def test_sync_mirrors_requires_grad_training_mode_and_flags():
    replicas = _cpu_replicas()
    primary = replicas.modules[0]
    primary.eval()
    primary.model._encoder_frozen = True
    primary.model.stem.weight.requires_grad_(True)
    replicas.sync()
    for replica in replicas.modules[1:]:
        assert all(not module.training for module in replica.modules())
        assert replica.model._encoder_frozen is True
        assert replica.model.stem.weight.requires_grad
        assert not replica.model.stem.bias.requires_grad

    primary.train()
    replicas.sync()
    for replica in replicas.modules[1:]:
        assert all(module.training for module in replica.modules())


def test_replica_window_matches_serial_microbatches():
    """Two minibatches of 4 replica shards of 3 match eight serial microbatches of 3."""
    replicas = _cpu_replicas()
    serial = copy.deepcopy(replicas.modules[0])
    images, source_labels = _batch(batch_size=24)

    for start in range(0, 24, 3):
        loss = serial(images[start : start + 3], source_labels[start : start + 3])[0]
        (loss / 8).backward()

    replicas.sync()
    for start in (0, 12):
        shard_losses = [
            replicas.modules[index](
                images[start + lo : start + hi], source_labels[start + lo : start + hi]
            )[0]
            for index, (lo, hi) in enumerate(contiguous_shard_bounds(12))
        ]
        (mean_shard_losses(shard_losses) / 2).backward()
    replicas.reduce_grads()

    primary_params = dict(replicas.modules[0].named_parameters())
    for name, param in serial.named_parameters():
        if not param.requires_grad:
            assert primary_params[name].grad is None
            continue
        assert primary_params[name].grad is not None, name
        torch.testing.assert_close(primary_params[name].grad, param.grad, rtol=1e-5, atol=1e-6)
    for replica in replicas.modules[1:]:
        assert all(p.grad is None for p in replica.parameters())


def test_zero_grads_drops_replica_grads_without_reducing():
    replicas = _cpu_replicas()
    images, source_labels = _batch(batch_size=3)
    replicas.modules[1](images, source_labels)[0].backward()
    assert any(p.grad is not None for p in replicas.modules[1].parameters())
    replicas.zero_grads()
    assert all(p.grad is None for p in replicas.modules[1].parameters())
    assert all(p.grad is None for p in replicas.modules[0].parameters())


def test_records_train_metrics_scores_every_nth_microbatch():
    assert [i for i in range(10) if records_train_metrics(i, 1)] == list(range(10))
    assert [i for i in range(10) if records_train_metrics(i, 3)] == [0, 3, 6, 9]
    assert records_train_metrics(5, 0)


def test_train_epoch_scores_every_nth_microbatch(
    minimal_config, make_meta_model, tiny_loader, monkeypatch
):
    from mermaidseg.model.eval import Evaluator

    cfg = copy.deepcopy(minimal_config)
    cfg.training = ConfigDict(
        {
            **dict(cfg.training),
            "training_mode": "standard",
            "iterations_per_train_epoch": 3,
            "grad_accum_steps": 2,
            "train_metrics_every": 4,
        }
    )
    meta = make_meta_model(cfg)
    evaluator = Evaluator(num_classes=4, device="cpu")
    metric = evaluator.metric_dict["accuracy"]
    calls = []
    original_update = metric.update

    def _counting_update(*args, **kwargs):
        calls.append(1)
        return original_update(*args, **kwargs)

    monkeypatch.setattr(metric, "update", _counting_update)
    meta.train_epoch(tiny_loader, evaluator)
    # Six microbatches, indices 0..5; every 4th is scored.
    assert len(calls) == 2
    assert meta._train_microbatches_seen == 6


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() != 4,
    reason="needs exactly 4 visible GPUs",
)
def test_four_gpu_window_matches_serial_microbatches_on_one_gpu():
    torch.manual_seed(0)
    target, concepts = _lookups()
    model = _TinyCBM().cuda()
    serial = ShardStep(
        copy.deepcopy(model),
        _TinyLoss().cuda(),
        None,
        None,
        target.cuda(),
        concepts.cuda(),
        False,
        torch.bfloat16,
    )
    replicas = build_shard_replicas(
        model,
        _TinyLoss().cuda(),
        None,
        None,
        target.cuda(),
        concepts.cuda(),
        use_amp=False,
        amp_dtype=torch.bfloat16,
    )
    images, source_labels = _batch(batch_size=24)
    images, source_labels = images.cuda(), source_labels.cuda()

    for start in range(0, 24, 3):
        loss = serial(images[start : start + 3], source_labels[start : start + 3])[0]
        (loss / 8).backward()

    for start in (0, 12):
        parallel = parallel_forward_loss(
            replicas,
            images[start : start + 12],
            source_labels[start : start + 12],
            return_maps=start == 0,
        )
        (parallel.loss / 2).backward()
        assert len(parallel.shards) == (4 if start == 0 else 0)
    replicas.reduce_grads()

    parallel_params = dict(model.named_parameters())
    for name, param in serial.model.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(parallel_params[name].grad, param.grad)
