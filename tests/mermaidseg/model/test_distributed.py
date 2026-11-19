"""Checks for torchrun DDP: sampler shards, collectives, and gradient parity.

The 768px job splits each 12-image microbatch into four contiguous shards of 3
and averages the shard losses. These tests lock that split, the loss-component
reduce, the non-finite abort, and a CPU (gloo) window that must match a serial
single-process accumulation. The last test runs only with 4 visible GPUs.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    TensorDataset,
)

import mermaidseg.model.meta as meta_module
import mermaidseg.model.models as models_module
from mermaidseg.model.data_parallel import DDPLossStep, ShardStep
from mermaidseg.model.distributed import (
    DistContext,
    ShardedBatchSampler,
    all_ranks_finite,
    broadcast_generator_from_rank0,
    init_distributed_from_env,
    on_main_process,
    reduce_components,
    require_world_size,
)
from tests.conftest import (
    CBM_CONCEPT_VALUE2ID,
    CBM_NUM_CONCEPTS,
    IMAGE_SIZE,
    NUM_CLASSES,
    cbm_concept_lookup,
    cbm_test_config,
)
from tests.mermaidseg.conftest import init_gloo, spawn_gloo

_NUM_SOURCE = 9
_NUM_CONCEPTS = 5
_NUM_CLASSES = 4
_WORLD = 4
_GLOBAL_BATCH = 12
_PER_RANK = _GLOBAL_BATCH // _WORLD
_ACCUM = 3


class _IndexDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


class _TinyCBM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 1)
        self.concept_head = nn.Conv2d(6, _NUM_CONCEPTS, 1)
        self.classifier = nn.Conv2d(_NUM_CONCEPTS, _NUM_CLASSES, 1)
        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        concept_logits = self.concept_head(torch.relu(self.stem(x)))
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
        return loss, {
            "concepts": float(concept_loss.detach()),
            "classification": float(class_loss.detach()),
        }


class _LinearRank(nn.Module):
    """One trainable weight so a skipped DDP backward still has a hook to fire."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (self.scale * values).mean()


def _lookups(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.randint(0, _NUM_CLASSES, (_NUM_SOURCE + 1,), generator=generator)
    target[0] = 0
    concepts = torch.rand(_NUM_SOURCE + 1, _NUM_CONCEPTS, generator=generator)
    concepts[0] = 0
    return target, concepts


def _batch(batch_size: int = 36, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(batch_size, 3, 8, 8, generator=generator)
    source_labels = torch.randint(0, _NUM_SOURCE + 1, (batch_size, 8, 8), generator=generator)
    return images, source_labels


def test_require_world_size_matches_expected():
    require_world_size(4, 4)
    for world_size in (1, 2, 3, 8):
        with pytest.raises(RuntimeError, match="4 ranks"):
            require_world_size(world_size, 4)


def test_init_distributed_single_process_when_one_rank_expected(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ctx = init_distributed_from_env(expected_world_size=1, nccl_timeout_min=30)
    assert ctx == DistContext()
    assert not ctx.distributed


def test_init_distributed_rejects_mismatched_world_size_before_any_group_starts(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    started: list[object] = []
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: started.append(k))
    with pytest.raises(RuntimeError, match="4 ranks"):
        init_distributed_from_env(expected_world_size=4, nccl_timeout_min=30)
    assert started == []


@pytest.mark.parametrize(
    ("minutes", "expected"), [(45.0, timedelta(minutes=45)), (2.5, timedelta(seconds=150))]
)
def test_init_distributed_passes_nccl_timeout_minutes(monkeypatch, minutes, expected):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    seen: dict[str, object] = {}
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: seen.update(kwargs))
    ctx = init_distributed_from_env(expected_world_size=2, nccl_timeout_min=minutes)
    assert seen == {"backend": "nccl", "timeout": expected}
    assert (ctx.rank, ctx.world_size, ctx.local_rank) == (1, 2, 1)


def test_distributed_init_requires_a_process_group(minimal_config, make_meta_model):
    cfg = copy.deepcopy(minimal_config)
    cfg.training.distributed = True
    with pytest.raises(RuntimeError, match="process group"):
        make_meta_model(cfg)


def test_sharded_sampler_matches_single_process_shuffle():
    dataset = _IndexDataset(96)
    expected = [
        list(batch)
        for batch in DataLoader(
            dataset,
            shuffle=True,
            batch_size=12,
            drop_last=True,
            generator=torch.Generator().manual_seed(7),
        )
    ]

    rank_batches: list[list[list[int]]] = []
    for rank in range(_WORLD):
        generator = torch.Generator().manual_seed(7)
        sampler = ShardedBatchSampler(
            dataset, 12, rank, _WORLD, generator=generator, broadcast_state=lambda _: None
        )
        loader = DataLoader(dataset, batch_sampler=sampler, generator=generator)
        rank_batches.append([list(batch) for batch in loader])
    reconstructed = [sum(slices, []) for slices in zip(*rank_batches, strict=True)]
    assert reconstructed == expected


def test_sharded_sampler_shuffles_from_the_broadcast_state():
    dataset = _IndexDataset(60)
    rank0_state = torch.Generator().manual_seed(12345).get_state()
    expected = list(
        BatchSampler(
            RandomSampler(dataset, generator=torch.Generator().manual_seed(12345)),
            12,
            drop_last=True,
        )
    )
    iters = [
        iter(
            ShardedBatchSampler(
                dataset,
                12,
                rank,
                _WORLD,
                generator=torch.Generator().manual_seed(rank),
                broadcast_state=lambda generator: generator.set_state(rank0_state),
            )
        )
        for rank in range(_WORLD)
    ]
    reconstructed = [sum((next(it) for it in iters), []) for _ in expected]
    assert reconstructed == expected


def _broadcast_generator_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        generator = torch.Generator().manual_seed(100 + rank)
        broadcast_generator_from_rank0(generator)
        queue.put((rank, torch.rand(3, generator=generator).tolist()))
    finally:
        dist.destroy_process_group()


def test_broadcast_generator_gives_every_rank_rank0_state():
    expected = torch.rand(3, generator=torch.Generator().manual_seed(100)).tolist()
    assert spawn_gloo(_broadcast_generator_worker, 2) == [expected, expected]


def _reduce_components_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        values = {f"k{i}": float(rank + 1 + i) for i in range(3)}
        values["shared"] = float(rank)
        reduced = reduce_components(values)
        queue.put((rank, reduced))
    finally:
        dist.destroy_process_group()


def _on_main_process_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        ran = 0

        def body() -> None:
            nonlocal ran
            ran += 1

        on_main_process(body)
        queue.put((rank, ran))
    finally:
        dist.destroy_process_group()


def test_on_main_process_runs_on_one_rank():
    counts = spawn_gloo(_on_main_process_worker, _WORLD)
    assert counts == [1, 0, 0, 0]


def test_reduce_components_is_the_mean_over_ranks():
    per_rank = [
        {f"k{i}": float(rank + 1 + i) for i in range(3)} | {"shared": float(rank)}
        for rank in range(_WORLD)
    ]
    expected = {key: sum(d[key] for d in per_rank) / _WORLD for key in per_rank[0]}
    reduced = spawn_gloo(_reduce_components_worker, _WORLD)
    for payload in reduced:
        assert payload == pytest.approx(expected)


def _finite_then_step_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        torch.manual_seed(0)
        module = DistributedDataParallel(
            _LinearRank(),
            device_ids=None,
            broadcast_buffers=False,
        )
        values = torch.tensor([float(rank + 1), 2.0, 3.0])
        loss = module(values)
        if rank == 1:
            loss = loss * float("inf")
        finite = all_ranks_finite(loss)
        if not finite:
            params = [param for param in module.module.parameters() if param.requires_grad]
            sum(param.sum() * 0.0 for param in params).backward()
            module.zero_grad()
        else:
            raise AssertionError("expected a non-finite shard")

        next_loss = module(values)
        next_loss.backward()
        scale = module.module.scale
        assert scale.grad is not None
        queue.put((rank, {"finite": finite, "grad": float(scale.grad), "next": float(next_loss)}))
    finally:
        dist.destroy_process_group()


def test_non_finite_shard_aborts_the_window_on_every_rank():
    results = spawn_gloo(_finite_then_step_worker, _WORLD)
    assert all(payload["finite"] is False for payload in results)
    grads = [payload["grad"] for payload in results]
    assert grads == pytest.approx([grads[0]] * _WORLD)


def _ddp_window_worker(
    rank: int,
    world_size: int,
    init_method: str,
    queue,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    source_labels: torch.Tensor,
) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        target, concepts = _lookups()
        step = ShardStep(
            _TinyCBM(),
            _TinyLoss(),
            None,
            target,
            concepts,
            False,
            torch.bfloat16,
        )
        step.load_state_dict(state)
        ddp = DistributedDataParallel(
            DDPLossStep(step),
            device_ids=None,
            broadcast_buffers=False,
        )
        for micro in range(_ACCUM):
            start = micro * _GLOBAL_BATCH + rank * _PER_RANK
            shard_images = images[start : start + _PER_RANK]
            shard_labels = source_labels[start : start + _PER_RANK]
            sync = micro == _ACCUM - 1
            context = ddp.no_sync() if not sync else torch.enable_grad()
            with context:
                loss = ddp(shard_images, shard_labels, return_maps=micro == 0)
                (loss / _ACCUM).backward()
        inner = ddp.module.step
        grads = {
            name: param.grad.detach().clone()
            for name, param in inner.named_parameters()
            if param.requires_grad
        }
        clip = torch.nn.utils.clip_grad_norm_(
            [param for param in inner.parameters() if param.requires_grad], 0.5
        )
        optimizer = torch.optim.AdamW(
            [param for param in inner.parameters() if param.requires_grad],
            lr=1e-3,
        )
        optimizer.step()
        weights = {
            name: param.detach().clone()
            for name, param in inner.named_parameters()
            if param.requires_grad
        }
        # numpy, not tensors: queued tensors go through shared memory, which can be gone by the
        # time the parent reads them once this worker exits (EOFError in queue.get).
        grads = {name: tensor.cpu().numpy() for name, tensor in grads.items()}
        weights = {name: tensor.cpu().numpy() for name, tensor in weights.items()}
        queue.put((rank, {"grads": grads, "weights": weights, "clip": float(clip)}))
    finally:
        dist.destroy_process_group()


def _serial_window(
    state: dict[str, torch.Tensor], images: torch.Tensor, source_labels: torch.Tensor
):
    target, concepts = _lookups()
    step = ShardStep(
        _TinyCBM(),
        _TinyLoss(),
        None,
        target,
        concepts,
        False,
        torch.bfloat16,
    )
    step.load_state_dict(state)
    for micro in range(_ACCUM):
        shard_losses = []
        for rank in range(_WORLD):
            start = micro * _GLOBAL_BATCH + rank * _PER_RANK
            loss = step(
                images[start : start + _PER_RANK],
                source_labels[start : start + _PER_RANK],
                return_maps=False,
            )[0]
            shard_losses.append(loss)
        (torch.stack(shard_losses).mean() / _ACCUM).backward()
    grads = {
        name: cast(torch.Tensor, param.grad).detach().clone()
        for name, param in step.named_parameters()
        if param.requires_grad
    }
    clip = torch.nn.utils.clip_grad_norm_(
        [param for param in step.parameters() if param.requires_grad], 0.5
    )
    optimizer = torch.optim.AdamW(
        [param for param in step.parameters() if param.requires_grad],
        lr=1e-3,
    )
    optimizer.step()
    weights = {
        name: param.detach().clone()
        for name, param in step.named_parameters()
        if param.requires_grad
    }
    return grads, weights, float(clip)


def test_replica_window_matches_serial_microbatches():
    torch.manual_seed(0)
    target, concepts = _lookups()
    reference = ShardStep(
        _TinyCBM(),
        _TinyLoss(),
        None,
        target,
        concepts,
        False,
        torch.bfloat16,
    )
    state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    images, source_labels = _batch()
    serial_grads, serial_weights, serial_clip = _serial_window(state, images, source_labels)
    payloads = spawn_gloo(_ddp_window_worker, _WORLD, extra=(state, images, source_labels))
    for payload in payloads:
        assert payload["clip"] == pytest.approx(serial_clip, rel=1e-5, abs=1e-6)
        for name, grad in serial_grads.items():
            torch.testing.assert_close(
                torch.from_numpy(payload["grads"][name]), grad, rtol=1e-5, atol=1e-6
            )
        for name, weight in serial_weights.items():
            torch.testing.assert_close(
                torch.from_numpy(payload["weights"][name]), weight, rtol=1e-5, atol=1e-6
            )


def test_ddp_loss_step_returns_only_the_scalar_loss():
    target, concepts = _lookups()
    step = DDPLossStep(
        ShardStep(_TinyCBM(), _TinyLoss(), None, target, concepts, False, torch.bfloat16)
    )
    images, source_labels = _batch(batch_size=3)
    loss = step(images, source_labels, return_maps=True)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert step.last is not None
    assert step.last[0] is loss
    assert step.last[1] is not None


def test_resume_at_an_epoch_boundary_then_validate_then_train(
    minimal_config, make_meta_model, tmp_path
):
    """A resume at an epoch's last update builds DDP inside validation, then trains."""
    cfg = copy.deepcopy(minimal_config)
    cfg.training.distributed = True
    cfg.training.iterations_per_train_epoch = 1
    cfg.training.iterations_per_val_epoch = 1
    init_gloo(0, 1, f"file://{tmp_path}/rendezvous")
    try:
        meta = make_meta_model(cfg, run_name="resume-boundary")
        meta.global_step = 1
        images = torch.randn(2, 3, *IMAGE_SIZE)
        labels = torch.ones(2, *images.shape[-2:], dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        meta.train_epoch(loader, evaluator=None, epoch=0)
        meta.validation_epoch(loader)
        loss, _, _ = meta.train_epoch(loader, evaluator=None, epoch=1)
        assert torch.isfinite(torch.tensor(loss))
    finally:
        dist.destroy_process_group()


class _MockEncoder(nn.Module):
    """Tiny DINOv3 stand-in with a ``q_proj`` LoRA target, as ``mock_dinov3_encoder`` builds."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=16, patch_size=14, num_hidden_layers=4, num_register_tokens=4
        )
        self.embed = nn.Linear(3, 16)
        self.q_proj = nn.Linear(16, 16)

    def forward(self, pixel_values: torch.Tensor, **_: object) -> SimpleNamespace:
        patches = nn.functional.avg_pool2d(pixel_values, 14).flatten(2).transpose(1, 2)
        tokens = self.embed(patches)
        prefix = tokens.new_zeros(tokens.shape[0], 5, tokens.shape[2])
        states = self.q_proj(torch.cat([prefix, tokens], dim=1))
        return SimpleNamespace(hidden_states=(states,) * 5)


class _StepLogger:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def log(self, metrics: dict[str, object], step: int) -> None:
        self.steps.append(step)


def _distributed_meta_model(world_size: int) -> meta_module.MetaModel:
    cfg = cbm_test_config()
    cfg.training.distributed = True
    cfg.training.world_size = world_size
    cfg.training.grad_accum_steps = 2
    cfg.training.iterations_per_train_epoch = 4
    cfg.training.iterations_per_val_epoch = 2
    torch.manual_seed(0)
    auto_model = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _MockEncoder())
    with patch.object(models_module, "AutoModel", auto_model):
        return meta_module.MetaModel(
            run_name=f"ddp-train-validate-{world_size}",
            num_classes=NUM_CLASSES,
            model_kwargs=cfg.model,
            training_kwargs=cfg.training,
            device="cpu",
            num_concepts=CBM_NUM_CONCEPTS,
            concept_value2id=cast("dict[str, dict[str, list[int]]]", CBM_CONCEPT_VALUE2ID),
            source_to_concepts_lookup=cbm_concept_lookup(),
        )


def _train_then_validate_worker(rank: int, world_size: int, init_method: str, queue) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        meta = _distributed_meta_model(world_size)
        run_logger = _StepLogger() if rank == 0 else None
        generator = torch.Generator().manual_seed(rank)
        images = torch.randn(16, 3, *IMAGE_SIZE, generator=generator)
        labels = torch.randint(1, NUM_CLASSES, (16, *IMAGE_SIZE), generator=generator)
        loader: DataLoader[Any] = DataLoader(TensorDataset(images, labels), batch_size=2)
        meta.train_epoch(loader, run_logger=run_logger, log_interval=2, epoch=0)
        meta.validation_epoch(loader)
        logged = None if run_logger is None else run_logger.steps
        queue.put((rank, {"global_step": meta.global_step, "logged": logged}))
    finally:
        dist.destroy_process_group()


def test_logger_only_on_rank_zero_trains_and_validates_without_desync():
    """Every rank joins the log-interval reduction, or rank 0 blocks in it alone."""
    results = spawn_gloo(_train_then_validate_worker, 2)
    assert [payload["global_step"] for payload in results] == [4, 4]
    assert [payload["logged"] for payload in results] == [[2, 4], None]
