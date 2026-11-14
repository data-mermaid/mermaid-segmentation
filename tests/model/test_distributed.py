"""Checks for torchrun DDP: sampler shards, collectives, and gradient parity.

The 768px job splits each 12-image microbatch into four contiguous shards of 3
and averages the shard losses. These tests lock that split, the loss-component
reduce, the non-finite abort, and a CPU (gloo) window that must match a serial
single-process accumulation. The last test runs only with 4 visible GPUs.
"""

from __future__ import annotations

import copy
import os
import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

import mermaidseg.model.meta as meta_module
from mermaidseg.model.data_parallel import (
    DDPLossStep,
    ShardStep,
    average_component_dicts,
    mean_shard_losses,
)
from mermaidseg.model.distributed import (
    NCCL_TIMEOUT_ENV,
    ShardedBatchSampler,
    all_ranks_finite,
    collective_timeout_from_env,
    on_main_process,
    reduce_components,
    require_world_size,
)

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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _spawn(worker, world_size: int, extra: tuple = ()) -> list:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    processes = [
        ctx.Process(target=worker, args=(rank, world_size, port, queue, *extra))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=90) for _ in processes]
    except Exception:
        for process in processes:
            process.kill()
        raise
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(f"gloo worker exited with {process.exitcode}")
    results.sort(key=lambda item: item[0])
    return [payload for _, payload in results]


def test_require_world_size_is_four():
    require_world_size(4)
    for world_size in (1, 2, 3, 8):
        with pytest.raises(RuntimeError, match="4 ranks"):
            require_world_size(world_size)


def test_collective_timeout_defaults_to_30_minutes(monkeypatch):
    monkeypatch.delenv(NCCL_TIMEOUT_ENV, raising=False)
    assert collective_timeout_from_env() == timedelta(minutes=30)
    monkeypatch.setenv(NCCL_TIMEOUT_ENV, "  ")
    assert collective_timeout_from_env() == timedelta(minutes=30)


def test_collective_timeout_reads_minutes_from_env(monkeypatch):
    monkeypatch.setenv(NCCL_TIMEOUT_ENV, "45")
    assert collective_timeout_from_env() == timedelta(minutes=45)
    monkeypatch.setenv(NCCL_TIMEOUT_ENV, "2.5")
    assert collective_timeout_from_env() == timedelta(seconds=150)


@pytest.mark.parametrize("value", ["0", "-5"])
def test_collective_timeout_rejects_non_positive(monkeypatch, value):
    monkeypatch.setenv(NCCL_TIMEOUT_ENV, value)
    with pytest.raises(ValueError, match=NCCL_TIMEOUT_ENV):
        collective_timeout_from_env()


def test_stall_dump_seconds_from_env(monkeypatch):
    monkeypatch.delenv(meta_module.STALL_DUMP_ENV, raising=False)
    assert meta_module.stall_dump_seconds_from_env() == 180.0
    monkeypatch.setenv(meta_module.STALL_DUMP_ENV, "45")
    assert meta_module.stall_dump_seconds_from_env() == 45.0
    monkeypatch.setenv(meta_module.STALL_DUMP_ENV, "0")
    assert meta_module.stall_dump_seconds_from_env() == 0.0
    monkeypatch.setenv(meta_module.STALL_DUMP_ENV, "-1")
    assert meta_module.stall_dump_seconds_from_env() == 0.0


class _FakeFaulthandler:
    def __init__(self) -> None:
        self.armed: list[float] = []
        self.cancelled = 0

    def dump_traceback_later(self, timeout, repeat=False, exit=False):
        assert repeat is False and exit is False
        self.armed.append(timeout)

    def cancel_dump_traceback_later(self):
        self.cancelled += 1


def test_watch_for_stalls_rearms_each_update_and_cancels(monkeypatch):
    fake = _FakeFaulthandler()
    monkeypatch.setattr(meta_module, "faulthandler", fake)
    assert list(meta_module.watch_for_stalls(range(3), 5.0)) == [0, 1, 2]
    assert fake.armed == [5.0, 5.0, 5.0]
    assert fake.cancelled == 1


def test_watch_for_stalls_cancels_on_break_and_error(monkeypatch):
    fake = _FakeFaulthandler()
    monkeypatch.setattr(meta_module, "faulthandler", fake)
    for update in meta_module.watch_for_stalls(range(5), 5.0):
        if update == 1:
            break
    assert fake.cancelled == 1

    def run() -> None:
        for update in meta_module.watch_for_stalls(range(5), 5.0):
            if update == 2:
                raise RuntimeError("stalled")

    with pytest.raises(RuntimeError, match="stalled"):
        run()
    assert fake.cancelled == 2


def test_watch_for_stalls_disabled_never_arms(monkeypatch):
    fake = _FakeFaulthandler()
    monkeypatch.setattr(meta_module, "faulthandler", fake)
    assert list(meta_module.watch_for_stalls(range(2), 0.0)) == [0, 1]
    assert fake.armed == [] and fake.cancelled == 0


def test_distributed_init_requires_a_process_group(minimal_config, make_meta_model):
    cfg = copy.deepcopy(minimal_config)
    cfg.training.training_mode = "standard"
    cfg.training.distributed = True
    with pytest.raises(RuntimeError, match="process group"):
        make_meta_model(cfg)


def test_distributed_rejects_data_parallel(minimal_config, make_meta_model):
    cfg = copy.deepcopy(minimal_config)
    cfg.training.training_mode = "standard"
    cfg.training.distributed = True
    cfg.training.data_parallel = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_meta_model(cfg)


def test_sharded_sampler_matches_single_process_shuffle():
    dataset = _IndexDataset(96)
    torch.manual_seed(7)
    expected = [
        list(batch) for batch in DataLoader(dataset, shuffle=True, batch_size=12, drop_last=True)
    ]

    reconstructed: list[list[int]] = []
    rank_batches: list[list[list[int]]] = []
    for rank in range(_WORLD):
        torch.manual_seed(7)
        sampler = ShardedBatchSampler(dataset, 12, rank, _WORLD, broadcast_seed=lambda seed: seed)
        rank_batches.append([list(batch) for batch in DataLoader(dataset, batch_sampler=sampler)])
    for slices in zip(*rank_batches, strict=True):
        reconstructed.append(sum(slices, []))
    assert reconstructed == expected


def test_sharded_sampler_slices_match_batch_sampler_with_injected_seed():
    dataset = _IndexDataset(60)
    seed = 12345
    generator = torch.Generator().manual_seed(seed)
    expected = list(BatchSampler(RandomSampler(dataset, generator=generator), 12, drop_last=True))
    iters = [
        iter(ShardedBatchSampler(dataset, 12, rank, _WORLD, broadcast_seed=lambda _: seed))
        for rank in range(_WORLD)
    ]
    reconstructed = []
    for _ in expected:
        reconstructed.append(sum((next(it) for it in iters), []))
    assert reconstructed == expected
    assert all(len(batch) == 12 for batch in reconstructed)


def _reduce_components_worker(rank: int, world_size: int, port: int, queue) -> None:
    _init_gloo(rank, world_size, port)
    try:
        values = {f"k{i}": float(rank + 1 + i) for i in range(3)}
        values["shared"] = float(rank)
        reduced = reduce_components(values)
        queue.put((rank, reduced))
    finally:
        dist.destroy_process_group()


def _on_main_process_worker(rank: int, world_size: int, port: int, queue) -> None:
    _init_gloo(rank, world_size, port)
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
    counts = _spawn(_on_main_process_worker, _WORLD)
    assert counts == [1, 0, 0, 0]


def test_reduce_components_equals_average_component_dicts():
    per_rank = [
        {f"k{i}": float(rank + 1 + i) for i in range(3)} | {"shared": float(rank)}
        for rank in range(_WORLD)
    ]
    expected = average_component_dicts(per_rank)
    reduced = _spawn(_reduce_components_worker, _WORLD)
    for payload in reduced:
        assert payload == pytest.approx(expected)


def _finite_then_step_worker(rank: int, world_size: int, port: int, queue) -> None:
    _init_gloo(rank, world_size, port)
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
    results = _spawn(_finite_then_step_worker, _WORLD)
    assert all(payload["finite"] is False for payload in results)
    grads = [payload["grad"] for payload in results]
    assert grads == pytest.approx([grads[0]] * _WORLD)


def _ddp_window_worker(
    rank: int,
    world_size: int,
    port: int,
    queue,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    source_labels: torch.Tensor,
) -> None:
    _init_gloo(rank, world_size, port)
    try:
        target, concepts = _lookups()
        step = ShardStep(
            _TinyCBM(),
            _TinyLoss(),
            None,
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
        (mean_shard_losses(shard_losses) / _ACCUM).backward()
    grads = {
        name: param.grad.detach().clone()
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
        None,
        target,
        concepts,
        False,
        torch.bfloat16,
    )
    state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    images, source_labels = _batch()
    serial_grads, serial_weights, serial_clip = _serial_window(state, images, source_labels)
    payloads = _spawn(_ddp_window_worker, _WORLD, extra=(state, images, source_labels))
    for payload in payloads:
        assert payload["clip"] == pytest.approx(serial_clip, rel=1e-5, abs=1e-6)
        for name, grad in serial_grads.items():
            torch.testing.assert_close(payload["grads"][name], grad, rtol=1e-5, atol=1e-6)
        for name, weight in serial_weights.items():
            torch.testing.assert_close(payload["weights"][name], weight, rtol=1e-5, atol=1e-6)


def _nccl_window_worker(
    rank: int,
    world_size: int,
    port: int,
    queue,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    source_labels: torch.Tensor,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        target, concepts = _lookups()
        step = ShardStep(
            _TinyCBM(),
            _TinyLoss(),
            None,
            None,
            target,
            concepts,
            False,
            torch.bfloat16,
        )
        step.load_state_dict(state)
        step = step.to(device)
        ddp = DistributedDataParallel(
            DDPLossStep(step),
            device_ids=[rank],
            broadcast_buffers=False,
        )
        images = images.to(device)
        source_labels = source_labels.to(device)
        for micro in range(_ACCUM):
            start = micro * _GLOBAL_BATCH + rank * _PER_RANK
            sync = micro == _ACCUM - 1
            context = ddp.no_sync() if not sync else torch.enable_grad()
            with context:
                loss = ddp(
                    images[start : start + _PER_RANK],
                    source_labels[start : start + _PER_RANK],
                    return_maps=micro == 0,
                )
                (loss / _ACCUM).backward()
        grads = {
            name: param.grad.detach().cpu().clone()
            for name, param in ddp.module.step.named_parameters()
            if param.requires_grad
        }
        queue.put((rank, grads))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() != 4,
    reason="needs exactly 4 visible GPUs",
)
def test_four_gpu_window_matches_serial_microbatches_on_one_gpu():
    torch.manual_seed(0)
    target, concepts = _lookups()
    reference = ShardStep(
        _TinyCBM(),
        _TinyLoss(),
        None,
        None,
        target,
        concepts,
        False,
        torch.bfloat16,
    )
    state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    images, source_labels = _batch()
    serial_grads, _, _ = _serial_window(state, images, source_labels)
    payloads = _spawn(_nccl_window_worker, _WORLD, extra=(state, images, source_labels))
    for grads in payloads:
        for name, grad in serial_grads.items():
            torch.testing.assert_close(grads[name], grad, rtol=1e-4, atol=1e-5)


def _two_window_maps_worker(rank: int, world_size: int, port: int, queue) -> None:
    """Reproduce the cluster crash: maps on the first micro, then a second window."""
    _init_gloo(rank, world_size, port)
    try:
        torch.manual_seed(0)
        target, concepts = _lookups()
        step = ShardStep(
            _TinyCBM(),
            _TinyLoss(),
            None,
            None,
            target,
            concepts,
            False,
            torch.bfloat16,
        )
        ddp = DistributedDataParallel(DDPLossStep(step), device_ids=None, broadcast_buffers=False)
        images, source_labels = _batch(batch_size=3, seed=rank + 1)
        for window in range(2):
            for micro in range(_ACCUM):
                sync = micro == _ACCUM - 1
                context = ddp.no_sync() if not sync else torch.enable_grad()
                with context:
                    loss = ddp(
                        images,
                        source_labels,
                        return_maps=window == 0 and micro == 0,
                    )
                    assert loss.ndim == 0
                    if window == 0 and micro == 0:
                        stashed = ddp.module.last
                        assert stashed is not None
                        assert stashed[1] is not None
                    (loss / _ACCUM).backward()
            for param in ddp.parameters():
                param.grad = None
        queue.put((rank, True))
    finally:
        dist.destroy_process_group()


def test_two_ddp_windows_survive_returned_metric_maps():
    results = _spawn(_two_window_maps_worker, _WORLD)
    assert results == [True] * _WORLD


def test_ddp_loss_step_returns_only_the_scalar_loss():
    target, concepts = _lookups()
    step = DDPLossStep(
        ShardStep(_TinyCBM(), _TinyLoss(), None, None, target, concepts, False, torch.bfloat16)
    )
    images, source_labels = _batch(batch_size=3)
    loss = step(images, source_labels, return_maps=True)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert step.last is not None
    assert step.last[0] is loss
    assert step.last[1] is not None
