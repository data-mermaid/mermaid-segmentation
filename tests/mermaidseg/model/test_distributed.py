"""Checks for torchrun DDP: rank setup, sampler shards, and the collective helpers.

The 768px job splits each 12-image microbatch into four contiguous shards of 3
and averages the shard losses. These tests lock that split, the loss-component
reduce, and the non-finite abort. No process group is started: the collectives
are faked serially by ``LockstepRanks``.
"""

from __future__ import annotations

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
)

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

_WORLD = 4
_GLOBAL_BATCH = 12
_PER_RANK = _GLOBAL_BATCH // _WORLD


class _IndexDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


def test_require_world_size_matches_expected():
    require_world_size(4, 4)
    with pytest.raises(RuntimeError, match="ranks"):
        require_world_size(2, 4)


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


def test_init_distributed_requires_cuda_for_a_multi_rank_job(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    started: list[object] = []
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: started.append(k))

    with pytest.raises(RuntimeError, match="requires CUDA"):
        init_distributed_from_env(expected_world_size=2, nccl_timeout_min=30)

    assert started == []


@pytest.mark.parametrize(("minutes", "expected"), [(45.0, timedelta(minutes=45))])
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


def test_sharded_sampler_length_is_the_number_of_whole_global_batches():
    sampler = ShardedBatchSampler(
        _IndexDataset(100), 12, 1, _WORLD, generator=torch.Generator().manual_seed(0)
    )

    # 100 images make 8 whole batches of 12; the 4 left over are dropped.
    assert len(sampler) == 8
    assert [len(batch) for batch in sampler] == [_PER_RANK] * 8


def test_broadcast_generator_gives_every_rank_rank0_state(lockstep_ranks):
    def body(rank: int) -> list[float]:
        generator = torch.Generator().manual_seed(100 + rank)
        broadcast_generator_from_rank0(generator)
        return torch.rand(3, generator=generator).tolist()

    expected = torch.rand(3, generator=torch.Generator().manual_seed(100)).tolist()
    assert lockstep_ranks(2).run(body) == [expected, expected]


def test_on_main_process_runs_on_one_rank_between_two_barriers(lockstep_ranks):
    ranks = lockstep_ranks(_WORLD)

    def body(_rank: int) -> int:
        ran = 0

        def fn() -> None:
            nonlocal ran
            ran += 1

        on_main_process(fn)
        return ran

    assert ranks.run(body) == [1, 0, 0, 0]
    assert ranks.barriers == [2] * _WORLD


def test_reduce_components_raises_on_every_rank_when_the_keys_differ(lockstep_ranks):
    def body(rank: int) -> str:
        values = {"classification": 1.0, "concepts": 2.0 if rank == 0 else 3.0}
        if rank == 1:
            values["extra"] = 4.0
        try:
            reduce_components(values)
        except RuntimeError as error:
            return str(error)
        return "reduced"

    assert lockstep_ranks(2).run(body) == [
        "ranks logged different loss components; this rank has ['classification', 'concepts'].",
        "ranks logged different loss components; "
        "this rank has ['classification', 'concepts', 'extra'].",
    ]


def _rank_components(rank: int) -> dict[str, float]:
    return {f"k{i}": float(rank + 1 + i) for i in range(3)} | {"shared": float(rank)}


def test_reduce_components_is_the_mean_over_ranks(lockstep_ranks):
    per_rank = [_rank_components(rank) for rank in range(_WORLD)]
    expected = {key: sum(d[key] for d in per_rank) / _WORLD for key in per_rank[0]}

    reduced = lockstep_ranks(_WORLD).run(lambda rank: reduce_components(_rank_components(rank)))

    for payload in reduced:
        assert payload == pytest.approx(expected)


def test_a_non_finite_loss_on_one_rank_reads_as_non_finite_on_every_rank(lockstep_ranks):
    def body(rank: int) -> dict[str, bool]:
        bad_on_rank_one = torch.tensor([1.0, float("inf") if rank == 1 else 2.0])
        return {
            "clean": all_ranks_finite(torch.tensor([1.0, 2.0])),
            "one_rank_bad": all_ranks_finite(bad_on_rank_one),
        }

    assert lockstep_ranks(_WORLD).run(body) == [{"clean": True, "one_rank_bad": False}] * _WORLD
