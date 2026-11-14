"""Multi-process data parallel (torchrun + DDP) for the 768px fine-tune.

One process per GPU. Each rank trains on a contiguous slice of every global
microbatch: rank ``r`` of ``W`` gets images ``[r * B / W, (r + 1) * B / W)``,
the same grouping ``scatter`` gives the single-process data-parallel path. DDP
averages the ranks' gradients, which is the gradient of the mean shard loss.
"""

from __future__ import annotations

import os
import random
import zlib
from collections.abc import Callable, Iterator, Sized
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, RandomSampler, Sampler

from mermaidseg.model.data_parallel import DATA_PARALLEL_GPUS, require_divisible_batch

NCCL_TIMEOUT_ENV = "MERMAIDSEG_NCCL_TIMEOUT_MIN"
DEFAULT_NCCL_TIMEOUT_MIN = 30.0


@dataclass(frozen=True)
class DistContext:
    """Rank layout of this process. A single-process run is rank 0 of 1."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def current_context() -> DistContext:
    """Layout of the initialized process group, or rank 0 of 1 without one."""
    if not is_distributed():
        return DistContext()
    return DistContext(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=int(os.environ.get("LOCAL_RANK", dist.get_rank())),
    )


def is_main_process() -> bool:
    return current_context().is_main


def on_main_process(fn: Callable[[], None]) -> None:
    """Run ``fn`` on rank 0. Other ranks wait so they do not enter the next collective.

    The first barrier is after the step's collectives have finished. Rank 0 then
    runs ``fn`` (a checkpoint write) while the others wait on the second barrier,
    instead of starting the next forward.
    """
    if is_distributed():
        dist.barrier()
    if is_main_process():
        fn()
    if is_distributed():
        dist.barrier()


def require_world_size(world_size: int, expected: int = DATA_PARALLEL_GPUS) -> None:
    """Raise unless the job runs ``expected`` ranks.

    The 768px minibatch is split into one shard per rank, and the KL term is
    normalized per shard. Another rank count would change the shard size.
    """
    if int(world_size) != int(expected):
        raise RuntimeError(f"distributed expects {expected} ranks, found {world_size}.")


def collective_timeout_from_env() -> timedelta:
    """How long a collective may wait for a slow rank before the job is killed.

    ``MERMAIDSEG_NCCL_TIMEOUT_MIN`` (minutes, default 30). PyTorch's own default
    is 10 minutes, shorter than the occasional DataLoader stall on network storage.
    """
    raw = os.environ.get(NCCL_TIMEOUT_ENV, "").strip()
    minutes = DEFAULT_NCCL_TIMEOUT_MIN if not raw else float(raw)
    if minutes <= 0:
        raise ValueError(f"{NCCL_TIMEOUT_ENV} must be > 0 minutes, got {raw!r}.")
    return timedelta(minutes=minutes)


def init_distributed_from_env(expected_world_size: int = DATA_PARALLEL_GPUS) -> DistContext:
    """Join the torchrun process group (NCCL) and bind this process to its GPU.

    Returns rank 0 of 1 when ``WORLD_SIZE`` is unset or 1.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistContext()
    require_world_size(world_size, expected_world_size)
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not is_distributed():
        dist.init_process_group(backend="nccl", timeout=collective_timeout_from_env())
    return DistContext(rank=rank, world_size=world_size, local_rank=local_rank)


def _collective_device() -> torch.device:
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def broadcast_from_rank0(value: int) -> int:
    """Rank 0's ``value`` on every rank."""
    tensor = torch.tensor([int(value)], dtype=torch.int64, device=_collective_device())
    dist.broadcast(tensor, src=0)
    return int(tensor.item())


class ShardedBatchSampler(Sampler[list[int]]):
    """This rank's contiguous slice of each shuffled global batch.

    Each pass draws one seed from the global torch RNG, as ``RandomSampler``
    does without a generator, and at the same point of ``DataLoader``
    iteration (the first index fetch, after the loader's ``_base_seed``). With
    ``broadcast_seed`` every rank then uses rank 0's seed. The global batches
    are those of ``DataLoader(shuffle=True, batch_size=batch_size,
    drop_last=True)`` in a single process with the same RNG state.
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        rank: int,
        world_size: int,
        *,
        broadcast_seed: Callable[[int], int] | None = None,
    ) -> None:
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(f"rank {rank} is outside 0..{int(world_size) - 1}.")
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.per_rank = require_divisible_batch(self.batch_size, int(world_size))
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.broadcast_seed = broadcast_seed

    def __len__(self) -> int:
        return len(self.data_source) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        if self.broadcast_seed is not None:
            seed = int(self.broadcast_seed(seed))
        generator = torch.Generator()
        generator.manual_seed(seed)
        batches = BatchSampler(
            RandomSampler(self.data_source, generator=generator),
            self.batch_size,
            drop_last=True,
        )
        start = self.rank * self.per_rank
        for batch in batches:
            yield batch[start : start + self.per_rank]


def make_rank_worker_init_fn(
    rank: int, inner: Callable[[int], None] | None = None
) -> Callable[[int], None]:
    """Worker init that reseeds ``random``, numpy, and torch per (rank, worker).

    Every rank's loader draws the same ``_base_seed`` from the shared global
    RNG, so without this worker ``k`` would get the same seeds on every rank,
    and the random fallback index for an empty crop would repeat across the
    ranks' slices of one batch.
    """

    def init_fn(worker_id: int) -> None:
        if inner is not None:
            inner(worker_id)
        state = np.random.SeedSequence([torch.initial_seed(), int(rank)]).generate_state(
            4, dtype=np.uint32
        )
        torch.manual_seed((int(state[0]) << 32) | int(state[1]))
        random.seed((int(state[2]) << 32) | int(state[3]))
        np.random.seed(state)

    return init_fn


def all_ranks_finite(loss: torch.Tensor) -> bool:
    """True when ``loss`` is finite on every rank (on this rank without a process group)."""
    finite = torch.isfinite(loss.detach()).all()
    if not is_distributed():
        return bool(finite)
    flag = finite.to(device=_collective_device(), dtype=torch.int32).reshape(1)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Mean of ``tensor`` over ranks (a copy; the input is unchanged)."""
    if not is_distributed():
        return tensor.clone()
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / dist.get_world_size()


def _key_fingerprint(keys: list[str]) -> int:
    return zlib.crc32("\n".join(keys).encode()) + (len(keys) << 32)


def reduce_components(values: dict[str, float]) -> dict[str, float]:
    """Mean of each logged value over ranks, in float64.

    Every rank must pass the same keys. A fingerprint of the sorted keys is
    compared first, so a mismatch raises instead of reducing misaligned values.
    """
    if not is_distributed():
        return dict(values)
    keys = sorted(values)
    device = _collective_device()
    fingerprint = _key_fingerprint(keys)
    bounds = torch.tensor([fingerprint, -fingerprint], dtype=torch.int64, device=device)
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX)
    if int(bounds[0]) != fingerprint or int(bounds[1]) != -fingerprint:
        raise RuntimeError(f"ranks logged different loss components; this rank has {keys}.")
    if not keys:
        return {}
    stacked = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    reduced = all_reduce_mean(stacked).tolist()
    return dict(zip(keys, reduced, strict=True))
