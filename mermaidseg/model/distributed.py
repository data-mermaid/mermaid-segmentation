"""Multi-process data parallel (torchrun + DDP) for the 768px fine-tune.

One process per GPU. Each rank trains on a contiguous slice of every global
microbatch: rank ``r`` of ``W`` gets images ``[r * B / W, (r + 1) * B / W)``,
the same grouping ``scatter`` gives the single-process data-parallel path. DDP
averages the ranks' gradients, which is the gradient of the mean shard loss.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Callable, Iterator, Mapping, Sized
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, RandomSampler, Sampler

from mermaidseg.model.data_parallel import require_divisible_batch
from mermaidseg.model.loss import components_to_floats, stack_components


@dataclass(frozen=True)
class DistContext:
    """Rank layout of this process. A single-process run is rank 0 of 1."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        """Whether this process is rank 0."""
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        """Whether the run has more than one process."""
        return self.world_size > 1


def is_distributed() -> bool:
    """Return whether a process group is initialized."""
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
    """Return whether this process is rank 0."""
    return current_context().is_main


def on_main_process(fn: Callable[[], None]) -> None:
    """Run ``fn`` on rank 0.

    Other ranks wait so they do not enter the next collective. The first barrier is after
    the step's collectives have finished. Rank 0 then runs ``fn`` (a checkpoint write)
    while the others wait on the second barrier, instead of starting the next forward.
    """
    if is_distributed():
        dist.barrier()
    if is_main_process():
        fn()
    if is_distributed():
        dist.barrier()


def require_world_size(world_size: int, expected: int) -> None:
    """Raise unless the job runs ``expected`` ranks.

    The minibatch is split into one shard per rank, and the KL term is
    normalized per shard. Another rank count would change the shard size.

    Args:
        world_size: Rank count of this job.
        expected: Rank count the run config requires.

    Raises:
        RuntimeError: If the counts differ.
    """
    if int(world_size) != int(expected):
        raise RuntimeError(f"distributed expects {expected} ranks, found {world_size}.")


def init_distributed_from_env(expected_world_size: int, nccl_timeout_min: float) -> DistContext:
    """Join the torchrun process group (NCCL) and bind this process to its GPU.

    ``WORLD_SIZE`` (unset means 1) must equal ``expected_world_size``, including
    single-GPU runs. Returns rank 0 of 1 when it is 1.

    Args:
        expected_world_size: Rank count from ``training.world_size``.
        nccl_timeout_min: Minutes a collective may wait for a slow rank before the
            job is killed. PyTorch's own default is 10, shorter than the occasional
            DataLoader stall on network storage.

    Returns:
        This process's rank layout.

    Raises:
        RuntimeError: If ``WORLD_SIZE`` differs from ``expected_world_size``, or a
            multi-rank job has no CUDA.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    require_world_size(world_size, expected_world_size)
    if world_size <= 1:
        return DistContext()
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not is_distributed():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=nccl_timeout_min))
    return DistContext(rank=rank, world_size=world_size, local_rank=local_rank)


def _collective_device() -> torch.device:
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def broadcast_generator_from_rank0(generator: torch.Generator) -> None:
    """Set ``generator`` to rank 0's state on every rank."""
    state = generator.get_state().to(_collective_device())
    dist.broadcast(state, src=0)
    generator.set_state(state.cpu())


class ShardedBatchSampler(Sampler[list[int]]):
    """This rank's contiguous slice of each shuffled global batch.

    Each pass shuffles with ``generator``, as ``RandomSampler`` does, at the same
    point of ``DataLoader`` iteration (the first index fetch, after the loader
    draws ``_base_seed`` from the same generator). With ``broadcast_state`` every
    rank first adopts rank 0's generator state. The global batches are those of a
    single-process ``DataLoader(shuffle=True, batch_size=batch_size,
    drop_last=True, generator=<generator in the same state>)``.
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        rank: int,
        world_size: int,
        *,
        generator: torch.Generator,
        broadcast_state: Callable[[torch.Generator], None] | None = None,
    ) -> None:
        """Validate the rank layout and store the sampler settings.

        Args:
            data_source: Dataset to sample from.
            batch_size: Global batch size.
            rank: Rank of this process.
            world_size: Number of processes.
            generator: Generator the shuffle draws from; the loader's own generator.
            broadcast_state: Optional function that sets the generator to rank 0's state.

        Raises:
            ValueError: If ``rank`` is outside ``0..world_size - 1``.
        """
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(f"rank {rank} is outside 0..{int(world_size) - 1}.")
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.per_rank = require_divisible_batch(self.batch_size, int(world_size))
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.generator = generator
        self.broadcast_state = broadcast_state

    def __len__(self) -> int:
        """Return the number of global batches per pass."""
        return len(self.data_source) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's slice of each shuffled global batch."""
        if self.broadcast_state is not None:
            self.broadcast_state(self.generator)
        batches = BatchSampler(
            RandomSampler(self.data_source, generator=self.generator),
            self.batch_size,
            drop_last=True,
        )
        start = self.rank * self.per_rank
        for batch in batches:
            yield batch[start : start + self.per_rank]


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


def reduce_components(values: Mapping[str, torch.Tensor | float]) -> dict[str, float]:
    """Mean of each logged value over ranks, in float64, as host floats.

    Takes scalar tensors or floats and moves them to the host in one transfer.

    Every rank must pass the same keys. A fingerprint of the sorted keys is
    compared first, so a mismatch raises instead of reducing misaligned values.
    """
    if not is_distributed():
        return components_to_floats(values)
    keys = sorted(values)
    device = _collective_device()
    fingerprint = _key_fingerprint(keys)
    bounds = torch.tensor([fingerprint, -fingerprint], dtype=torch.int64, device=device)
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX)
    if int(bounds[0]) != fingerprint or int(bounds[1]) != -fingerprint:
        raise RuntimeError(f"ranks logged different loss components; this rank has {keys}.")
    if not keys:
        return {}
    stacked = stack_components([values[key] for key in keys], device)
    reduced = all_reduce_mean(stacked).tolist()
    return dict(zip(keys, reduced, strict=True))
