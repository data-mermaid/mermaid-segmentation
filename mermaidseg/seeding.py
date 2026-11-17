"""Seeding for a run's Python, NumPy and torch RNGs and for DataLoader workers."""

from __future__ import annotations

import functools
import random
from collections.abc import Callable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy's global RNG and torch (CPU and every CUDA device).

    Args:
        seed: Seed for all three RNGs, in ``0..2**32 - 1`` (NumPy's range).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(rank: int, inner: Callable[[int], None] | None, worker_id: int) -> None:
    """Run ``inner``, then reseed ``random``, NumPy and torch for this (rank, worker)."""
    if inner is not None:
        inner(worker_id)
    state = np.random.SeedSequence([torch.initial_seed(), int(rank)]).generate_state(
        4, dtype=np.uint32
    )
    torch.manual_seed((int(state[0]) << 32) | int(state[1]))
    random.seed((int(state[2]) << 32) | int(state[3]))
    np.random.seed(state)


def make_worker_seeder(
    rank: int, inner: Callable[[int], None] | None = None
) -> Callable[[int], None]:
    """DataLoader ``worker_init_fn`` that runs ``inner``, then reseeds the worker's RNGs.

    The seeds derive from the worker's torch seed (the loader's ``_base_seed`` plus the
    worker id, drawn from the loader's ``generator``) and ``rank``. Every rank's loader
    draws the same ``_base_seed``, so without the rank worker ``k`` would get the same
    seeds on every rank, and the random fallback index for an empty crop would repeat
    across the ranks' slices of one batch.

    Args:
        rank: Rank of this process; 0 outside distributed runs.
        inner: Worker init to run first, or ``None``.

    Returns:
        A picklable worker init function.
    """
    # A partial over a module-level function pickles, so spawn-started workers can load it.
    return functools.partial(_seed_worker, rank, inner)
