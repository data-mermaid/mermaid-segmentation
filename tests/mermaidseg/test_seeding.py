"""Run and DataLoader-worker seeding: Python, NumPy and torch draws repeat per seed."""

from __future__ import annotations

import functools
import multiprocessing as mp
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from mermaidseg.seeding import make_worker_seeder, seed_everything

DRAWS: dict[str, Callable[[], float]] = {
    "random": random.random,
    "numpy": lambda: float(np.random.rand()),
    "torch": lambda: float(torch.rand(1).item()),
}


@pytest.mark.parametrize("draw", DRAWS.values(), ids=DRAWS.keys())
def test_seed_everything_repeats_draws_for_a_seed(draw: Callable[[], float]) -> None:
    seed_everything(11)
    first = draw()
    seed_everything(11)
    assert draw() == first
    seed_everything(12)
    assert draw() != first


def _seeded_draw(
    monkeypatch: pytest.MonkeyPatch,
    draw: Callable[[], float],
    initial_seed: int,
    rank: int,
    prior: int = 0,
) -> float:
    """A draw after the seeder runs for ``(initial_seed, rank)`` from global RNGs at ``prior``."""
    random.seed(prior)
    np.random.seed(prior)
    torch.manual_seed(prior)
    monkeypatch.setattr(torch, "initial_seed", lambda: initial_seed)
    make_worker_seeder(rank)(0)
    return draw()


@pytest.mark.parametrize("draw", DRAWS.values(), ids=DRAWS.keys())
def test_worker_seeder_repeats_draws_for_a_worker_and_rank(
    monkeypatch: pytest.MonkeyPatch, draw: Callable[[], float]
) -> None:
    first = _seeded_draw(monkeypatch, draw, 1000, 1, prior=1)
    assert _seeded_draw(monkeypatch, draw, 1000, 1, prior=2) == first


@pytest.mark.parametrize("draw", DRAWS.values(), ids=DRAWS.keys())
def test_worker_seeder_differs_across_ranks(
    monkeypatch: pytest.MonkeyPatch, draw: Callable[[], float]
) -> None:
    assert _seeded_draw(monkeypatch, draw, 1000, 0) != _seeded_draw(monkeypatch, draw, 1000, 1)


@pytest.mark.parametrize("draw", DRAWS.values(), ids=DRAWS.keys())
def test_worker_seeder_differs_across_workers(
    monkeypatch: pytest.MonkeyPatch, draw: Callable[[], float]
) -> None:
    assert _seeded_draw(monkeypatch, draw, 1000, 0) != _seeded_draw(monkeypatch, draw, 1001, 0)


def _write_worker_id(path: Path, worker_id: int) -> None:
    path.write_text(str(worker_id))


def test_worker_seeder_runs_the_inner_init_in_a_spawned_worker(tmp_path):
    marker = tmp_path / "worker_id"
    init_fn = make_worker_seeder(1, functools.partial(_write_worker_id, marker))
    worker = mp.get_context("spawn").Process(target=init_fn, args=(3,))
    worker.start()
    try:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join()
    assert marker.read_text() == "3"
