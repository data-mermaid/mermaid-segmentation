"""Rank-aware DataLoader worker init under spawn-started workers."""

from __future__ import annotations

from mermaidseg.datasets import make_worker_init_fn
from mermaidseg.model.distributed import make_rank_worker_init_fn
from tests.mermaidseg.datasets import test_base_dataset as base

spawn_cache_stats = base.spawn_cache_stats


def test_rank_worker_init_runs_the_inner_init_in_a_spawned_worker(
    monkeypatch, tmp_path, spawn_cache_stats
):
    init_fn = make_rank_worker_init_fn(1, make_worker_init_fn(spawn_cache_stats))
    base.run_in_spawned_worker(init_fn, tmp_path / "cache", monkeypatch)
    assert spawn_cache_stats.local_hits.value == 1
