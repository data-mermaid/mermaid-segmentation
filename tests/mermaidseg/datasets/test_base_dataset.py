"""DataLoader worker init: it reaches spawn-started workers and attaches the shared stats."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path

import pytest

import mermaidseg.datasets.local_cache
from mermaidseg.datasets import make_worker_init_fn
from mermaidseg.datasets.local_cache import CacheStatsHandles, LocalS3Cache, create_cache_stats

BUCKET = "bucket"
KEY = "image.jpg"


def read_cached_object_in_worker(init_fn: Callable[[int], None]) -> None:
    """A worker body: run the init, then read one object the local cache already holds."""
    init_fn(0)
    LocalS3Cache.get().read_bytes(BUCKET, KEY)


@pytest.fixture
def spawn_cache_stats(monkeypatch) -> CacheStatsHandles:
    """Cache stats whose locks a spawn-started process can inherit, whatever the default."""
    # A fork-context SemLock cannot be passed to a spawn-context process.
    monkeypatch.setattr(mermaidseg.datasets.local_cache, "mp", mp.get_context("spawn"))
    return create_cache_stats()


def run_in_spawned_worker(init_fn: Callable[[int], None], cache_root: Path, monkeypatch) -> None:
    """Run ``read_cached_object_in_worker`` in a spawn-started process over ``cache_root``."""
    (cache_root / BUCKET).mkdir(parents=True)
    (cache_root / BUCKET / KEY).write_bytes(b"pixels")
    monkeypatch.setenv("MERMAIDSEG_LOCAL_CACHE_DIR", str(cache_root))
    worker = mp.get_context("spawn").Process(target=read_cached_object_in_worker, args=(init_fn,))
    worker.start()
    try:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join()


def test_worker_init_counts_a_spawned_workers_cache_hit_in_the_parent(
    monkeypatch, tmp_path, spawn_cache_stats
):
    init_fn = make_worker_init_fn(spawn_cache_stats)
    run_in_spawned_worker(init_fn, tmp_path / "cache", monkeypatch)
    assert spawn_cache_stats.local_hits.value == 1
