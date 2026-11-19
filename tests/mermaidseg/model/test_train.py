"""Tests for the epoch load-failure guard of ``mermaidseg.model.train``."""

from __future__ import annotations

from typing import Any

import pytest
import torch.distributed as dist

from mermaidseg.datasets.local_cache import LocalS3Cache, create_cache_stats
from mermaidseg.model.train import _enforce_load_failure_rate
from tests.mermaidseg.conftest import init_gloo, spawn_gloo

pytestmark = pytest.mark.usefixtures("reset_local_cache")


def _track_failures(count: int) -> None:
    cache = LocalS3Cache.get()
    cache.attach_stats(create_cache_stats())
    for _ in range(count):
        cache.record_load_failure()


def test_raises_when_epoch_rate_exceeds_max() -> None:
    _track_failures(6)
    with pytest.raises(RuntimeError, match="3/10"):
        _enforce_load_failure_rate(3, 7, 0.2, epoch=0)


def test_rate_at_max_does_not_raise_and_uses_epoch_delta() -> None:
    _track_failures(4)
    # Delta 1 over 10 attempts is exactly 0.1; the cumulative 4 over 13 would raise.
    _enforce_load_failure_rate(3, 9, 0.1, epoch=0)


def test_disabled_guard_does_not_raise() -> None:
    _track_failures(9)
    _enforce_load_failure_rate(0, 1, None, epoch=0)


def test_untracked_failures_do_not_raise() -> None:
    LocalS3Cache.get().record_load_failure()
    _enforce_load_failure_rate(None, 7, 0.0, epoch=0)


def test_zero_attempts_does_not_raise() -> None:
    _track_failures(0)
    _enforce_load_failure_rate(0, 0, 0.0, epoch=0)


def _guard_worker(
    rank: int,
    world_size: int,
    init_method: str,
    queue: Any,
    deltas: tuple[int, ...],
    num_loaded_samples: int,
    max_rate: float,
) -> None:
    init_gloo(rank, world_size, init_method)
    try:
        _track_failures(deltas[rank])
        try:
            _enforce_load_failure_rate(0, num_loaded_samples, max_rate, epoch=0)
            outcome = "ok"
        except RuntimeError:
            outcome = "raised"
        queue.put((rank, outcome))
    finally:
        dist.destroy_process_group()


def test_failures_on_one_rank_raise_on_every_rank() -> None:
    assert spawn_gloo(_guard_worker, 2, ((3, 0), 7, 0.2)) == ["raised", "raised"]


@pytest.mark.parametrize(
    ("max_rate", "expected"),
    [(0.15, ["raised", "raised"]), (0.25, ["ok", "ok"])],
)
def test_rate_is_summed_failures_over_summed_attempts(max_rate: float, expected: list[str]) -> None:
    # 2 failures over 8 samples + 2 failures = 0.2, whatever each rank's local share.
    assert spawn_gloo(_guard_worker, 2, ((1, 1), 8, max_rate)) == expected
