"""prefetch_cache: the cache root comes from --cache-dir, never from the committed config."""

from __future__ import annotations

import logging

import pytest

import scripts.prefetch_cache as prefetch
from mermaidseg.datasets import setup_local_cache
from mermaidseg.datasets.local_cache import LocalS3Cache


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    # setup_local_cache exports the root to the environment; keep that out of other tests.
    # setenv first so monkeypatch records a value to restore even when the var is absent.
    for var in ("MERMAIDSEG_LOCAL_CACHE_DIR", "MERMAIDSEG_S3_OFFLINE"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    LocalS3Cache.reset_singleton()
    monkeypatch.setattr(prefetch, "build_datasets", lambda *a, **k: {})
    yield
    LocalS3Cache.reset_singleton()


def test_cache_dir_flag_reaches_setup_local_cache(monkeypatch, tmp_path):
    seen = []

    def spy(data):
        seen.append(data.get("local_cache_dir"))
        return setup_local_cache(data)

    monkeypatch.setattr(prefetch, "setup_local_cache", spy)

    prefetch.main(["--cache-dir", str(tmp_path)])

    assert seen == [str(tmp_path)]


def test_missing_cache_dir_exits_with_not_configured_error(caplog):
    with caplog.at_level(logging.ERROR, logger="prefetch_cache"):
        code = prefetch.main([])

    assert code == 2
    assert "Local cache is not configured" in caplog.text
