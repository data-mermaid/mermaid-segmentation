"""Tests for environment-variable configuration."""

from __future__ import annotations

import os

import pytest

from coralnet_scraper import config

_DEFAULT_PREFIX = "coralnet-public-images"


def test_get_bucket_returns_env_value(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", "b")
    assert config.get_bucket() == "b"


@pytest.mark.parametrize("value", [None, ""])
def test_get_bucket_raises_when_missing_or_empty(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MERMAID_CORALNET_BUCKET", raising=False)
    else:
        monkeypatch.setenv("MERMAID_CORALNET_BUCKET", value)
    with pytest.raises(ValueError, match="MERMAID_CORALNET_BUCKET"):
        config.get_bucket()


def test_get_prefix_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("MERMAID_CORALNET_PREFIX", raising=False)
    assert config.get_prefix() == _DEFAULT_PREFIX


def test_get_prefix_returns_env_value(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_PREFIX", "p")
    assert config.get_prefix() == "p"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("coralnet-public-images/", "coralnet-public-images"), ("/a/b/", "a/b")],
)
def test_get_prefix_strips_surrounding_slashes(monkeypatch, value, expected):
    monkeypatch.setenv("MERMAID_CORALNET_PREFIX", value)
    assert config.get_prefix() == expected


@pytest.mark.parametrize("value", ["", "/"])
def test_get_prefix_raises_when_empty_after_stripping(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_PREFIX", value)
    with pytest.raises(ValueError, match="MERMAID_CORALNET_PREFIX"):
        config.get_prefix()


def test_get_scraper_workers_unset_uses_cpu_based_default(monkeypatch):
    monkeypatch.delenv("MERMAID_CORALNET_SCRAPER_WORKERS", raising=False)
    expected = min(32, max(8, (os.cpu_count() or 4) * 4))
    assert config.get_scraper_workers() == expected


def test_get_scraper_workers_parses_integer(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_SCRAPER_WORKERS", "8")
    assert config.get_scraper_workers() == 8


@pytest.mark.parametrize("value", ["abc", "0", "-3"])
def test_get_scraper_workers_rejects_invalid(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_SCRAPER_WORKERS", value)
    with pytest.raises(ValueError, match="MERMAID_CORALNET_SCRAPER_WORKERS|workers"):
        config.get_scraper_workers()


def test_validate_workers_rejects_zero():
    with pytest.raises(ValueError, match="workers"):
        config.validate_workers(0)


@pytest.mark.parametrize("workers", [2.5, True, "4"])
def test_validate_workers_rejects_non_integer(workers):
    with pytest.raises(ValueError, match="workers"):
        config.validate_workers(workers)


def test_validate_workers_accepts_one():
    assert config.validate_workers(1) == 1
