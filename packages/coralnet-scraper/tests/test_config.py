"""Tests for environment-variable configuration."""

from __future__ import annotations

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


def test_get_scraper_workers_parses_integer(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_SCRAPER_WORKERS", "8")
    assert config.get_scraper_workers() == 8


@pytest.mark.parametrize("value", ["abc", "0"])
def test_get_scraper_workers_rejects_invalid(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_SCRAPER_WORKERS", value)
    with pytest.raises(ValueError, match="MERMAID_CORALNET_SCRAPER_WORKERS|workers"):
        config.get_scraper_workers()


def test_validate_workers_rejects_zero():
    with pytest.raises(ValueError, match="workers"):
        config.validate_workers(0)


@pytest.mark.parametrize("workers", [2.5, True])
def test_validate_workers_rejects_non_integer(workers):
    with pytest.raises(ValueError, match="workers"):
        config.validate_workers(workers)


def test_validate_workers_accepts_one():
    assert config.validate_workers(1) == 1


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("2.5", 2.5)])
def test_get_requests_per_second_reads_the_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MERMAID_CORALNET_REQUESTS_PER_SECOND", raising=False)
    else:
        monkeypatch.setenv("MERMAID_CORALNET_REQUESTS_PER_SECOND", value)
    assert config.get_requests_per_second() == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_get_requests_per_second_rejects_non_positive_or_non_numeric(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_REQUESTS_PER_SECOND", value)
    with pytest.raises(
        ValueError, match="MERMAID_CORALNET_REQUESTS_PER_SECOND|requests per second"
    ):
        config.get_requests_per_second()


@pytest.mark.parametrize(("value", "expected"), [(None, 3), ("4", 4)])
def test_get_export_workers_reads_the_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MERMAID_CORALNET_EXPORT_WORKERS", raising=False)
    else:
        monkeypatch.setenv("MERMAID_CORALNET_EXPORT_WORKERS", value)
    assert config.get_export_workers() == expected


@pytest.mark.parametrize("value", ["0", "x"])
def test_get_export_workers_rejects_invalid(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_EXPORT_WORKERS", value)
    with pytest.raises(ValueError, match="MERMAID_CORALNET_EXPORT_WORKERS|workers"):
        config.get_export_workers()
