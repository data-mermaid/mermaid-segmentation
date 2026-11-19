"""Tests for the dataset build's environment configuration."""

from __future__ import annotations

import pytest

from coralnet_dataset import config


@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_get_workers_rejects_invalid_env(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_ETL_WORKERS", value)
    with pytest.raises(ValueError, match=f"MERMAID_CORALNET_ETL_WORKERS .*{value!r}"):
        config.get_workers()


def test_get_workers_returns_env_value(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_ETL_WORKERS", "3")
    assert config.get_workers() == 3


def test_get_resized_prefix_strips_slashes(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_RESIZED_PREFIX", "/x/y/")
    assert config.get_resized_prefix() == "x/y"


@pytest.mark.parametrize("value", ["", "/"])
def test_get_resized_prefix_rejects_empty(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_RESIZED_PREFIX", value)
    with pytest.raises(ValueError, match="MERMAID_CORALNET_RESIZED_PREFIX"):
        config.get_resized_prefix()


def test_get_resized_prefix_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("MERMAID_CORALNET_RESIZED_PREFIX", raising=False)
    assert config.get_resized_prefix() == "dev/images"
