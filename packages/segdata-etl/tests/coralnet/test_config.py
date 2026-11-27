"""Tests for the dataset build's environment configuration."""

from __future__ import annotations

import pytest

from segdata_etl.datasets.coralnet import config


@pytest.mark.parametrize("value", ["abc", "0"])
def test_get_workers_rejects_invalid_env(monkeypatch, value):
    monkeypatch.setenv("MERMAID_CORALNET_ETL_WORKERS", value)
    with pytest.raises(ValueError, match=f"MERMAID_CORALNET_ETL_WORKERS .*{value!r}"):
        config.get_workers()


@pytest.mark.parametrize(("args", "expected"), [((), 16)], ids=["module-default"])
def test_get_workers_falls_back_to_the_default_when_unset(monkeypatch, args, expected):
    monkeypatch.delenv("MERMAID_CORALNET_ETL_WORKERS", raising=False)
    assert config.get_workers(*args) == expected


def test_get_workers_returns_env_value(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_ETL_WORKERS", "3")
    assert config.get_workers() == 3
