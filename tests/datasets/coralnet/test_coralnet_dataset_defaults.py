"""Verify env-driven default-path resolution for CoralNetDataset."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fresh_module(monkeypatch):
    """Reload the dataset module after env tweaks so the literal default is re-evaluated."""

    def _reload():
        import mermaidseg.datasets.coralnet.coralnet_dataset as module

        return importlib.reload(module)

    return _reload


def test_default_fallback_when_no_env(monkeypatch, fresh_module):
    monkeypatch.delenv("MERMAID_CORALNET_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("MERMAID_CORALNET_MANIFEST_VERSION", raising=False)
    mod = fresh_module()
    assert mod._resolve_default_manifest_path() == mod._DEFAULT_MANIFEST_PATH


def test_version_env_builds_versioned_filename(monkeypatch, fresh_module):
    monkeypatch.delenv("MERMAID_CORALNET_MANIFEST_PATH", raising=False)
    monkeypatch.setenv("MERMAID_CORALNET_MANIFEST_VERSION", "20260515_deadbeef")
    mod = fresh_module()
    assert (
        mod._resolve_default_manifest_path()
        == "etl-outputs/coralnet/20260515_deadbeef/coralnet_training_manifest_20260515_deadbeef.parquet"
    )


def test_explicit_path_env_wins(monkeypatch, fresh_module):
    monkeypatch.setenv("MERMAID_CORALNET_MANIFEST_PATH", "custom/path.parquet")
    monkeypatch.setenv("MERMAID_CORALNET_MANIFEST_VERSION", "ignored")
    mod = fresh_module()
    assert mod._resolve_default_manifest_path() == "custom/path.parquet"
