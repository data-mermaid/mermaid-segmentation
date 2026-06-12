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


def test_legacy_fallback_when_no_env(monkeypatch, fresh_module):
    monkeypatch.delenv("MERMAID_CORALNET_ANNOTATIONS_PATH", raising=False)
    monkeypatch.delenv("MERMAID_CORALNET_ANNOTATIONS_VERSION", raising=False)
    mod = fresh_module()
    assert mod._resolve_default_annotations_path() == "coralnet_annotations_30112025.parquet"


def test_version_env_builds_versioned_filename(monkeypatch, fresh_module):
    monkeypatch.delenv("MERMAID_CORALNET_ANNOTATIONS_PATH", raising=False)
    monkeypatch.setenv("MERMAID_CORALNET_ANNOTATIONS_VERSION", "20260515_deadbeef")
    mod = fresh_module()
    assert (
        mod._resolve_default_annotations_path() == "coralnet_annotations_20260515_deadbeef.parquet"
    )


def test_explicit_path_env_wins(monkeypatch, fresh_module):
    monkeypatch.setenv("MERMAID_CORALNET_ANNOTATIONS_PATH", "custom/path.parquet")
    monkeypatch.setenv("MERMAID_CORALNET_ANNOTATIONS_VERSION", "ignored")
    mod = fresh_module()
    assert mod._resolve_default_annotations_path() == "custom/path.parquet"
