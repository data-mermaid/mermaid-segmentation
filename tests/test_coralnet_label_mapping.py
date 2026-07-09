"""Tests for the CoralNet provider_id -> MERMAID label mapping.

Guards the regression where CoralNet labels (numeric ``provider_id`` strings, e.g.
``"59"``) were looked up against a name-keyed mapping and silently collapsed to
background, so every CoralNet pixel mapped to ignore and CoralNet contributed nothing to
training/validation.
"""

from __future__ import annotations

import json

import requests

from mermaidseg.dataset_reconciliation import SourceLabelRegistry, label_mapping
from mermaidseg.dataset_reconciliation import registry as registry_mod


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - test stub
        return None

    def json(self) -> dict:
        return self._payload


def test_fetch_coralnet_to_mermaid_pages_api_and_keys_by_provider_id(monkeypatch):
    """Fetcher follows ``next`` pagination and keys by the numeric provider_id
    string."""
    pages = {
        "PAGE1": {
            "next": "PAGE2",
            "results": [
                {"provider_id": "59", "benthic_attribute_name": "Acropora"},
                {"provider_id": "58", "benthic_attribute_name": "Acanthastrea"},
            ],
        },
        "PAGE2": {
            "next": None,
            "results": [
                {"provider_id": "7462", "benthic_attribute_name": None},  # unmapped -> kept as None
            ],
        },
    }

    def fake_get(url, timeout=30):
        key = "PAGE1" if url == label_mapping._CORALNET_LABELMAPPINGS_URL else url
        return _FakeResponse(pages[key])

    monkeypatch.setattr(label_mapping.requests, "get", fake_get)

    mapping = label_mapping.fetch_coralnet_to_mermaid()

    assert mapping == {"59": "Acropora", "58": "Acanthastrea", "7462": None}
    # numeric keys (matches CoralNetDataset source_label_name = str(coralnet_id))
    assert all(k.isdigit() for k in mapping)


def test_fetch_coralnet_to_mermaid_falls_back_to_snapshot_on_api_failure(monkeypatch):
    """A network/HTTP failure falls back to the committed numeric-keyed snapshot."""

    def boom(url, timeout=30):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(label_mapping.requests, "get", boom)

    mapping = label_mapping.fetch_coralnet_to_mermaid()

    # The snapshot file exists, is non-empty, and is keyed by numeric provider ids.
    assert mapping, "expected non-empty fallback snapshot"
    assert all(k.isdigit() for k in mapping)
    on_disk = json.loads(label_mapping._CORALNET_SNAPSHOT_PATH.read_text())
    assert mapping == on_disk


def test_registry_resolves_numeric_coralnet_ids_to_targets(monkeypatch):
    """Regression guard: numeric CoralNet provider ids must resolve to real targets, not
    0.

    With the old name-keyed mapping, ``source_name2id`` keys like ``"59"`` matched
    nothing and ``source_to_target`` was all-background. Here we assert the numeric ids
    map through.
    """

    class CoralNetStub:
        SOURCE_NAME = "coralnet"
        # source_label_name == str(coralnet provider_id), exactly as CoralNetDataset emits.
        source_id2name = {1: "58", 2: "59"}
        source_name2id = {"58": 1, "59": 2}

        def set_global_offset(self, offset: int) -> None:
            self._global_offset = offset

    monkeypatch.setattr(
        registry_mod,
        "fetch_coralnet_to_mermaid",
        lambda: {"58": "Acanthastrea", "59": "Acropora", "9999": "Sand"},
    )

    registry = SourceLabelRegistry(
        [CoralNetStub()],
        target_label_subset=["Acanthastrea", "Acropora", "Background"],
        fetch_remote=True,
    )

    offset = registry.dataset_offsets["coralnet"]
    target_58 = int(registry.source_to_target[1 + offset])
    target_59 = int(registry.source_to_target[2 + offset])

    # Both numeric ids resolve to a non-background (non-zero) target id.
    assert target_58 != 0
    assert target_59 != 0
    assert target_58 != target_59
    assert registry.target_id2label[target_58] == "acanthastrea"
    assert registry.target_id2label[target_59] == "acropora"
