"""Characterization tests for SourceLabelRegistry._resolve_source_to_target_maps.

Lock the source->target dispatch behavior before collapsing the per-source if-ladder
into a table: the fetch_remote guard for remote sources, the provided-map short-circuit,
the built-in identity/static path, unknown-source errors, and output lowercasing. Pure
dispatch — no network, exercised via the unbound method with a stub `self`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry


class _StubDS:
    """Minimal dataset surface used by _resolve_source_to_target_maps."""

    def __init__(self, source_name: str, names: list[str]):
        self.SOURCE_NAME = source_name
        self.source_name2id = {n: i + 1 for i, n in enumerate(names)}


def _resolve(datasets, provided=None, fetch_remote=True):
    fake_self = SimpleNamespace(datasets=datasets)
    return SourceLabelRegistry._resolve_source_to_target_maps(
        fake_self, provided or {}, fetch_remote
    )


def test_provided_map_short_circuits_without_network():
    """An explicit provided map is used verbatim (lowercased) — no fetcher call, even
    for a remote source with fetch_remote=False."""
    ds = _StubDS("coralnet", ["1", "2"])
    result = _resolve([ds], provided={"coralnet": {"1": "Coral", "2": "Sand"}}, fetch_remote=False)
    assert result["coralnet"] == {"1": "coral", "2": "sand"}


def test_remote_source_without_fetch_remote_raises():
    """A remote source with no provided map requires fetch_remote=True."""
    ds = _StubDS("coralnet", ["1"])
    with pytest.raises(ValueError, match="coralnet"):
        _resolve([ds], fetch_remote=False)


def test_builtin_identity_source_resolves_without_network():
    """The built-in 'mermaid' identity map resolves without needing fetch_remote."""
    ds = _StubDS("mermaid", ["Coral", "Sand"])
    result = _resolve([ds], fetch_remote=False)
    assert result["mermaid"] == {"coral": "coral", "sand": "sand"}


def test_unknown_source_raises():
    """A SOURCE_NAME with no provided map, no fetcher, and no built-in entry errors."""
    ds = _StubDS("totally_unknown_dataset", ["x"])
    with pytest.raises(ValueError, match="No default source-to-target mapping"):
        _resolve([ds], fetch_remote=True)


def test_multiple_sources_resolve_independently():
    """A provided remote source and a built-in source resolve together, each keyed by
    SOURCE_NAME."""
    coralnet = _StubDS("coralnet", ["1"])
    mermaid = _StubDS("mermaid", ["Coral"])
    result = _resolve(
        [coralnet, mermaid], provided={"coralnet": {"1": "Coral"}}, fetch_remote=False
    )
    assert result == {"coralnet": {"1": "coral"}, "mermaid": {"coral": "coral"}}
