"""Tests for morphology CSV lookup builders."""

from __future__ import annotations

from mermaidseg.dataset_reconciliation.morphology import (
    DEFAULT_MORPHOLOGY_CHANNELS,
    build_source_to_morphology_from_concept_csv,
)


def test_build_source_to_morphology_from_concept_csv_smoke():
    # global_id 1: a CoralNet row that exists in the committed CSV
    global_id2source = {
        1: ("coralnet", "acropora (branching)"),
        2: ("coralnet", "label-that-does-not-exist-xyz"),
    }
    table, channels = build_source_to_morphology_from_concept_csv(
        global_id2source, channels=DEFAULT_MORPHOLOGY_CHANNELS
    )
    assert channels == list(DEFAULT_MORPHOLOGY_CHANNELS)
    assert table.shape == (3, len(channels))
    assert table[0].abs().sum().item() == 0.0  # background
    assert table[2].abs().sum().item() == 0.0  # unknown source label
    # branching channel should be active (2) when the CSV marks branching TRUE
    branching_idx = channels.index("branching")
    assert table[1, branching_idx].item() in (0.0, 2.0, 1.0)
