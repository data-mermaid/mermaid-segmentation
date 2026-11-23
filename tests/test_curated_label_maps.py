"""The curated CSVs carry the same name -> MERMAID target pairs as the mermaidseg dicts they replace."""

from __future__ import annotations

import pytest

from mermaidseg.dataset_reconciliation.label_mapping import (
    coralscapes_v2_to_mermaid,
    fetch_ucsd_mosaics_to_mermaid,
)
from segdata_etl.label_maps.curated import load_curated


def _csv_mapping(dataset: str) -> dict[str, str]:
    rows = load_curated(dataset).to_pylist()
    return {
        r["name"].lower(): r["mermaid_benthic_attribute"]
        for r in rows
        if r["mermaid_benthic_attribute"] is not None
    }


@pytest.mark.parametrize(
    ("dataset", "legacy"),
    [
        ("coralscapes_v2", coralscapes_v2_to_mermaid),
        ("ucsd_mosaics", fetch_ucsd_mosaics_to_mermaid),
    ],
)
def test_curated_csv_equals_the_mermaidseg_dict(dataset, legacy):
    assert _csv_mapping(dataset) == legacy()
