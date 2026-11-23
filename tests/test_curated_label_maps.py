"""Checks that each curated CSV under segdata-etl equals its ``label_mapping.py`` dict.

Each CSV's native-name to MERMAID benthic-attribute pairs and row count must match the dict
returned by the matching ``fetch_*_to_mermaid`` function. Reads the CSVs by file path because the
root environment does not install segdata-etl and mermaidseg must not import segdata_etl.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from mermaidseg.dataset_reconciliation import label_mapping

_CURATED = (
    Path(__file__).resolve().parent.parent
    / "packages"
    / "segdata-etl"
    / "src"
    / "segdata_etl"
    / "label_maps"
    / "curated"
)

_DICTS: dict[str, Callable[[], dict[str, str]]] = {
    "catlin_seaview": label_mapping.fetch_catlin_seaview_to_mermaid,
    "moorea_labeled_corals": label_mapping.fetch_moorea_labeled_corals_to_mermaid,
    "pacific_labeled_corals": label_mapping.fetch_pacific_labeled_corals_to_mermaid,
    "benthos_yuval": label_mapping.fetch_benthos_yuval_to_mermaid,
}


@pytest.mark.parametrize("dataset", sorted(_DICTS))
def test_csv_equals_the_label_mapping_dict(dataset: str) -> None:
    with (_CURATED / f"{dataset}.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    expected = _DICTS[dataset]()
    assert {r["native_name"]: r["mermaid_benthic_attribute"] for r in rows} == expected
    assert len(rows) == len(expected)
