"""The curated CSVs carry the same name -> MERMAID target pairs as the mermaidseg dicts they replace.

The CSVs are read from their path in the repo rather than through ``segdata_etl``,
because the mermaidseg test environment does not install that package.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from mermaidseg.dataset_reconciliation.label_mapping import (
    coralscapes_v2_to_mermaid,
    fetch_benthos_yuval_to_mermaid,
    fetch_catlin_seaview_to_mermaid,
    fetch_moorea_labeled_corals_to_mermaid,
    fetch_pacific_labeled_corals_to_mermaid,
    fetch_ucsd_mosaics_to_mermaid,
)

CURATED_DIR = (
    Path(__file__).resolve().parents[1] / "packages/segdata-etl/src/segdata_etl/label_maps/curated"
)


def _mapped_rows(dataset: str) -> list[dict[str, str]]:
    with (CURATED_DIR / f"{dataset}.csv").open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["mermaid_benthic_attribute"]]


@pytest.mark.parametrize(
    ("dataset", "legacy"),
    [
        ("coralscapes_v2", coralscapes_v2_to_mermaid),
        ("ucsd_mosaics", fetch_ucsd_mosaics_to_mermaid),
        ("catlin_seaview", fetch_catlin_seaview_to_mermaid),
        ("moorea_labeled_corals", fetch_moorea_labeled_corals_to_mermaid),
        ("pacific_labeled_corals", fetch_pacific_labeled_corals_to_mermaid),
        ("benthos_yuval", fetch_benthos_yuval_to_mermaid),
    ],
)
def test_curated_csv_equals_the_mermaidseg_dict(
    dataset: str, legacy: Callable[[], dict[str, str]]
) -> None:
    rows = _mapped_rows(dataset)
    expected = legacy()
    assert {row["name"].lower(): row["mermaid_benthic_attribute"] for row in rows} == expected
    assert len(rows) == len(expected)
