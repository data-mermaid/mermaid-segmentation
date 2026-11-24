"""The curated CSVs carry the same name -> MERMAID target pairs as the mermaidseg dicts they replace.

The CSVs are read from their path in the repo rather than through ``segdata_etl``,
because the mermaidseg test environment does not install that package.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mermaidseg.dataset_reconciliation.label_mapping import (
    coralscapes_v2_to_mermaid,
    fetch_ucsd_mosaics_to_mermaid,
)

CURATED_DIR = (
    Path(__file__).resolve().parents[1] / "packages/segdata-etl/src/segdata_etl/label_maps/curated"
)


def _csv_mapping(dataset: str) -> dict[str, str]:
    with (CURATED_DIR / f"{dataset}.csv").open(newline="") as f:
        return {
            row["name"].lower(): row["mermaid_benthic_attribute"]
            for row in csv.DictReader(f)
            if row["mermaid_benthic_attribute"]
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
