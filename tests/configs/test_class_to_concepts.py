"""Invariants of the authoritative configs/class_to_concepts.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_CSV = _REPO / "configs" / "class_to_concepts.csv"
_DATA_CONFIG = _REPO / "configs" / "data_config.yaml"


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(_CSV, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def configured_datasets() -> set[str]:
    return set(yaml.safe_load(_DATA_CONFIG.read_text())["data"]["datasets"])


# Sediment is sand with rubble left unknown. Branching and calcareous algae has
# unknown taxonomy and rubble left unknown. The derived rule would force FALSE.
_RUBBLE_NOT_GIVEN = {
    ("sediment", "coralnet"),
    ("sediment", "catlin_seaview"),
    ("branching and calcareous algae", "coralnet"),
}


def _expected_rubble(row: pd.Series) -> str:
    key = (row["source_label_class_name"], row["source_dataset_source"])
    if key in _RUBBLE_NOT_GIVEN:
        return "not_given"
    if row["kingdom"] != "none":
        return "FALSE"
    if "TRUE" in (row["anthropogenic"], row["background"], row["dark"]):
        return "FALSE"
    names = f"{row['source_label_class_name']} {row['target_label_class_name']}"
    if "rubble" in names.lower():
        return "TRUE"
    if "TRUE" in (row["sand"], row["hard_substrate"]):
        return "FALSE"
    return "not_given"


def test_rubble_column_follows_the_rules(table):
    assert "rubble" in table.columns
    assert set(table["rubble"]) <= {"TRUE", "FALSE", "not_given"}
    expected = table.apply(_expected_rubble, axis=1)
    mismatched = table.loc[expected != table["rubble"], "source_label_class_name"]
    assert mismatched.empty, mismatched.tolist()


def test_rubble_rows_are_neither_sand_nor_hard_substrate(table):
    rubble = table[table["rubble"] == "TRUE"]
    assert len(rubble) > 0
    assert (rubble["sand"] == "FALSE").all()
    assert (rubble["hard_substrate"] == "FALSE").all()


def test_every_source_is_a_configured_dataset(table, configured_datasets):
    sources = set(table["source_dataset_source"])
    assert sources <= configured_datasets, sorted(sources - configured_datasets)
