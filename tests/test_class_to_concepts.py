"""Invariants of the authoritative configs/class_to_concepts.csv."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.datasets.factory import configured_source_names

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "configs" / "class_to_concepts.csv"
_TAXONOMY = ["kingdom", "phylum", "order", "family", "genus", "species"]


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(_CSV, dtype=str, keep_default_na=False)


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


def test_only_the_authoritative_table_exists():
    assert _CSV.is_file()
    assert not (_REPO / "configs" / "class_to_concepts_final.csv").exists()


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


def test_rubble_rows_keep_their_other_concepts(table):
    turf = table[table["source_label_class_name"] == "turf on rubble"]
    assert (turf["rubble"] == "TRUE").all() and (turf["algae"] == "TRUE").all()


def test_red_algae_unidentified_encrusting_is_rhodophyta(table):
    rows = table[table["source_label_class_name"] == "red algae, unidentified encrusting"]
    assert len(rows) >= 1
    for _, row in rows.iterrows():
        assert row["kingdom"] == "plantae" and row["phylum"] == "rhodophyta"
        assert [row[rank] for rank in _TAXONOMY[2:]] == ["not_given"] * 4


def test_chondrilla_is_live(table):
    rows = table[table["source_label_class_name"].str.startswith("chondrilla")]
    assert len(rows) == 3
    assert (rows["live"] == "TRUE").all()


def test_schema_layout_matches_demo():
    data = yaml.safe_load((_REPO / "configs" / "data_config.yaml").read_text())["data"]
    schema = ConceptSchema.from_csv(_CSV, sources=configured_source_names(data))
    demo = json.loads((_REPO / "demo" / "concept_id2name.json").read_text())
    names = {str(channel_id): name for channel_id, name in schema.channel_id2name().items()}
    assert names == demo
    assert "rubble" in demo.values()
