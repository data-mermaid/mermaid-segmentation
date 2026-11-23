"""Tests for the cleaned_up_rerun CoralNet label map."""

from __future__ import annotations

import hashlib
from importlib import resources

import pyarrow as pa
import pytest

from segdata_etl.label_maps.coralnet_cleaned_up_rerun import cleaned_up_rerun_labels
from segdata_spec.schema import BENTHIC_ATTRIBUTES_SCHEMA, LABELS_SCHEMA

JSON_SHA256 = "05d67ac71e20a665357ee9ffbe3d4234f45c5533816d2e52cd805823f478ac0d"


def _taxonomy(*rows: tuple[str, str]) -> pa.Table:
    return pa.table(
        {
            "id": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "parent_id": [None] * len(rows),
        },
        schema=BENTHIC_ATTRIBUTES_SCHEMA,
    )


def _rows(table: pa.Table) -> dict[int, dict]:
    return {r["label_id"]: r for r in table.to_pylist()}


def test_json_sha256_is_pinned() -> None:
    data = (
        resources.files("segdata_etl.label_maps")
        .joinpath("coralnet_cleaned_up_rerun.json")
        .read_bytes()
    )
    assert hashlib.sha256(data).hexdigest() == JSON_SHA256


def test_mapped_label_resolves_id_case_insensitively() -> None:
    table, mapping = cleaned_up_rerun_labels(
        {1: "Acropora (branching)"}, _taxonomy(("ba-1", "acropora"))
    )
    assert table.schema == LABELS_SCHEMA
    assert table.to_pylist() == [
        {
            "label_id": 1,
            "name": "Acropora (branching)",
            "mermaid_benthic_attribute_id": "ba-1",
            "mermaid_benthic_attribute": "Acropora",
            "mermaid_growth_form": None,
            "mapping_source": "cleaned_up_rerun_json",
        }
    ]
    assert mapping.source == "cleaned_up_rerun_json"
    assert mapping.git_ref == "e0d3876"
    assert mapping.path == "configs/coralnet_to_mermaid_mapping_temporary.json"
    assert mapping.blob == "4f37f59ec8f57e26540f57adfa093830c8305f64"
    assert mapping.sha256 == JSON_SHA256


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("Coral Rubble", id="json-null"),
        pytest.param("not a coralnet label", id="absent-from-json"),
        pytest.param(None, id="none-name"),
    ],
)
def test_unmapped_cases_have_null_target_and_id(name: str | None) -> None:
    table, mapping = cleaned_up_rerun_labels({5: name}, _taxonomy(("ba-1", "Acropora")))
    row = _rows(table)[5]
    assert row["mermaid_benthic_attribute"] is None
    assert row["mermaid_benthic_attribute_id"] is None
    assert mapping.n_mapped == 0
    assert mapping.n_unresolved == 0


def test_target_absent_from_taxonomy_is_counted_unresolved() -> None:
    table, mapping = cleaned_up_rerun_labels({1: "Sand"}, _taxonomy(("ba-1", "Acropora")))
    row = _rows(table)[1]
    assert row["mermaid_benthic_attribute"] == "Sand"
    assert row["mermaid_benthic_attribute_id"] is None
    assert mapping.n_mapped == 1
    assert mapping.n_unresolved == 1


def test_ambiguous_taxonomy_name_raises() -> None:
    taxonomy = _taxonomy(("ba-1", "Porites"), ("ba-2", "porites"))
    with pytest.raises(ValueError, match="Porites"):
        cleaned_up_rerun_labels({1: "Porites"}, taxonomy)


def test_ambiguity_ignores_names_no_label_targets() -> None:
    taxonomy = _taxonomy(("ba-1", "Porites"), ("ba-2", "porites"), ("ba-3", "Sand"))
    table, _ = cleaned_up_rerun_labels({1: "Sand"}, taxonomy)
    assert _rows(table)[1]["mermaid_benthic_attribute_id"] == "ba-3"


def test_counts_cover_the_labels_passed() -> None:
    id2name = {1: "Acropora (branching)", 2: "Sand", 3: "Coral Rubble", 4: "zzz", 5: None}
    _, mapping = cleaned_up_rerun_labels(id2name, _taxonomy(("ba-1", "Acropora")))
    assert (mapping.n_labels, mapping.n_mapped, mapping.n_unresolved) == (5, 2, 1)
