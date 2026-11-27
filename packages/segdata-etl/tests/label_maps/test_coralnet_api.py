"""Tests for the CoralNet LabelMapping API fetch, parse, digest and label table."""

from __future__ import annotations

import pytest

from segdata_etl.label_maps.coralnet_api import (
    CoralNetLabelMapping,
    CoralNetLabelMappings,
    api_labels,
    coralnet_mapping_digest,
    fetch_coralnet_label_mappings,
    parse_coralnet_label_mappings,
)
from segdata_spec.schema import LABELS_SCHEMA

from .conftest import ScriptedServer

# The digest of _DIGEST_RECORDS, pinned so a change to the digest scheme is caught.
MAIN_DIGEST = "562cc2d876b238eda53a7274f7fc3e4191f1f84c8c9d382f936d37e050985b63"

_ACROPORA = CoralNetLabelMapping(
    id="a",
    benthic_attribute_id="b1",
    benthic_attribute_name="Acropora",
    growth_form_id="g1",
    growth_form_name="Branching",
    provider="CoralNet",
    provider_id=82,
    provider_label="Acropora (branching)",
)
_UNKNOWN = CoralNetLabelMapping(
    id="b",
    benthic_attribute_id=None,
    benthic_attribute_name=None,
    growth_form_id=None,
    growth_form_name=None,
    provider="CoralNet",
    provider_id=7,
    provider_label="Unknown",
)
_DIGEST_RECORDS = {82: _ACROPORA, 7: _UNKNOWN}

PATH = "/v1/labelmappings/"


def _row(provider_id: object, **extra: object) -> dict:
    return {"id": f"id-{provider_id}", "provider_id": provider_id, **extra}


def _mappings(records: dict[int, CoralNetLabelMapping]) -> CoralNetLabelMappings:
    return CoralNetLabelMappings(
        records=records, source="api", fetched_at="2026-11-23T10:00:00+00:00"
    )


def test_fetch_follows_next_and_keys_records_by_int(server: ScriptedServer) -> None:
    server.respond(
        PATH,
        lambda base: (200, {"results": [_row("82.0")], "next": f"{base}{PATH}?page=2"}),
    )
    server.respond(f"{PATH}?page=2", (200, {"results": [_row(7)], "next": None}))
    mappings = fetch_coralnet_label_mappings(server.base + PATH)
    assert sorted(mappings.records) == [7, 82]
    assert mappings.records[82]["provider_id"] == 82
    assert mappings.source == "api"


def test_fetch_recovers_from_one_503(server: ScriptedServer) -> None:
    server.respond(PATH, (503, {}), (200, {"results": [_row(1)], "next": None}))
    mappings = fetch_coralnet_label_mappings(server.base + PATH)
    assert list(mappings.records) == [1]
    assert server.hits[PATH] == 2


def test_fetch_raises_on_a_repeated_next_url(server: ScriptedServer) -> None:
    server.respond(
        PATH,
        lambda base: (200, {"results": [_row(1)], "next": f"{base}{PATH}?page=2"}),
    )
    server.respond(
        f"{PATH}?page=2",
        lambda base: (200, {"results": [_row(2)], "next": f"{base}{PATH}"}),
    )
    with pytest.raises(ValueError, match="Pagination loop"):
        fetch_coralnet_label_mappings(server.base + PATH)


def test_parse_rejects_duplicate_provider_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate.*82"):
        parse_coralnet_label_mappings([_row(82), _row("82.0")])


def test_parse_rejects_non_integer_provider_id() -> None:
    with pytest.raises(ValueError, match="non-integer"):
        parse_coralnet_label_mappings([_row("82.5")])


def test_digest_matches_mermaidseg() -> None:
    assert coralnet_mapping_digest(_DIGEST_RECORDS) == MAIN_DIGEST


def test_api_labels_carry_attribute_growth_form_and_unmapped_ids() -> None:
    table, mapping = api_labels(
        {82: "Acropora (branching)", 7: "Unknown", 99: "No record", 5: None},
        _mappings(_DIGEST_RECORDS),
        {"b1"},
        {"g1"},
    )
    assert table.schema == LABELS_SCHEMA
    rows = {r["label_id"]: r for r in table.to_pylist()}
    assert rows[82] == {
        "label_id": 82,
        "name": "Acropora (branching)",
        "mermaid_benthic_attribute_id": "b1",
        "mermaid_benthic_attribute": "Acropora",
        "mermaid_growth_form": "Branching",
        "mapping_source": "mermaid_api",
    }
    for unmapped in (7, 99, 5):
        assert rows[unmapped]["mermaid_benthic_attribute"] is None
        assert rows[unmapped]["mermaid_benthic_attribute_id"] is None
        assert rows[unmapped]["mermaid_growth_form"] is None
    assert mapping.source == "mermaid_api"
    assert mapping.sha256 == MAIN_DIGEST
    assert mapping.fetched_at is not None
    assert (mapping.n_labels, mapping.n_mapped) == (4, 1)


def test_api_labels_treat_a_label_without_a_name_as_unmapped_even_with_a_record() -> None:
    table, mapping = api_labels({82: None}, _mappings(_DIGEST_RECORDS), {"b1"}, {"g1"})
    row = table.to_pylist()[0]
    assert row["mermaid_benthic_attribute"] is None
    assert row["mermaid_benthic_attribute_id"] is None
    assert row["mermaid_growth_form"] is None
    assert (mapping.n_labels, mapping.n_mapped) == (1, 0)


def test_api_labels_raise_listing_every_attribute_id_absent_from_the_taxonomy() -> None:
    def record(provider_id: int, attribute_id: str) -> CoralNetLabelMapping:
        return _ACROPORA | {"provider_id": provider_id, "benthic_attribute_id": attribute_id}  # type: ignore[return-value]

    records = {1: record(1, "zz"), 2: record(2, "aa"), 3: record(3, "zz"), 4: record(4, "b1")}
    with pytest.raises(ValueError, match=r"\['aa', 'zz'\]"):
        api_labels({i: f"L{i}" for i in records}, _mappings(records), {"b1"}, {"g1"})


def test_api_labels_ignore_an_absent_attribute_id_on_an_unmapped_record() -> None:
    absent = _ACROPORA | {"benthic_attribute_id": "gone", "benthic_attribute_name": None}
    _, mapping = api_labels({82: "Acropora"}, _mappings({82: absent}), set(), set())  # type: ignore[dict-item]
    assert mapping.n_mapped == 0


def test_api_labels_raise_listing_every_mapped_label_with_no_attribute_id() -> None:
    no_id = _ACROPORA | {"benthic_attribute_id": None}
    records = {5: no_id, 3: no_id, 82: _ACROPORA}
    with pytest.raises(ValueError, match=r"no benthic_attribute_id: \[3, 5\]"):
        api_labels({i: f"L{i}" for i in records}, _mappings(records), {"b1"}, {"g1"})  # type: ignore[arg-type]


def test_api_labels_raise_listing_every_growth_form_id_absent_from_the_taxonomy() -> None:
    records = {
        1: _ACROPORA | {"provider_id": 1, "growth_form_id": "gz"},
        2: _ACROPORA | {"provider_id": 2, "growth_form_id": "ga"},
        3: _ACROPORA | {"provider_id": 3, "growth_form_id": None},
        82: _ACROPORA,
    }
    with pytest.raises(
        ValueError, match=r"growth_form_id not in the pinned taxonomy: \['ga', 'gz'\]"
    ):
        api_labels({i: f"L{i}" for i in records}, _mappings(records), {"b1"}, {"g1"})  # type: ignore[arg-type]
