"""Tests for the CoralNet LabelMapping API fetch, parse, digest and label table."""

from __future__ import annotations

import json

import pytest
import requests

from segdata_etl.label_maps.coralnet_api import (
    CoralNetLabelMapping,
    CoralNetLabelMappings,
    api_labels,
    coralnet_mapping_digest,
    fetch_coralnet_label_mappings,
    parse_coralnet_label_mappings,
    snapshot_json_text,
)
from segdata_spec.schema import LABELS_SCHEMA

from .conftest import ScriptedServer

# The digest mermaidseg's coralnet_mapping_digest gives for _DIGEST_RECORDS.
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


def test_fetch_raises_after_three_retries_on_persistent_503(server: ScriptedServer) -> None:
    server.respond(PATH, (503, {}))
    with pytest.raises(requests.RequestException):
        fetch_coralnet_label_mappings(server.base + PATH)
    assert server.hits[PATH] == 4  # the first attempt plus 3 retries


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
    table, mapping = api_labels({82: None}, _mappings(_DIGEST_RECORDS))
    row = table.to_pylist()[0]
    assert row["mermaid_benthic_attribute"] is None
    assert row["mermaid_benthic_attribute_id"] is None
    assert row["mermaid_growth_form"] is None
    assert (mapping.n_labels, mapping.n_mapped) == (1, 0)


def test_snapshot_json_text_has_the_snapshot_shape() -> None:
    mappings = _mappings(_DIGEST_RECORDS)
    data = json.loads(snapshot_json_text(mappings))
    assert data["fetched_at"] == "2026-11-23T10:00:00+00:00"
    assert data["sha256"] == MAIN_DIGEST
    assert [r["provider_id"] for r in data["records"]] == [7, 82]
