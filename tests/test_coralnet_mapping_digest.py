"""Checks mermaidseg's CoralNet label-mapping digest and snapshot against fixed golden values.

Counterpart of ``packages/segdata-etl/tests/label_maps/test_coralnet_api.py``, which pins
segdata-etl's copy of the parse and digest to the same ``MAIN_DIGEST`` for the same rows. The rows
and the digest are copied as literals because the root environment does not install segdata-etl
and mermaidseg must not import segdata_etl; a change to either copy breaks one of the two tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from mermaidseg.dataset_reconciliation.label_mapping import (
    CoralNetLabelMapping,
    CoralNetLabelMappings,
    coralnet_mapping_digest,
    parse_coralnet_label_mappings,
    read_coralnet_snapshot,
    write_coralnet_snapshot,
)

MAIN_DIGEST = "562cc2d876b238eda53a7274f7fc3e4191f1f84c8c9d382f936d37e050985b63"

_ACROPORA: CoralNetLabelMapping = {
    "id": "a",
    "benthic_attribute_id": "b1",
    "benthic_attribute_name": "Acropora",
    "growth_form_id": "g1",
    "growth_form_name": "Branching",
    "provider": "CoralNet",
    "provider_id": 82,
    "provider_label": "Acropora (branching)",
}
_UNKNOWN: CoralNetLabelMapping = {
    "id": "b",
    "benthic_attribute_id": None,
    "benthic_attribute_name": None,
    "growth_form_id": None,
    "growth_form_name": None,
    "provider": "CoralNet",
    "provider_id": 7,
    "provider_label": "Unknown",
}
_DIGEST_RECORDS = {82: _ACROPORA, 7: _UNKNOWN}
_FETCHED_AT = "2026-11-23T10:00:00+00:00"


def _mappings() -> CoralNetLabelMappings:
    return CoralNetLabelMappings(records=_DIGEST_RECORDS, source="api", fetched_at=_FETCHED_AT)


def test_digest_matches_segdata_etl() -> None:
    assert coralnet_mapping_digest(_DIGEST_RECORDS) == MAIN_DIGEST


def test_parse_of_the_digest_rows_gives_the_same_digest() -> None:
    parsed = parse_coralnet_label_mappings([_ACROPORA, _UNKNOWN])
    assert coralnet_mapping_digest(parsed) == MAIN_DIGEST


def test_snapshot_json_has_the_snapshot_shape(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_coralnet_snapshot(_mappings(), path)
    data = json.loads(path.read_text())
    assert data["fetched_at"] == _FETCHED_AT
    assert data["sha256"] == MAIN_DIGEST
    assert [r["provider_id"] for r in data["records"]] == [7, 82]


def test_snapshot_reads_back_with_the_same_digest(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_coralnet_snapshot(_mappings(), path)
    assert read_coralnet_snapshot(path).sha256 == MAIN_DIGEST
