"""Tests for the curated label map validation and provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from segdata_etl.label_maps.curated import CuratedSource, curated_source, load_curated

_HEADER = "label_id,name,mermaid_benthic_attribute\n"


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("1,Sand,sand\n1,Rock,rock\n", "duplicate label_id 1"),
        (",cca,crustose coralline algae\n,cca,turf algae\n", "duplicate name 'cca'"),
        (",CCA,crustose coralline algae\n", "'CCA' is not normalized"),
        (", cca,crustose coralline algae\n", "' cca' is not normalized"),
        (",,crustose coralline algae\n", "neither label_id nor name"),
    ],
)
def test_load_curated_refuses_a_malformed_map(tmp_path: Path, body: str, match: str) -> None:
    (tmp_path / "catlin_seaview.csv").write_text(_HEADER + body)
    with pytest.raises(ValueError, match=f"catlin_seaview.*{match}"):
        load_curated("catlin_seaview", root=tmp_path)


def test_curated_source_hashes_the_raw_csv_bytes(tmp_path: Path) -> None:
    data = b"label_id,name,mermaid_benthic_attribute\n,cca,cca\n"
    (tmp_path / "demo.csv").write_bytes(data)

    source = curated_source("demo", root=tmp_path)

    assert isinstance(source, CuratedSource)
    assert source.sha256 == hashlib.sha256(data).hexdigest()
    assert source.path.endswith("demo.csv")
