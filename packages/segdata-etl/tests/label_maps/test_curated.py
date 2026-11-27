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


@pytest.mark.parametrize(
    ("dataset", "key", "target"),
    [
        pytest.param("benthos_yuval", "coral", "hard coral", id="benthos_yuval"),
        pytest.param("catlin_seaview", "acervicornis", "acropora cervicornis", id="catlin_seaview"),
        pytest.param("coralscapes_v2", 2, "acropora", id="coralscapes_v2"),
        pytest.param("moorea_labeled_corals", "acrop", "acropora", id="moorea_labeled_corals"),
        pytest.param(
            "pacific_labeled_corals",
            "bare substrate",
            "bare substrate",
            id="pacific_labeled_corals",
        ),
        pytest.param("ucsd_mosaics", 1, "acropora", id="ucsd_mosaics"),
    ],
)
def test_shipped_curated_map_loads(dataset: str, key: int | str, target: str) -> None:
    rows = load_curated(dataset).to_pylist()

    assert rows
    if isinstance(key, int):
        matches = [row for row in rows if row["label_id"] == key]
    else:
        matches = [row for row in rows if row["name"] == key]
    assert len(matches) == 1
    assert matches[0]["mermaid_benthic_attribute"] == target
