"""Tests for the curated native-label to MERMAID maps."""

from __future__ import annotations

import hashlib
from importlib import resources

import pytest

from segdata_etl.label_maps import curated
from segdata_etl.label_maps.curated import CuratedMapInfo, load_curated


def test_pacific_map_has_a_known_entry_and_the_csv_sha256() -> None:
    mapping, info = load_curated("pacific_labeled_corals")
    csv_bytes = (
        resources.files("segdata_etl.label_maps")
        .joinpath("curated/pacific_labeled_corals.csv")
        .read_bytes()
    )

    assert mapping["other scleractinians"] == "hard coral"
    assert info == CuratedMapInfo(
        path="packages/segdata-etl/src/segdata_etl/label_maps/curated/pacific_labeled_corals.csv",
        sha256=hashlib.sha256(csv_bytes).hexdigest(),
        n_entries=len(mapping),
    )


def test_unknown_dataset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no_such_dataset"):
        load_curated("no_such_dataset")


class _StubFiles:
    def __init__(self, data: bytes):
        self._data = data

    def joinpath(self, _name: str) -> _StubFiles:
        return self

    def read_bytes(self) -> bytes:
        return self._data


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("cca,crustose coralline algae\ncca,turf algae", "duplicate native_name 'cca'"),
        ("CCA,crustose coralline algae", "'CCA' is not normalized"),
        (" cca,crustose coralline algae", "' cca' is not normalized"),
        ("cca,", "'cca' has an empty target"),
    ],
)
def test_load_curated_refuses_a_malformed_map(
    monkeypatch: pytest.MonkeyPatch, body: str, match: str
) -> None:
    data = f"native_name,mermaid_benthic_attribute\n{body}\n".encode()
    monkeypatch.setattr(curated.resources, "files", lambda _pkg: _StubFiles(data))
    with pytest.raises(ValueError, match=f"catlin_seaview.*{match}"):
        load_curated("catlin_seaview")
