"""Tests for the shared dataset build loop (mermaidseg.datasets.factory)."""

from __future__ import annotations

import pytest

from mermaidseg.datasets.factory import build_datasets
from mermaidseg.io import ConfigDict


class _RecordingDataset:
    """Stub dataset that records the kwargs it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 3


class _NoPaddingDataset(_RecordingDataset):
    def __init__(self, **kwargs):
        if "padding" in kwargs:
            raise TypeError("this dataset does not accept padding")
        super().__init__(**kwargs)


@pytest.fixture()
def stub_classes():
    return {"points": _RecordingDataset, "dense": _NoPaddingDataset}


def test_skips_none_and_string_none_splits(stub_classes, monkeypatch):
    monkeypatch.setattr(
        "mermaidseg.datasets.factory.NO_PADDING_DATASETS", frozenset({"dense"})
    )
    data_cfg = ConfigDict(
        {
            "points": {"train": {"whitelist": [1]}, "val": None, "test": "None"},
            "dense": {"train": {"split": "train"}},
        }
    )
    built = build_datasets(
        data_cfg, padding=7, dataset_classes=stub_classes, verbose=False
    )

    assert set(built) == {("points", "train"), ("dense", "train")}
    # Point dataset receives padding; dense (no-padding) dataset does not.
    assert built[("points", "train")].kwargs == {"whitelist": [1], "padding": 7}
    assert built[("dense", "train")].kwargs == {"split": "train"}


def test_names_and_splits_filters(stub_classes, monkeypatch):
    monkeypatch.setattr(
        "mermaidseg.datasets.factory.NO_PADDING_DATASETS", frozenset({"dense"})
    )
    data_cfg = ConfigDict(
        {
            "points": {"train": {"a": 1}, "val": {"b": 2}},
            "dense": {"train": {"split": "train"}},
        }
    )
    built = build_datasets(
        data_cfg,
        padding=0,
        dataset_classes=stub_classes,
        names=["points"],
        splits=["train"],
        verbose=False,
    )
    assert set(built) == {("points", "train")}


def test_missing_section_is_skipped(stub_classes):
    data_cfg = ConfigDict({"points": {"train": {"a": 1}}})  # no "dense" section
    built = build_datasets(
        data_cfg, padding=0, dataset_classes=stub_classes, verbose=False
    )
    assert set(built) == {("points", "train")}
