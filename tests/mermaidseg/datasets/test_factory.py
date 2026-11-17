"""Tests for the shared dataset build loop (mermaidseg.datasets.factory)."""

from __future__ import annotations

import albumentations as A
import pytest

from mermaidseg.datasets.factory import build_datasets, configured_source_names
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


def test_skips_none_splits(stub_classes, monkeypatch):
    monkeypatch.setattr("mermaidseg.datasets.factory.NO_PADDING_DATASETS", frozenset({"dense"}))
    data_cfg = ConfigDict(
        {
            "points": {"train": {"whitelist": [1]}, "val": None},
            "dense": {"train": {"split": "train"}},
        }
    )
    built = build_datasets(data_cfg, padding=7, dataset_classes=stub_classes, verbose=False)

    assert set(built) == {("points", "train"), ("dense", "train")}
    # Point dataset receives padding; dense (no-padding) dataset does not.
    assert built[("points", "train")].kwargs == {"whitelist": [1], "padding": 7}
    assert built[("dense", "train")].kwargs == {"split": "train"}


def test_transform_spec_reaches_the_dataset_compiled_in_order(stub_classes):
    data_cfg = ConfigDict(
        {
            "points": {
                "val": {
                    "transform": {
                        "Resize": {"width": 8, "height": 8},
                        "Normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
                    }
                }
            }
        }
    )
    built = build_datasets(data_cfg, padding=0, dataset_classes=stub_classes, verbose=False)
    transform = built[("points", "val")].kwargs["transform"]
    assert isinstance(transform, A.Compose)
    assert [type(t) for t in transform.transforms] == [A.Resize, A.Normalize]


@pytest.mark.parametrize("build", ["build_datasets", "configured_source_names"])
def test_unknown_data_key_raises(stub_classes, build):
    data_cfg = ConfigDict({"points": {"train": {"a": 1}}, "coralscapes": {"train": None}})
    with pytest.raises(ValueError, match="coralscapes"):
        if build == "build_datasets":
            build_datasets(data_cfg, padding=0, dataset_classes=stub_classes, verbose=False)
        else:
            configured_source_names(data_cfg, dataset_classes=stub_classes)


def test_reserved_data_keys_are_not_datasets(stub_classes):
    data_cfg = ConfigDict(
        {
            "local_cache_write_through": True,
            "local_cache_dir": "/cache",
            "points": {"train": {"a": 1}},
        }
    )
    built = build_datasets(data_cfg, padding=0, dataset_classes=stub_classes, verbose=False)
    assert set(built) == {("points", "train")}


def test_names_and_splits_filters(stub_classes, monkeypatch):
    monkeypatch.setattr("mermaidseg.datasets.factory.NO_PADDING_DATASETS", frozenset({"dense"}))
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
    built = build_datasets(data_cfg, padding=0, dataset_classes=stub_classes, verbose=False)
    assert set(built) == {("points", "train")}


def test_repeat_key_is_not_passed_to_the_dataset_constructor(stub_classes):
    data_cfg = ConfigDict({"points": {"train": {"whitelist": [1], "repeat": 5}}})
    built = build_datasets(data_cfg, padding=7, dataset_classes=stub_classes, verbose=False)
    assert built[("points", "train")].kwargs == {"whitelist": [1], "padding": 7}


def test_repeat_on_a_non_train_split_is_rejected(stub_classes):
    data_cfg = ConfigDict({"points": {"val": {"whitelist": [1], "repeat": 2}}})
    with pytest.raises(ValueError, match="repeat"):
        build_datasets(data_cfg, padding=0, dataset_classes=stub_classes, verbose=False)


def test_build_order_follows_the_registry_not_the_config(stub_classes, monkeypatch):
    monkeypatch.setattr("mermaidseg.datasets.factory.NO_PADDING_DATASETS", frozenset({"dense"}))
    # Registry order is points, dense; the config lists them in reverse.
    data_cfg = ConfigDict(
        {
            "dense": {"train": {"split": "train"}},
            "points": {"train": {"whitelist": [1]}},
        }
    )
    built = build_datasets(data_cfg, padding=0, dataset_classes=stub_classes, verbose=False)
    assert list(built) == [("points", "train"), ("dense", "train")]
