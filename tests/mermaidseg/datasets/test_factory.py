"""Tests for the shared dataset build loop (mermaidseg.datasets.factory)."""

from __future__ import annotations

import albumentations as A
import pytest

from mermaidseg.datasets.factory import build_datasets, configured_source_names
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from mermaidseg.io import ConfigDict

SIZE = (32, 48)


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
    built = build_datasets(
        data_cfg, padding=7, dataset_classes=stub_classes, verbose=False, input_size=SIZE
    )

    assert set(built) == {("points", "train"), ("dense", "train")}
    # Point dataset receives padding; dense (no-padding) dataset does not.
    points = {k: v for k, v in built[("points", "train")].kwargs.items() if k != "transform"}
    dense = {k: v for k, v in built[("dense", "train")].kwargs.items() if k != "transform"}
    assert points == {"whitelist": [1], "padding": 7}
    assert dense == {"split": "train"}


def test_train_transform_spec_reaches_the_dataset_compiled_and_normalized(stub_classes):
    data_cfg = ConfigDict({"points": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}}})
    built = build_datasets(
        data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
    )
    transform = built[("points", "train")].kwargs["transform"]
    assert [type(t) for t in transform.transforms] == [A.HorizontalFlip, A.Normalize]


def test_train_split_without_a_transform_key_ends_in_imagenet_normalize(stub_classes):
    data_cfg = ConfigDict({"points": {"train": {"whitelist": [1]}}})
    built = build_datasets(
        data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
    )
    transforms = built[("points", "train")].kwargs["transform"].transforms
    assert [type(t) for t in transforms] == [A.Normalize]
    assert tuple(transforms[0].mean) == pytest.approx(IMAGENET_MEAN)
    assert tuple(transforms[0].std) == pytest.approx(IMAGENET_STD)


@pytest.mark.parametrize("split", ["val", "test"])
def test_eval_splits_get_the_eval_transform_at_the_input_size(stub_classes, split):
    data_cfg = ConfigDict({"points": {split: {"whitelist": [1]}}})
    built = build_datasets(
        data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
    )
    resize, normalize = built[("points", split)].kwargs["transform"].transforms
    assert (type(resize), resize.height, resize.width) == (A.Resize, 32, 48)
    assert isinstance(normalize, A.Normalize)


@pytest.mark.parametrize("split", ["val", "test"])
def test_yaml_transform_on_an_eval_split_raises_naming_dataset_and_split(stub_classes, split):
    data_cfg = ConfigDict({"points": {split: {"transform": {"Resize": {"width": 8, "height": 8}}}}})
    with pytest.raises(ValueError, match=f"points.{split}"):
        build_datasets(
            data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
        )


@pytest.mark.parametrize("build", ["build_datasets", "configured_source_names"])
def test_unknown_data_key_raises(stub_classes, build):
    data_cfg = ConfigDict({"points": {"train": {"a": 1}}, "unknown_source": {"train": None}})
    with pytest.raises(ValueError, match="unknown_source"):
        if build == "build_datasets":
            build_datasets(
                data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
            )
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
    built = build_datasets(
        data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
    )
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
        input_size=SIZE,
        names=["points"],
        splits=["train"],
        verbose=False,
    )
    assert set(built) == {("points", "train")}


def test_repeat_key_is_not_passed_to_the_dataset_constructor(stub_classes):
    data_cfg = ConfigDict({"points": {"train": {"whitelist": [1], "repeat": 5}}})
    built = build_datasets(
        data_cfg, padding=7, dataset_classes=stub_classes, verbose=False, input_size=SIZE
    )
    kwargs = built[("points", "train")].kwargs
    assert {k: v for k, v in kwargs.items() if k != "transform"} == {
        "whitelist": [1],
        "padding": 7,
    }


def test_repeat_on_a_non_train_split_is_rejected(stub_classes):
    data_cfg = ConfigDict({"points": {"val": {"whitelist": [1], "repeat": 2}}})
    with pytest.raises(ValueError, match="repeat"):
        build_datasets(
            data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
        )


def test_build_order_follows_the_registry_not_the_config(stub_classes, monkeypatch):
    monkeypatch.setattr("mermaidseg.datasets.factory.NO_PADDING_DATASETS", frozenset({"dense"}))
    # Registry order is points, dense; the config lists them in reverse.
    data_cfg = ConfigDict(
        {
            "dense": {"train": {"split": "train"}},
            "points": {"train": {"whitelist": [1]}},
        }
    )
    built = build_datasets(
        data_cfg, padding=0, dataset_classes=stub_classes, verbose=False, input_size=SIZE
    )
    assert list(built) == [("points", "train"), ("dense", "train")]


def test_configured_source_names_counts_only_datasets_with_a_live_split():
    class _Live(_RecordingDataset):
        SOURCE_NAME = "live_source"

    class _Partial(_RecordingDataset):
        SOURCE_NAME = "partial_source"

    class _AllSkipped(_RecordingDataset):
        SOURCE_NAME = "skipped_source"

    class _NullSection(_RecordingDataset):
        SOURCE_NAME = "null_source"

    class _Absent(_RecordingDataset):
        SOURCE_NAME = "absent_source"

    classes = {
        "live": _Live,
        "partial": _Partial,
        "all_skipped": _AllSkipped,
        "null_section": _NullSection,
        "absent": _Absent,
    }
    data_cfg = ConfigDict(
        {
            "live": {"train": {"a": 1}, "val": {"a": 1}},
            "partial": {"train": None, "val": None, "test": {"a": 1}},
            "all_skipped": {"train": None, "val": None, "test": None},
            "null_section": None,
            "local_cache_dir": "/cache",
        }
    )

    assert configured_source_names(data_cfg, dataset_classes=classes) == {
        "live_source",
        "partial_source",
    }
