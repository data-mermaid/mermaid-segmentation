"""build_datasets: readers, transforms, groups, resize and image ids from a DataSection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mermaidseg.data import DenseDataset, PointDataset
from mermaidseg.data.config import DataSection
from mermaidseg.data.factory import build_datasets, open_dataset_version
from segdata_spec.errors import NotSynced
from segdata_spec.manifest import Pin
from tests.mermaidseg.data.conftest import ImageSpec, write_version

LABELS = {5: "five"}


def _section(pin: Pin, **entry: Any) -> DataSection:
    cfg = {"version": pin.version, "manifest_sha256": pin.manifest_sha256, **entry}
    return DataSection.model_validate({"datasets": {pin.name: cfg}})


def _points(tmp_path: Path, images: list[ImageSpec], points: list[Any]) -> tuple[Path, Pin]:
    cache_dir, _, pin = write_version(
        tmp_path, kind="points", images=images, labels=LABELS, points=points
    )
    return cache_dir, pin


def _build(
    section: DataSection,
    cache_dir: Path,
    tmp_path: Path,
    *,
    padding: int = 0,
    input_size: tuple[int, int] = (4, 6),
    image_ids: Any = None,
) -> Any:
    return build_datasets(
        section,
        cache_dir,
        repo_root=tmp_path,
        padding=padding,
        input_size=input_size,
        image_ids=image_ids,
    )


def test_point_version_gives_a_point_dataset_with_padding_and_annotators(tmp_path: Path) -> None:
    cache_dir, pin = _points(
        tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5, "a"), ("i1", 2, 2, 5, "b")]
    )
    section = _section(pin, val={"groups": "all", "annotators": ["b"]})
    ds = _build(section, cache_dir, tmp_path, padding=2)[("ds", "val")]
    assert isinstance(ds, PointDataset)
    assert ds.padding == 2
    assert list(ds.points["annotator"]) == ["b"]
    assert ds.split == "val"


def test_dense_version_gives_a_dense_dataset(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), np.uint8)
    mask[0, 0] = 5
    cache_dir, _, pin = write_version(
        tmp_path, kind="dense", images=[ImageSpec("i1", mask=mask)], labels=LABELS
    )
    ds = _build(_section(pin, test={"groups": "all"}), cache_dir, tmp_path)[("ds", "test")]
    assert isinstance(ds, DenseDataset)


def test_only_defined_blocks_are_built_and_keyed_by_dataset_and_split(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    section = _section(pin, train={"groups": "all"}, test={"groups": "all"})
    assert sorted(_build(section, cache_dir, tmp_path)) == [("ds", "test"), ("ds", "train")]


def test_train_gets_the_compiled_transform_and_eval_splits_the_resize(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    section = _section(
        pin,
        train={"groups": "all", "transform": {"SmallestMaxSize": {"max_size": 16}}},
        val={"groups": "all"},
        test={"groups": "all"},
    )
    built = _build(section, cache_dir, tmp_path, input_size=(4, 6))
    assert built[("ds", "train")][0][0].shape == (3, 16, 16)
    assert built[("ds", "val")][0][0].shape == (3, 4, 6)
    assert built[("ds", "test")][0][0].shape == (3, 4, 6)
    assert built[("ds", "val")][0][0].dtype == np.float32


def test_use_over_two_splits_selects_the_union_of_their_groups(tmp_path: Path) -> None:
    images = [ImageSpec("i1", group="g1"), ImageSpec("i2", group="g2"), ImageSpec("i3", group="g3")]
    cache_dir, pin = _points(tmp_path, images, [("i1", 1, 1, 5), ("i2", 1, 1, 5), ("i3", 1, 1, 5)])
    (tmp_path / "splits.csv").write_text("group,split\ng1,train\ng2,val\ng3,test\n")
    section = _section(pin, split_file="splits.csv", train={"use": ["train", "val"]})
    ds = _build(section, cache_dir, tmp_path)[("ds", "train")]
    assert sorted(ds.images["image_id"]) == ["i1", "i2"]


def test_entry_resize_is_what_open_version_requires_synced(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    section = _section(pin, resize={"max_edge": 4}, val={"groups": "all"})
    with pytest.raises(NotSynced):
        _build(section, cache_dir, tmp_path)


def test_on_missing_images_drop_reaches_the_reader(tmp_path: Path) -> None:
    cache_dir, pin = _points(
        tmp_path, [ImageSpec("i1"), ImageSpec("i2")], [("i1", 1, 1, 5), ("i2", 1, 1, 5)]
    )
    (cache_dir / "img-bucket" / "imgs" / "i2.jpg").unlink()
    section = _section(pin, on_missing_images="drop", val={"groups": "all"})
    ds = _build(section, cache_dir, tmp_path)[("ds", "val")]
    assert (len(ds), ds.images_dropped) == (1, 1)


def test_image_ids_subset_a_dataset_and_split(tmp_path: Path) -> None:
    cache_dir, pin = _points(
        tmp_path, [ImageSpec("i1"), ImageSpec("i2")], [("i1", 1, 1, 5), ("i2", 1, 1, 5)]
    )
    section = _section(pin, train={"groups": "all"}, val={"groups": "all"})
    built = _build(section, cache_dir, tmp_path, image_ids={("ds", "val"): ["i2"]})
    assert list(built[("ds", "val")].images["image_id"]) == ["i2"]
    assert len(built[("ds", "train")]) == 2


def test_open_dataset_version_opens_the_pinned_version(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    version = open_dataset_version(_section(pin, val={"groups": "all"}), "ds", cache_dir)
    assert version.manifest.version == pin.version


def test_open_dataset_version_requires_the_entrys_resize_synced(tmp_path: Path) -> None:
    cache_dir, pin = _points(tmp_path, [ImageSpec("i1")], [("i1", 1, 1, 5)])
    section = _section(pin, resize={"max_edge": 4}, val={"groups": "all"})
    with pytest.raises(NotSynced):
        open_dataset_version(section, "ds", cache_dir)
