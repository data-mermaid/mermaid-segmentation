from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mermaidseg.data import DenseDataset, PointDataset
from segdata_spec.read import DatasetVersion
from tests.mermaidseg.data.conftest import ImageSpec, open_written

LABELS = {5: "five", 7: "seven", 9: "nine"}


def _crop_top_left(image: Any, mask: Any) -> dict[str, Any]:
    """Keep the top-left 4x4 corner, as a random crop that misses a point would."""
    return {"image": image[:4, :4], "mask": mask[:4, :4]}


def _flip(image: Any, mask: Any) -> dict[str, Any]:
    return {"image": image[:, ::-1], "mask": mask[:, ::-1]}


def _point_version(
    tmp_path: Path, points: list[tuple[str, int, int, int]], **kwargs: Any
) -> DatasetVersion:
    ids = list(dict.fromkeys(p[0] for p in points))
    images = kwargs.pop("images", [ImageSpec(i) for i in ids])
    return open_written(
        tmp_path, kind="points", images=images, labels=LABELS, points=points, **kwargs
    )


def _cropped_version(tmp_path: Path, inside: dict[str, bool]) -> DatasetVersion:
    """One point per image: inside the top-left 4x4 crop, or outside it."""
    points = [(i, *((1, 1) if is_in else (7, 7)), 5) for i, is_in in inside.items()]
    return _point_version(tmp_path, points)


def _index_of(ds: PointDataset, image_id: str) -> int:
    return int(ds.images.index[ds.images["image_id"] == image_id][0])


def test_train_split_resamples_an_image_whose_mask_is_empty(tmp_path: Path) -> None:
    version = _cropped_version(tmp_path, {"full": True, "empty": False})
    ds = PointDataset(version, split="train", transform=_crop_top_left)
    empty_idx = _index_of(ds, "empty")
    np.random.seed(0)
    for _ in range(20):
        _, mask = ds[empty_idx]
        assert mask[1, 1] == 1


def test_val_split_returns_an_image_whose_mask_is_empty(tmp_path: Path) -> None:
    version = _cropped_version(tmp_path, {"full": True, "empty": False})
    ds = PointDataset(version, split="val", transform=_crop_top_left)
    _, mask = ds[_index_of(ds, "empty")]
    assert mask.shape == (4, 4)
    assert not mask.any()


def test_train_split_raises_when_every_mask_is_empty(tmp_path: Path) -> None:
    version = _cropped_version(tmp_path, {"a": False, "b": False, "c": False})
    ds = PointDataset(version, split="train", transform=_crop_top_left)
    with pytest.raises(RuntimeError):
        ds[0]


def test_groups_select_images(tmp_path: Path) -> None:
    images = [ImageSpec("i1", "g1"), ImageSpec("i2", "g1"), ImageSpec("i3", "g2")]
    points = [("i1", 0, 0, 5), ("i2", 0, 0, 5), ("i3", 0, 0, 5)]
    version = _point_version(tmp_path, points, images=images)
    everything = PointDataset(version, split="val")
    only_g1 = PointDataset(version, split="val", groups=["g1"])
    assert len(everything) == 3
    assert sorted(only_g1.images["image_id"]) == ["i1", "i2"]


def test_vocabulary_orders_by_count_then_label_id(tmp_path: Path) -> None:
    # 5 and 9 tie on one point each; the lower id has the later name, so a name sort differs.
    labels = {5: "zeta", 7: "seven", 9: "alpha"}
    points = [
        ("i1", 0, 0, 9),
        ("i1", 1, 1, 7),
        ("i1", 2, 2, 7),
        ("i1", 3, 3, 7),
        ("i1", 4, 4, 5),
    ]
    version = open_written(
        tmp_path,
        kind="points",
        images=[ImageSpec("i1")],
        labels=labels,
        points=points,
    )
    ds = PointDataset(version, split="val")
    assert ds.source_id2label_id == {1: 7, 2: 5, 3: 9}
    assert ds.source_id2name == {1: "seven", 2: "zeta", 3: "alpha"}
    assert ds.source_name2id == {"seven": 1, "zeta": 2, "alpha": 3}
    assert ds.num_source_classes == 4


def test_global_offset_shifts_foreground_only(tmp_path: Path) -> None:
    points = [("i1", 1, 1, 7), ("i1", 2, 2, 7), ("i1", 3, 3, 5)]
    ds = PointDataset(_point_version(tmp_path, points), split="val")
    ds.set_global_offset(10)
    _, mask = ds[0]
    assert (mask[1, 1], mask[2, 2], mask[3, 3], mask[0, 0]) == (11, 11, 12, 0)


def test_transform_receives_the_mask_and_the_image_comes_back_channel_first(
    tmp_path: Path,
) -> None:
    images = [ImageSpec("i1", height=6, width=8)]
    version = _point_version(tmp_path, [("i1", 2, 1, 5)], images=images)
    ds = PointDataset(version, split="val", transform=_flip)
    image, mask = ds[0]
    assert image.shape == (3, 6, 8)
    assert (mask[2, 6], mask[2, 1]) == (1, 0)


def test_unknown_group_raises(tmp_path: Path) -> None:
    version = _point_version(tmp_path, [("i1", 0, 0, 5)])
    with pytest.raises(ValueError, match="nope"):
        PointDataset(version, split="val", groups=["g1", "nope"])


def test_reader_kind_must_match_the_manifest(tmp_path: Path) -> None:
    points_version = _point_version(tmp_path / "p", [("i1", 0, 0, 5)])
    mask = np.zeros((8, 8), np.uint8)
    dense_version = open_written(
        tmp_path / "d", kind="dense", images=[ImageSpec("i1", mask=mask)], labels=LABELS
    )
    with pytest.raises(ValueError, match="dense"):
        DenseDataset(points_version, split="val")
    with pytest.raises(ValueError, match="points"):
        PointDataset(dense_version, split="val")


def test_missing_image_fails_at_construction(tmp_path: Path) -> None:
    version = _point_version(tmp_path, [("i1", 0, 0, 5), ("i2", 0, 0, 5)])
    version.image_path({"key": "i2.jpg"}).unlink()
    with pytest.raises(FileNotFoundError, match="i2.jpg"):
        PointDataset(version, split="val")


def test_missing_image_is_dropped_and_counted_in_drop_mode(tmp_path: Path) -> None:
    version = _point_version(tmp_path, [("i1", 0, 0, 5), ("i2", 0, 0, 5)])
    version.image_path({"key": "i2.jpg"}).unlink()
    ds = PointDataset(version, split="val", on_missing_images="drop")
    assert (len(ds), ds.images_dropped) == (1, 1)
    assert list(ds.images["image_id"]) == ["i1"]


def test_missing_mask_fails_at_construction(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), np.uint8)
    version = open_written(
        tmp_path, kind="dense", images=[ImageSpec("i1", mask=mask)], labels=LABELS
    )
    version.mask_path({"mask_key": "i1.png", "image_id": "i1"}).unlink()
    with pytest.raises(FileNotFoundError, match="i1.png"):
        DenseDataset(version, split="val")
