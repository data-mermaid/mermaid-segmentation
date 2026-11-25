from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mermaidseg.data import DenseDataset, PointDataset
from mermaidseg.data.base import _next_random_idx
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
    ds = PointDataset(version, split="train", global_offset=0, transform=_crop_top_left)
    empty_idx = _index_of(ds, "empty")
    np.random.seed(0)
    for _ in range(20):
        _, mask = ds[empty_idx]
        assert mask[1, 1] == 5


def test_val_split_returns_an_image_whose_mask_is_empty(tmp_path: Path) -> None:
    version = _cropped_version(tmp_path, {"full": True, "empty": False})
    ds = PointDataset(version, split="val", global_offset=0, transform=_crop_top_left)
    _, mask = ds[_index_of(ds, "empty")]
    assert mask.shape == (4, 4)
    assert not mask.any()


def test_train_split_raises_when_every_mask_is_empty(tmp_path: Path) -> None:
    version = _cropped_version(tmp_path, {"a": False, "b": False, "c": False})
    ds = PointDataset(version, split="train", global_offset=0, transform=_crop_top_left)
    with pytest.raises(RuntimeError):
        ds[0]


def test_groups_select_images(tmp_path: Path) -> None:
    images = [ImageSpec("i1", "g1"), ImageSpec("i2", "g1"), ImageSpec("i3", "g2")]
    points = [("i1", 0, 0, 5), ("i2", 0, 0, 5), ("i3", 0, 0, 5)]
    version = _point_version(tmp_path, points, images=images)
    everything = PointDataset(version, split="val", global_offset=0)
    only_g1 = PointDataset(version, split="val", global_offset=0, groups=["g1"])
    assert len(everything) == 3
    assert sorted(only_g1.images["image_id"]) == ["i1", "i2"]


def test_global_offset_shifts_foreground_only(tmp_path: Path) -> None:
    points = [("i1", 1, 1, 7), ("i1", 2, 2, 7), ("i1", 3, 3, 5)]
    ds = PointDataset(_point_version(tmp_path, points), split="val", global_offset=10)
    _, mask = ds[0]
    assert (mask[1, 1], mask[2, 2], mask[3, 3], mask[0, 0]) == (17, 17, 15, 0)


def test_transform_receives_the_mask_and_the_image_comes_back_channel_first(
    tmp_path: Path,
) -> None:
    images = [ImageSpec("i1", height=6, width=8)]
    version = _point_version(tmp_path, [("i1", 2, 1, 5)], images=images)
    ds = PointDataset(version, split="val", global_offset=0, transform=_flip)
    image, mask = ds[0]
    assert image.shape == (3, 6, 8)
    assert (mask[2, 6], mask[2, 1]) == (5, 0)


def test_unknown_group_raises(tmp_path: Path) -> None:
    version = _point_version(tmp_path, [("i1", 0, 0, 5)])
    with pytest.raises(ValueError, match="nope"):
        PointDataset(version, split="val", global_offset=0, groups=["g1", "nope"])


def test_reader_kind_must_match_the_manifest(tmp_path: Path) -> None:
    points_version = _point_version(tmp_path / "p", [("i1", 0, 0, 5)])
    mask = np.zeros((8, 8), np.uint8)
    dense_version = open_written(
        tmp_path / "d", kind="dense", images=[ImageSpec("i1", mask=mask)], labels=LABELS
    )
    with pytest.raises(ValueError, match="dense"):
        DenseDataset(points_version, split="val", global_offset=0)
    with pytest.raises(ValueError, match="points"):
        PointDataset(dense_version, split="val", global_offset=0)


def test_missing_image_fails_at_construction(tmp_path: Path) -> None:
    version = _point_version(tmp_path, [("i1", 0, 0, 5), ("i2", 0, 0, 5)])
    version.image_path({"key": "i2.jpg"}).unlink()
    with pytest.raises(FileNotFoundError, match="i2.jpg"):
        PointDataset(version, split="val", global_offset=0)


def test_missing_mask_fails_at_construction(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), np.uint8)
    version = open_written(
        tmp_path, kind="dense", images=[ImageSpec("i1", mask=mask)], labels=LABELS
    )
    version.mask_path({"mask_key": "i1.png", "image_id": "i1"}).unlink()
    with pytest.raises(FileNotFoundError, match="i1.png"):
        DenseDataset(version, split="val", global_offset=0)


def test_empty_selection_raises_on_access(tmp_path: Path) -> None:
    version = _point_version(tmp_path, [("i1", 0, 0, 5)])
    ds = PointDataset(version, split="val", global_offset=0, groups=[])
    with pytest.raises(RuntimeError, match="dataset is empty"):
        ds[0]


@pytest.mark.parametrize(
    ("idx", "n", "expected"),
    [(0, 1, {0}), (0, 3, {1, 2}), (1, 3, {0, 2}), (2, 3, {0, 1})],
)
def test_next_random_idx_skips_idx_and_reaches_every_other_index(
    idx: int, n: int, expected: set[int]
) -> None:
    seen = set()
    for seed in range(50):
        np.random.seed(seed)
        seen.add(_next_random_idx(idx, n))
    assert seen == expected


def _two_group_version(tmp_path: Path) -> DatasetVersion:
    images = [ImageSpec("i1", group="g1"), ImageSpec("i2", group="g2"), ImageSpec("i3", group="g2")]
    return _point_version(
        tmp_path, [("i1", 0, 0, 5), ("i2", 0, 0, 5), ("i3", 0, 0, 5)], images=images
    )


def test_selected_group_missing_from_the_sync_marker_raises(tmp_path: Path) -> None:
    version = replace(_two_group_version(tmp_path), synced_groups=frozenset({"g1"}))
    with pytest.raises(ValueError, match=r"\['g2'\].*sync_data\.py"):
        PointDataset(version, split="val", global_offset=0, groups=["g1", "g2"])


def test_all_groups_selected_checks_every_group_against_the_marker(tmp_path: Path) -> None:
    version = replace(_two_group_version(tmp_path), synced_groups=frozenset({"g1"}))
    with pytest.raises(ValueError, match=r"\['g2'\]"):
        PointDataset(version, split="val", global_offset=0)


def test_selection_inside_the_synced_groups_is_accepted(tmp_path: Path) -> None:
    version = replace(_two_group_version(tmp_path), synced_groups=frozenset({"g1"}))
    assert len(PointDataset(version, split="val", global_offset=0, groups=["g1"])) == 1


def test_synced_groups_none_accepts_any_group(tmp_path: Path) -> None:
    version = _two_group_version(tmp_path)
    assert version.synced_groups is None
    assert len(PointDataset(version, split="val", global_offset=0)) == 3


def test_image_ids_keep_only_the_listed_images(tmp_path: Path) -> None:
    ds = PointDataset(
        _two_group_version(tmp_path), split="val", global_offset=0, image_ids=["i1", "i3"]
    )
    assert sorted(ds.images["image_id"]) == ["i1", "i3"]
    assert sorted(ds.points["image_id"]) == ["i1", "i3"]


def test_image_id_outside_the_group_selection_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="i2"):
        PointDataset(
            _two_group_version(tmp_path),
            split="val",
            global_offset=0,
            groups=["g1"],
            image_ids=["i1", "i2"],
        )


def test_image_ids_apply_to_dense_readers(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), np.uint8)
    mask[0, 0] = 5
    version = open_written(
        tmp_path,
        kind="dense",
        images=[ImageSpec("i1", mask=mask), ImageSpec("i2", mask=mask)],
        labels=LABELS,
    )
    ds = DenseDataset(version, split="val", global_offset=0, image_ids=["i2"])
    assert list(ds.images["image_id"]) == ["i2"]
    with pytest.raises(ValueError, match="nope"):
        DenseDataset(version, split="val", global_offset=0, image_ids=["nope"])
