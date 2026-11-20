from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mermaidseg.data import DenseDataset
from segdata_spec.read import DatasetVersion
from tests.mermaidseg.data.conftest import ImageSpec, open_written

LABELS = {4: "four", 7: "seven", 9: "nine"}


def _mask(*pixels: tuple[int, int, int], channels: int | None = None) -> np.ndarray:
    mask = np.zeros((6, 6) if channels is None else (6, 6, channels), np.uint8)
    for row, col, value in pixels:
        mask[row, col] = value
    return mask


def _version(tmp_path: Path, mask: np.ndarray, **kwargs: Any) -> DatasetVersion:
    return open_written(
        tmp_path, kind="dense", images=[ImageSpec("i1", mask=mask)], labels=LABELS, **kwargs
    )


def test_native_ids_map_to_local_ids_and_the_offset(tmp_path: Path) -> None:
    # Label 4 covers two pixels and 7 one, so the local ids are 4 -> 1 and 7 -> 2.
    version = _version(tmp_path, _mask((0, 0, 4), (1, 1, 4), (2, 2, 7)))
    ds = DenseDataset(version, split="val")
    _, plain = ds[0]
    ds.set_global_offset(10)
    _, shifted = ds[0]
    assert (plain[0, 0], plain[1, 1], plain[2, 2], plain[3, 3]) == (1, 1, 2, 0)
    assert (shifted[0, 0], shifted[1, 1], shifted[2, 2], shifted[3, 3]) == (11, 11, 12, 0)


def test_label_outside_the_selection_maps_to_background(tmp_path: Path) -> None:
    # Label 9 is in labels.parquet and in the mask but not in the selection's counts.
    version = _version(tmp_path, _mask((0, 0, 4), (1, 1, 9)), label_counts=[("i1", 4, 1)])
    _, mask = DenseDataset(version, split="val")[0]
    assert (mask[0, 0], mask[1, 1]) == (1, 0)


def test_mask_value_above_the_largest_label_id_raises(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="dense",
        images=[ImageSpec("i1", mask=_mask((0, 0, 4), (1, 1, 12)))],
        labels=LABELS,
        label_counts=[("i1", 4, 1)],
    )
    with pytest.raises(ValueError, match="12"):
        DenseDataset(version, split="val")[0]


def test_three_channel_mask_uses_channel_zero(tmp_path: Path) -> None:
    mask = _mask((0, 0, 4), channels=3)
    mask[..., 1:] = 200
    _, local = DenseDataset(_version(tmp_path, mask), split="val")[0]
    assert (local[0, 0], local[1, 1]) == (1, 0)


def test_label_id_zero_in_the_counts_raises(tmp_path: Path) -> None:
    # Mask 0 is unlabeled, so a counted label 0 would turn every unlabeled pixel foreground.
    version = open_written(
        tmp_path,
        kind="dense",
        images=[ImageSpec("i1", mask=_mask((0, 0, 4)))],
        labels={0: "unlabeled", 4: "four"},
        label_counts=[("i1", 4, 1), ("i1", 0, 35)],
    )
    with pytest.raises(ValueError, match="label_id 0"):
        DenseDataset(version, split="val")
