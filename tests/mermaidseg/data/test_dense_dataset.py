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


def test_native_ids_map_to_the_offset_plus_the_id(tmp_path: Path) -> None:
    version = _version(tmp_path, _mask((0, 0, 4), (1, 1, 4), (2, 2, 7)))
    _, mask = DenseDataset(version, split="val", global_offset=10)[0]
    assert (mask[0, 0], mask[1, 1], mask[2, 2], mask[3, 3]) == (14, 14, 17, 0)


def test_mask_value_missing_from_labels_parquet_maps_to_background(tmp_path: Path) -> None:
    # 5 lies below the largest label_id (9) but is not a row of labels.parquet.
    version = _version(tmp_path, _mask((0, 0, 4), (1, 1, 5)), label_counts=[("i1", 4, 1)])
    _, mask = DenseDataset(version, split="val", global_offset=10)[0]
    assert (mask[0, 0], mask[1, 1]) == (14, 0)


def test_mask_value_above_the_largest_label_id_raises(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="dense",
        images=[ImageSpec("i1", mask=_mask((0, 0, 4), (1, 1, 12)))],
        labels=LABELS,
        label_counts=[("i1", 4, 1)],
    )
    with pytest.raises(ValueError, match="12"):
        DenseDataset(version, split="val", global_offset=0)[0]


def test_three_channel_mask_uses_channel_zero(tmp_path: Path) -> None:
    mask = _mask((0, 0, 4), channels=3)
    mask[..., 1:] = 200
    _, local = DenseDataset(_version(tmp_path, mask), split="val", global_offset=0)[0]
    assert (local[0, 0], local[1, 1]) == (4, 0)
