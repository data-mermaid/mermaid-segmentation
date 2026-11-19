"""CoralscapesV2Dataset on a two-row stand-in for the HuggingFace corpus."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest
from PIL import Image

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CoralscapesV2Dataset

# Native Coralscapes V2 ids used below.
ACROPORA_ALIVE, FISH, SAND = 2, 25, 69

# One 2x3 (height x width) sample; 0 is unannotated.
LABELS = np.array([[0, ACROPORA_ALIVE, FISH], [SAND, SAND, 0]], dtype=np.uint8)


@pytest.fixture
def hf_rows(monkeypatch) -> list[dict]:
    """Make ``datasets.load_dataset`` return one train split of in-memory rows."""
    rows = [
        {
            "image": Image.new("RGB", (3, 2), (5, 6, 7)),
            "label": Image.fromarray(LABELS, mode="L"),
        }
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda repo_id: {"train": rows})
    return rows


def test_item_mask_is_the_native_label_ids_by_default(hf_rows):
    ds = CoralscapesV2Dataset(split="train")

    image, mask = ds[0]

    assert len(ds) == 1
    assert ds.num_source_classes == 96
    assert image.shape == (2, 3, 3) and np.all(image == (5, 6, 7))
    np.testing.assert_array_equal(mask, LABELS.astype(np.int32))
    assert mask.dtype == np.int32


def test_class_subset_renumbers_kept_classes_and_sends_the_rest_to_background(hf_rows):
    ds = CoralscapesV2Dataset(split="train", class_subset=["sand", "fish"])

    _, mask = ds[0]

    assert ds.source_id2name == {1: "sand", 2: "fish"}
    # acropora alive is outside the subset; sand -> 1, fish -> 2.
    np.testing.assert_array_equal(mask, np.array([[0, 0, 2], [1, 1, 0]], dtype=np.int32))


def test_global_offset_shifts_foreground_ids_only(hf_rows):
    ds = CoralscapesV2Dataset(split="train", class_subset=["sand", "fish"])
    ds.set_global_offset(10)

    _, mask = ds[0]

    np.testing.assert_array_equal(mask, np.array([[0, 0, 12], [11, 11, 0]], dtype=np.int32))


def test_transform_resizes_image_and_mask_and_puts_channels_first(hf_rows):
    ds = CoralscapesV2Dataset(
        split="train", class_subset=["sand"], transform=A.Resize(height=4, width=6)
    )

    image, mask = ds[0]

    assert image.shape == (3, 4, 6)
    assert mask.shape == (4, 6)
    # Nearest-neighbour x2: the bottom-left 2x2 block of the 4x6 mask is sand.
    assert np.all(mask[2:4, 0:4] == 1)
    assert np.all(mask[0:2] == 0)


def test_an_unreadable_item_is_skipped_as_none_none(hf_rows):
    ds = CoralscapesV2Dataset(split="train")

    assert ds[5] == (None, None)
