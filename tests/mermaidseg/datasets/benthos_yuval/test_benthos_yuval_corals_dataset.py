"""BenthosYuvalCoralsDataset on tiny dense label PNGs in a local bucket."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mermaidseg.datasets.benthos_yuval import BenthosYuvalCoralsDataset

BUCKET = "bkt"
ANNOTATIONS_KEY = "benthos/annotations.parquet"
PREFIX = "benthos"

# classes.json ids; "background" is not an annotated source class.
CLASSES = {"background": 0, "sand": 1, "coral": 2}


@pytest.fixture
def dataset(bucket_store) -> BenthosYuvalCoralsDataset:
    # Source labels by frequency: coral (2 rows) -> local 1, sand (1 row) -> local 2.
    frame = pd.DataFrame(
        {
            "site": ["RS24", "RS24", "CR"],
            "image_id": ["a", "a", "b"],
            "source_label_name": ["coral", "sand", "coral"],
        }
    )
    bucket_store.put_parquet(BUCKET, ANNOTATIONS_KEY, frame)
    bucket_store.put_json(BUCKET, f"{PREFIX}/classes.json", CLASSES)
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/RS24/a.png", (4, 2), (7, 7, 7))
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/CR/b.png", (4, 2), (1, 2, 3))
    # Image a: a grayscale label PNG. Image b: an RGB label PNG whose red channel carries the ids.
    bucket_store.put_label(
        BUCKET,
        f"{PREFIX}/labels/RS24/a.png",
        np.array([[0, 1, 2, 1], [2, 2, 0, 1]]),
    )
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    rgb[..., 0] = [[2, 0, 0, 1], [0, 0, 0, 0]]
    rgb[..., 1] = 200  # ignored: only the first channel is the label
    bucket_store.put_png(BUCKET, f"{PREFIX}/labels/CR/b.png", Image.fromarray(rgb))
    return BenthosYuvalCoralsDataset(
        annotations_path=ANNOTATIONS_KEY,
        source_bucket=BUCKET,
        source_s3_prefix=PREFIX,
        split="val",
    )


def test_grayscale_label_png_is_remapped_from_classes_json_ids_to_local_ids(dataset):
    assert dataset.source_id2name == {1: "coral", 2: "sand"}

    image, mask = dataset[0]

    assert image.shape == (2, 4, 3) and np.all(image == (7, 7, 7))
    # classes.json: sand=1 -> local 2, coral=2 -> local 1, background=0 -> 0.
    np.testing.assert_array_equal(mask, np.array([[0, 2, 1, 2], [1, 1, 0, 2]], dtype=np.int32))
    assert mask.dtype == np.int32


def test_rgb_label_png_uses_its_first_channel(dataset):
    image, mask = dataset[1]

    assert np.all(image == (1, 2, 3))
    np.testing.assert_array_equal(mask, np.array([[1, 0, 0, 2], [0, 0, 0, 0]], dtype=np.int32))


def test_global_offset_shifts_foreground_ids_only(dataset):
    dataset.set_global_offset(10)

    _, mask = dataset[0]

    np.testing.assert_array_equal(mask, np.array([[0, 12, 11, 12], [11, 11, 0, 12]]))


def test_annotated_unlabeled_id_zero_maps_to_nonzero_local_id_and_lookup_is_indexed_by_classes_json_id(
    bucket_store,
):
    # The writer emits `unlabeled` = 0 as an annotated class.
    classes = {"unlabeled": 0, "coral": 1, "sand": 2}
    # Annotation rows per name: coral 3, unlabeled 2, sand 1 -> local coral=1, unlabeled=2, sand=3.
    frame = pd.DataFrame(
        {
            "site": ["RS24"] * 6,
            "image_id": ["t0", "t1", "t2", "t0", "t1", "t0"],
            "source_label_name": ["coral"] * 3 + ["unlabeled"] * 2 + ["sand"],
        }
    )
    bucket_store.put_parquet(BUCKET, ANNOTATIONS_KEY, frame)
    bucket_store.put_json(BUCKET, f"{PREFIX}/classes.json", classes)

    dataset = BenthosYuvalCoralsDataset(
        annotations_path=ANNOTATIONS_KEY,
        source_bucket=BUCKET,
        source_s3_prefix=PREFIX,
        split="val",
    )

    # Indexed by classes.json id: 0 unlabeled -> 2, 1 coral -> 1, 2 sand -> 3.
    np.testing.assert_array_equal(
        dataset._classes_global_to_local, np.array([2, 1, 3], dtype=np.int32)
    )
