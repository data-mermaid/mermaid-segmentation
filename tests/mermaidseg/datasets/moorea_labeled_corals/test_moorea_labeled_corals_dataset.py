"""MooreaLabeledCoralsDataset on a tiny annotations parquet and a two-image bucket."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mermaidseg.datasets.moorea_labeled_corals import MooreaLabeledCoralsDataset

BUCKET = "bkt"
ANNOTATIONS_KEY = "moorea/annotations.parquet"
PREFIX = "moorea"


@pytest.fixture
def dataset(bucket_store) -> MooreaLabeledCoralsDataset:
    rows = [
        # 2008 frame: three acrop and one turf. 2009 frame: one turf, one sand.
        (2008, "frame1", ".jpg", 1, 1, "acrop"),
        (2008, "frame1", ".jpg", 2, 2, "acrop"),
        (2008, "frame1", ".jpg", 3, 3, "acrop"),
        (2008, "frame1", ".jpg", 3, 4, "turf"),
        (2009, "frame2", ".png", 0, 0, "turf"),
        (2009, "frame2", ".png", 1, 0, "sand"),
    ]
    frame = pd.DataFrame(
        rows, columns=["year", "image_id", "image_ext", "row", "col", "source_label_name"]
    )
    bucket_store.put_parquet(BUCKET, ANNOTATIONS_KEY, frame)
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/2008/frame1.jpg", (6, 5), (10, 0, 0))
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/2009/frame2.png", (4, 3), (0, 20, 0))
    return MooreaLabeledCoralsDataset(
        annotations_path=ANNOTATIONS_KEY,
        source_bucket=BUCKET,
        source_s3_prefix=PREFIX,
        split="val",
    )


def test_each_item_reads_its_years_image_with_its_own_points(dataset):
    assert len(dataset) == 2
    assert dataset.source_id2name == {1: "acrop", 2: "turf", 3: "sand"}
    assert dataset.df_images["year"].tolist() == ["2008", "2009"]  # years are strings

    first_image, first_mask = dataset[0]
    second_image, second_mask = dataset[1]

    assert first_image.shape == (5, 6, 3) and np.all(first_image == (10, 0, 0))
    expected_first = np.zeros((5, 6), dtype=np.int32)
    expected_first[1, 1] = expected_first[2, 2] = expected_first[3, 3] = 1
    expected_first[3, 4] = 2
    np.testing.assert_array_equal(first_mask, expected_first)

    assert second_image.shape == (3, 4, 3) and np.all(second_image == (0, 20, 0))
    expected_second = np.zeros((3, 4), dtype=np.int32)
    expected_second[0, 0] = 2
    expected_second[1, 0] = 3
    np.testing.assert_array_equal(second_mask, expected_second)
