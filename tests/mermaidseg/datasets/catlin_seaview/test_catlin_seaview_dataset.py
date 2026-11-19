"""CatlinSeaviewDataset on a tiny annotations parquet and a two-image bucket."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mermaidseg.datasets.catlin_seaview import CatlinSeaviewDataset

BUCKET = "bkt"
ANNOTATIONS_KEY = "catlin/annotations.parquet"
PREFIX = "catlin"


@pytest.fixture
def dataset(bucket_store) -> CatlinSeaviewDataset:
    rows = [
        # ATL image 1: three cca, one sand. IND image 2: one sand, one turf.
        ("ATL", 1, 1, 1, "cca"),
        ("ATL", 1, 2, 2, "cca"),
        ("ATL", 1, 3, 3, "cca"),
        ("ATL", 1, 3, 4, "sand"),
        ("IND", 2, 0, 0, "sand"),
        ("IND", 2, 1, 0, "turf"),
    ]
    frame = pd.DataFrame(rows, columns=["region", "image_id", "row", "col", "source_label_name"])
    bucket_store.put_parquet(BUCKET, ANNOTATIONS_KEY, frame)
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/ATL/1.jpg", (6, 5), (10, 0, 0))
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/IND/2.jpg", (4, 3), (0, 20, 0))
    return CatlinSeaviewDataset(
        annotations_path=ANNOTATIONS_KEY,
        source_bucket=BUCKET,
        source_s3_prefix=PREFIX,
        split="val",
    )


def test_each_item_reads_its_regions_image_with_its_own_points(dataset):
    assert len(dataset) == 2
    # Local ids by frequency: cca 3, sand 2, turf 1.
    assert dataset.source_id2name == {1: "cca", 2: "sand", 3: "turf"}

    atl_image, atl_mask = dataset[0]
    ind_image, ind_mask = dataset[1]

    assert atl_image.shape == (5, 6, 3) and np.all(atl_image == (10, 0, 0))
    expected_atl = np.zeros((5, 6), dtype=np.int32)
    expected_atl[1, 1] = expected_atl[2, 2] = expected_atl[3, 3] = 1
    expected_atl[3, 4] = 2
    np.testing.assert_array_equal(atl_mask, expected_atl)

    assert ind_image.shape == (3, 4, 3) and np.all(ind_image == (0, 20, 0))
    expected_ind = np.zeros((3, 4), dtype=np.int32)
    expected_ind[0, 0] = 2
    expected_ind[1, 0] = 3
    np.testing.assert_array_equal(ind_mask, expected_ind)
