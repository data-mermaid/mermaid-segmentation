"""PacificLabeledCoralsDataset on a tiny annotations parquet and a two-image bucket."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mermaidseg.datasets.pacific_labeled_corals import PacificLabeledCoralsDataset

BUCKET = "bkt"
ANNOTATIONS_KEY = "pacific/annotations.parquet"
PREFIX = "pacific"


@pytest.fixture
def dataset(bucket_store) -> PacificLabeledCoralsDataset:
    rows = [
        # site, subset, image_id, image_ext, row, col, host, archived
        ("s1", "sub1", "p1", ".png", 1, 1, "acropora", "x"),
        ("s1", "sub1", "p1", ".png", 2, 2, "acropora", "x"),
        ("s1", "sub1", "p1", ".png", 3, 3, None, "porites"),  # host missing -> archived
        ("s1", "sub1", "p1", ".png", 4, 4, "sand", "sand"),
        ("s2", "sub2", "p2", ".png", 0, 0, "sand", "x"),
        ("s2", "sub2", "p2", ".png", 1, 0, "sand", "x"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["site", "subset", "image_id", "image_ext", "row", "col", "host", "archived"],
    )
    bucket_store.put_parquet(BUCKET, ANNOTATIONS_KEY, frame)
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/s1/sub1/p1.png", (6, 5), (10, 0, 0))
    bucket_store.put_solid(BUCKET, f"{PREFIX}/images/s2/sub2/p2.png", (4, 3), (0, 20, 0))
    return PacificLabeledCoralsDataset(
        annotations_path=ANNOTATIONS_KEY,
        source_bucket=BUCKET,
        source_s3_prefix=PREFIX,
        split="val",
    )


def test_labels_come_from_the_host_annotator_with_archived_as_fallback(dataset):
    assert dataset.df_annotations["annotator_used"].tolist() == [
        "host",
        "host",
        "archived",
        "host",
        "host",
        "host",
    ]
    # Local ids by frequency: sand 3, acropora 2, porites 1.
    assert dataset.source_id2name == {1: "sand", 2: "acropora", 3: "porites"}


def test_each_item_reads_its_site_and_subset_image_with_its_own_points(dataset):
    assert len(dataset) == 2

    first_image, first_mask = dataset[0]
    second_image, second_mask = dataset[1]

    assert first_image.shape == (5, 6, 3) and np.all(first_image == (10, 0, 0))
    expected_first = np.zeros((5, 6), dtype=np.int32)
    expected_first[1, 1] = expected_first[2, 2] = 2  # acropora
    expected_first[3, 3] = 3  # porites, from the archived fallback
    expected_first[4, 4] = 1  # sand
    np.testing.assert_array_equal(first_mask, expected_first)

    assert second_image.shape == (3, 4, 3) and np.all(second_image == (0, 20, 0))
    expected_second = np.zeros((3, 4), dtype=np.int32)
    expected_second[0, 0] = expected_second[1, 0] = 1
    np.testing.assert_array_equal(second_mask, expected_second)
