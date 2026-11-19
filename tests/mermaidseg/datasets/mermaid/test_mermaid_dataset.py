"""MermaidDataset on a tiny annotations parquet and a one-image bucket."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mermaidseg.datasets.mermaid import MermaidDataset

BUCKET = "bkt"
BEFORE_CUTOFF = "2026-01-01"
AFTER_CUTOFF = "2026-09-01"


def _annotations_file(tmp_path):
    # img_a has three audited points and one updated after the cutoff; img_b only has a late one.
    rows = [
        ("img_a", 1, "r1", "Hard coral", 1, 2, BEFORE_CUTOFF),
        ("img_a", 1, "r1", "Hard coral", 2, 3, BEFORE_CUTOFF),
        ("img_a", 1, "r1", "Sand", 4, 5, BEFORE_CUTOFF),
        ("img_a", 1, "r1", "Sand", 0, 0, AFTER_CUTOFF),
        ("img_b", 2, "r2", "Sand", 1, 1, AFTER_CUTOFF),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "region_id",
            "region_name",
            "benthic_attribute_name",
            "row",
            "col",
            "updated_on",
        ],
    )
    frame["updated_on"] = pd.to_datetime(frame["updated_on"], utc=True)
    path = tmp_path / "annotations.parquet"
    frame.to_parquet(path, index=False)
    return str(path)


def test_item_is_the_rgb_image_and_a_point_mask_of_audited_annotations(bucket_store, tmp_path):
    bucket_store.put_solid(BUCKET, "mermaid/img_a.png", (8, 6), (9, 8, 7))

    ds = MermaidDataset(
        annotations_path=_annotations_file(tmp_path),
        source_bucket=BUCKET,
        annotation_date_cutoff="2026-08-12",
        split="val",
    )
    image, mask = ds[0]

    assert len(ds) == 1  # img_b has no annotation up to the cutoff
    assert ds.df_images["image_id"].tolist() == ["img_a"]
    assert ds.source_id2name == {1: "Hard coral", 2: "Sand"}  # by frequency after the cutoff
    assert image.shape == (6, 8, 3)
    assert np.all(image == (9, 8, 7))
    expected = np.zeros((6, 8), dtype=np.int32)
    expected[1, 2] = expected[2, 3] = 1  # Hard coral
    expected[4, 5] = 2  # Sand; the (0, 0) Sand point is after the cutoff
    np.testing.assert_array_equal(mask, expected)


def test_construction_without_a_cutoff_raises_a_missing_argument_error(bucket_store, tmp_path):
    with pytest.raises(TypeError, match="annotation_date_cutoff"):
        MermaidDataset(annotations_path=_annotations_file(tmp_path), source_bucket=BUCKET)
