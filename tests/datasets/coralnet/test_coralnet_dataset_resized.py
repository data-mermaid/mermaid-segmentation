"""CoralNetDataset consuming the resized training parquet (per-image image_s3_key).

These tests are hermetic: parquet reads, the boto3 client, and S3 image fetches are all stubbed, so
nothing touches AWS.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from PIL import Image

import mermaidseg.datasets.coralnet.coralnet_dataset as cnd
from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset


def _make_dataset(monkeypatch, df_annotations, captured_keys, image_provider, **base_kwargs):
    """Build a CoralNetDataset offline.

    ``df_annotations`` is returned in place of any S3 parquet read. ``image_provider`` maps a
    requested S3 key to a PIL image; every requested key is appended to ``captured_keys``.
    """
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: df_annotations.copy())
    monkeypatch.setattr(cnd.boto3, "client", lambda *a, **k: MagicMock())

    def fake_get_image_s3(s3, bucket, key, **kwargs):
        captured_keys.append(key)
        return image_provider(key)

    monkeypatch.setattr(cnd, "get_image_s3", fake_get_image_s3)
    return CoralNetDataset(**base_kwargs)


def _resized_parquet_df():
    """One image, two points, with a resolved resized image_s3_key (new-parquet shape)."""
    return pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["img1", "img1"],
            "row": [10, 5],
            "col": [20, 7],
            "coralnet_id": [82, 82],
            "source_label_name": ["82", "82"],
            "image_s3_key": ["dev/images/resized/s1/images/img1.jpg"] * 2,
        }
    )


def test_read_image_prefers_resolved_key(monkeypatch):
    """When the parquet carries image_s3_key, read_image loads exactly that key."""
    keys: list[str] = []
    ds = _make_dataset(
        monkeypatch,
        _resized_parquet_df(),
        keys,
        lambda key: Image.new("RGB", (50, 30)),
    )
    # df_images must carry the resolved key through to read_image.
    assert "image_s3_key" in ds.df_images.columns
    ds.read_image(
        image_id="img1", source_id="1", image_s3_key="dev/images/resized/s1/images/img1.jpg"
    )
    assert keys[-1] == "dev/images/resized/s1/images/img1.jpg"


def test_read_image_falls_back_to_constructed_key(monkeypatch):
    """Legacy parquet (no image_s3_key) -> read_image constructs the original key."""
    legacy = _resized_parquet_df().drop(columns=["image_s3_key"])
    keys: list[str] = []
    ds = _make_dataset(monkeypatch, legacy, keys, lambda key: Image.new("RGB", (50, 30)))
    assert "image_s3_key" not in ds.df_images.columns
    ds.read_image(image_id="img1", source_id="1")
    assert keys[-1] == "coralnet-public-images/s1/images/img1.jpg"


def test_getitem_places_label_at_point_with_correct_orientation(monkeypatch):
    """End-to-end alignment: the (row, col) from the parquet lands at mask[row, col] on the loaded
    image, with row indexing height and col indexing width (guards a row/col transpose).

    Uses a single point and a non-square image so a transpose would move the label.
    """
    df = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["img1"],
            "row": [10],
            "col": [20],
            "coralnet_id": [82],
            "source_label_name": ["82"],
            "image_s3_key": ["dev/images/resized/s1/images/img1.jpg"],
        }
    )
    keys: list[str] = []
    # PIL size is (width, height) -> numpy array shape (30, 50, 3): H=30, W=50.
    ds = _make_dataset(monkeypatch, df, keys, lambda key: Image.new("RGB", (50, 30)), padding=None)

    image, mask = ds[0]

    assert keys == ["dev/images/resized/s1/images/img1.jpg"]
    assert image.shape == (30, 50, 3)
    assert mask.shape == (30, 50)  # (height, width) — matches the loaded image
    # Single foreground class -> local id 1, placed exactly at (row=10, col=20).
    assert mask[10, 20] == 1
    assert int((mask != 0).sum()) == 1  # nothing else set; no transpose to (20, 10)


def test_collate_fn_skips_failed_loads(monkeypatch):
    """A (None, None) failed-load item is dropped; the surviving sample is stacked."""
    import torch

    keys: list[str] = []
    ds = _make_dataset(
        monkeypatch, _resized_parquet_df(), keys, lambda key: Image.new("RGB", (8, 8))
    )
    good_img = np.zeros((3, 8, 8), dtype=np.float32)
    good_mask = np.zeros((8, 8), dtype=np.int64)
    images, masks = ds.collate_fn([(None, None), (good_img, good_mask)])
    assert isinstance(images, torch.Tensor)
    assert images.shape[0] == 1 and masks.shape[0] == 1
