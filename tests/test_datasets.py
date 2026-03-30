"""Unit tests for mermaidseg.datasets.utils and BaseCoralDataset error handling."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.datasets.dataset import BaseCoralDataset
from mermaidseg.datasets.utils import DataLoadError, create_annotation_mask, get_image_s3


def _make_annotations(rows, cols, labels):
    """Build a minimal annotations DataFrame."""
    return pd.DataFrame({"row": rows, "col": cols, "benthic_attribute_name": labels})


def _valid_png_bytes():
    """Return raw bytes for a tiny valid 2x2 RGB PNG."""
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_s3_response(data: bytes):
    """Return a mock boto3 get_object response dict."""
    body = MagicMock()
    body.read.return_value = data
    return {"Body": body}


def test_create_annotation_mask_basic():
    annotations = _make_annotations([10, 20, 30], [5, 15, 25], ["Coral", "Sand", "Rubble"])
    label2id = {"Coral": 1, "Sand": 2, "Rubble": 3}
    mask = create_annotation_mask(annotations, (50, 50), label2id)

    assert mask[10, 5] == 1
    assert mask[20, 15] == 2
    assert mask[30, 25] == 3
    # All other pixels should be background
    assert mask[0, 0] == 0


def test_create_annotation_mask_with_padding():
    annotations = _make_annotations([10], [10], ["Coral"])
    label2id = {"Coral": 1}
    padding = 2
    mask = create_annotation_mask(annotations, (50, 50), label2id, padding=padding)

    # The padded region [8:12, 8:12] should all be 1
    assert np.all(mask[8:12, 8:12] == 1)
    # Outside padding should be 0
    assert mask[7, 10] == 0
    assert mask[12, 10] == 0


def test_create_annotation_mask_padding_bounds_clamped():
    """Annotation at top-left corner with large padding must not raise IndexError."""
    annotations = _make_annotations([0], [0], ["Coral"])
    label2id = {"Coral": 1}
    # Should not raise even though padding extends outside image bounds
    mask = create_annotation_mask(annotations, (20, 20), label2id, padding=5)
    assert mask[0, 0] == 1


def test_create_annotation_mask_unknown_label_skipped(caplog):
    annotations = _make_annotations([5, 10], [5, 10], ["Coral", "UnknownLabel"])
    label2id = {"Coral": 1}

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.utils"):
        mask = create_annotation_mask(annotations, (20, 20), label2id)

    # Known label is written
    assert mask[5, 5] == 1
    # Unknown label point stays 0
    assert mask[10, 10] == 0
    # A warning was emitted
    assert "unknown label" in caplog.text.lower() or "UnknownLabel" in caplog.text


def test_create_annotation_mask_empty():
    annotations = _make_annotations([], [], [])
    mask = create_annotation_mask(annotations, (10, 10), {"Coral": 1})
    assert np.all(mask == 0)


def test_create_annotation_mask_all_null_labels():
    annotations = _make_annotations([5], [5], [None])
    mask = create_annotation_mask(annotations, (10, 10), {"Coral": 1})
    assert np.all(mask == 0)


def test_get_image_s3_success():
    s3 = MagicMock()
    s3.get_object.return_value = _make_s3_response(_valid_png_bytes())

    image = get_image_s3(s3, "my-bucket", "path/to/image.png")

    assert image is not None
    s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="path/to/image.png")


def test_get_image_s3_thumbnail_modifies_key():
    s3 = MagicMock()
    s3.get_object.return_value = _make_s3_response(_valid_png_bytes())

    get_image_s3(s3, "my-bucket", "path/to/image.png", thumbnail=True)

    s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="path/to/image_thumbnail.png")


def test_get_image_s3_client_error_raises_data_load_error(caplog):
    from botocore.exceptions import ClientError

    s3 = MagicMock()
    s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
        "GetObject",
    )

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.utils"):
        with pytest.raises(DataLoadError):
            get_image_s3(s3, "my-bucket", "missing/image.png")

    assert "NoSuchKey" in caplog.text or "S3 error" in caplog.text


def test_get_image_s3_corrupted_image_raises_data_load_error(caplog):
    s3 = MagicMock()
    s3.get_object.return_value = _make_s3_response(b"not-a-valid-image")

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.utils"):
        with pytest.raises(DataLoadError):
            get_image_s3(s3, "my-bucket", "corrupt/image.png")

    assert "corrupt" in caplog.text.lower() or "PIL" in caplog.text


def _make_minimal_dataset() -> BaseCoralDataset:
    """Build the smallest valid BaseCoralDataset (no real images needed)."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1", "img2"],
            "region_id": [1, 2],
            "region_name": ["r1", "r2"],
            "benthic_attribute_name": ["Coral", "Sand"],
            "row": [10, 20],
            "col": [10, 20],
        }
    )
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    class_subset = ["Coral", "Sand"]
    ds = BaseCoralDataset.__new__(BaseCoralDataset)
    # Bypass __init__ by setting attributes directly
    ds.df_annotations = df_annotations
    ds.df_images = df_images
    ds.split = None
    ds.transform = None
    ds.padding = None
    ds.class_subset = class_subset
    ds.num_classes = len(class_subset) + 1
    ds.id2label = {1: "Coral", 2: "Sand"}
    ds.label2id = {"Coral": 1, "Sand": 2}
    ds.concept_mapping_flag = False
    return ds


def test_collate_fn_filters_none_items(caplog):
    ds = _make_minimal_dataset()
    h, w = 4, 4
    img = torch.zeros(3, h, w)
    msk = torch.zeros(h, w, dtype=torch.long)
    batch = [(img, msk), (None, None), (img, msk), (None, None)]

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.dataset"):
        images, masks = ds.collate_fn(batch)

    assert images.shape[0] == 2
    assert masks.shape[0] == 2
    assert "skipped 2/4" in caplog.text


def test_collate_fn_all_none_returns_empty_tensors(caplog):
    ds = _make_minimal_dataset()
    batch = [(None, None), (None, None)]

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.dataset"):
        images, masks = ds.collate_fn(batch)

    assert images.numel() == 0
    assert masks.numel() == 0
    assert "entire batch" in caplog.text


class _AlwaysFailDataset(BaseCoralDataset):
    """Minimal subclass whose read_image always raises."""

    def read_image(self, **row_kwargs) -> Any:
        raise RuntimeError("simulated read failure")


def test_base_dataset_getitem_skips_and_logs_on_read_failure(caplog):
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1"],
            "region_id": [1],
            "region_name": ["r1"],
            "benthic_attribute_name": ["Coral"],
            "row": [5],
            "col": [5],
        }
    )
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.dataset"):
        result = ds[0]

    assert result == (None, None)
    assert "img1" in caplog.text
    assert "RuntimeError" in caplog.text


def test_data_load_error_importable():
    from mermaidseg.datasets.utils import DataLoadError as DLE

    assert issubclass(DLE, Exception)
