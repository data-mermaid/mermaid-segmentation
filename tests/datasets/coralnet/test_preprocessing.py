"""Unit tests for preprocessing module."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from unittest.mock import Mock

import pandas as pd
from PIL import Image

from mermaidseg.datasets.coralnet.preprocessing.resize import (
    get_pending_items,
    phase_1_scan_for_resize,
    read_checkpoint,
    resize_image_to_threshold,
    write_checkpoint,
)


def test_resize_image_maintains_aspect_ratio():
    """Resizing longest edge > threshold maintains aspect ratio."""
    # Create 2000x1000 image (longest edge = 2000 > threshold of 1024)
    img = Image.new("RGB", (2000, 1000), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Resize to threshold=1024
    resized_bytes = resize_image_to_threshold(img_bytes, threshold=1024)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    # New width should be 1024, height scaled proportionally: 1024 * 1000 / 2000 = 512
    assert resized_img.width == 1024
    assert resized_img.height == 512


def test_resize_image_no_resize_if_below_threshold():
    """Image below threshold is not resized."""
    # Create 800x400 image (longest edge = 800 < threshold of 1024)
    img = Image.new("RGB", (800, 400), color="blue")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes, threshold=1024)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    # Should return original dimensions
    assert resized_img.width == 800
    assert resized_img.height == 400


def test_resize_image_square():
    """Resizing square image maintains square aspect ratio."""
    # Create 3000x3000 image
    img = Image.new("RGB", (3000, 3000), color="green")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes, threshold=2048)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    assert resized_img.width == 2048
    assert resized_img.height == 2048


def test_phase_1_scan_builds_todo_list():
    """Phase 1 scan identifies images that need resizing and don't exist on S3."""
    # Create mock images parquet
    df_images = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["img_a", "img_b", "img_c"],
            "width": [3000, 800, 2500],
            "height": [2000, 600, 1500],
            "needs_resize": [True, False, True],
        }
    )

    # Mock S3 client: only img_a exists (img_b has needs_resize=False anyway)
    mock_s3 = Mock()

    def head_object_side_effect(Bucket, Key):
        if "img_a" in Key:
            return {"ContentLength": 1000}  # exists
        raise Exception("NotFound")

    mock_s3.head_object.side_effect = head_object_side_effect

    # Run Phase 1
    todo_df = phase_1_scan_for_resize(
        df_images=df_images,
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        threshold=2048,
        s3_client=mock_s3,
        workers=2,
    )

    # Should only include img_c (needs_resize=True and doesn't exist on S3)
    assert len(todo_df) == 1
    assert todo_df.iloc[0]["image_id"] == "img_c"
    assert todo_df.iloc[0]["source_id"] == 2
    assert "original_s3_key" in todo_df.columns
    assert "output_s3_key" in todo_df.columns


def test_checkpoint_write_and_read(tmp_path):
    """Checkpoint parquet can be written and read back."""
    checkpoint_path = tmp_path / "checkpoint.parquet"

    # Write
    df_checkpoint = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["img_a", "img_b", "img_c"],
            "status": ["completed", "pending", "failed"],
            "resize_timestamp": [datetime.now(), None, datetime.now()],
            "error_message": [None, None, "PIL decode failed"],
        }
    )

    write_checkpoint(checkpoint_path, df_checkpoint)

    # Read
    df_read = read_checkpoint(checkpoint_path)

    assert len(df_read) == 3
    assert list(df_read["status"]) == ["completed", "pending", "failed"]
    assert pd.isna(df_read.loc[1, "resize_timestamp"])


def test_checkpoint_pending_items_extracted():
    """Only pending items are returned for processing."""
    df_checkpoint = pd.DataFrame(
        {
            "source_id": [1, 1, 2, 2],
            "image_id": ["a", "b", "c", "d"],
            "status": ["completed", "pending", "failed", "pending"],
            "resize_timestamp": [datetime.now(), None, datetime.now(), None],
            "error_message": [None, None, "error", None],
        }
    )

    df_pending = get_pending_items(df_checkpoint)

    assert len(df_pending) == 2
    assert list(df_pending["image_id"]) == ["b", "d"]
