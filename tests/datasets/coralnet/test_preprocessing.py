"""Unit tests for preprocessing module."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, Mock

import pandas as pd
from PIL import Image

from mermaidseg.datasets.coralnet.preprocessing.manifest import build_manifest
from mermaidseg.datasets.coralnet.preprocessing.resize import (
    get_pending_items,
    read_checkpoint,
    resize_and_upload_image,
    resize_image_to_threshold,
    scan_for_missing_resized_images,
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


def test_scan_for_missing_resized_images_builds_todo_list():
    """Scan identifies images that need resizing and don't exist on S3."""
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

    # Run scan
    todo_df = scan_for_missing_resized_images(
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


def test_phase_2_resize_one_image(tmp_path):
    """Single image is downloaded, resized, and uploaded."""
    # Create mock S3 client
    mock_s3 = Mock()

    # Mock GET (download): return a 3000x2000 JPEG
    original_img = Image.new("RGB", (3000, 2000), color="red")
    img_bytes = BytesIO()
    original_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    mock_s3.get_object.return_value = {"Body": BytesIO(img_bytes.getvalue())}

    # Mock PUT (upload): track calls
    mock_s3.put_object = MagicMock()
    mock_s3.head_object = MagicMock()

    checkpoint_path = tmp_path / "checkpoint.parquet"

    # Create todo item
    todo_item = {
        "source_id": 1,
        "image_id": "img_a",
        "width": 3000,
        "height": 2000,
        "original_s3_key": "coralnet-public-images/s1/images/img_a.jpg",
        "output_s3_key": "etl-outputs/coralnet/resized/2048/s1/images/img_a.jpg",
    }

    # Create initial checkpoint
    checkpoint_df = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["img_a"],
            "status": ["pending"],
            "resize_timestamp": [None],
            "error_message": [None],
        }
    )

    write_checkpoint(checkpoint_path, checkpoint_df)

    # Run single image resize/upload
    resize_and_upload_image(
        todo_item=todo_item,
        bucket="test-bucket",
        checkpoint_path=checkpoint_path,
        threshold=2048,
        s3_client=mock_s3,
    )

    # Verify PUT was called
    assert mock_s3.put_object.called
    call_args = mock_s3.put_object.call_args
    assert call_args[1]["Bucket"] == "test-bucket"
    assert call_args[1]["Key"] == "etl-outputs/coralnet/resized/2048/s1/images/img_a.jpg"

    # Verify checkpoint updated to 'completed'
    checkpoint_df_updated = read_checkpoint(checkpoint_path)
    assert checkpoint_df_updated.loc[0, "status"] == "completed"
    assert checkpoint_df_updated.loc[0, "resize_timestamp"] is not None


def test_manifest_schema_is_correct():
    """Manifest has required columns."""
    df_images = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["a", "b", "c"],
            "width": [3000, 800, 2500],
            "height": [2000, 600, 1500],
        }
    )

    checkpoint_df = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["a", "b", "c"],
            "status": ["completed", "failed", "completed"],
            "resize_timestamp": [datetime.now(), None, datetime.now()],
            "error_message": [None, "decode failed", None],
        }
    )

    output_prefix = "etl-outputs/coralnet"
    threshold = 2048

    manifest_df = build_manifest(
        df_images=df_images,
        df_checkpoint=checkpoint_df,
        output_prefix=output_prefix,
        threshold=threshold,
    )

    # Check schema
    required_columns = [
        "source_id",
        "image_id",
        "original_width",
        "original_height",
        "resized_width",
        "resized_height",
        "output_s3_key",
        "resize_timestamp",
        "status",
    ]
    for col in required_columns:
        assert col in manifest_df.columns, f"Missing column: {col}"

    # Check data
    assert len(manifest_df) == 3
    assert manifest_df.loc[0, "original_width"] == 3000
