"""Live integration tests for preprocessing (requires S3 access).

Mark with @pytest.mark.live to skip in CI without S3 credentials. Run locally: pytest -m live
tests/datasets/coralnet/test_preprocessing_live.py
"""

from __future__ import annotations

import io
import os

import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("boto3")

import boto3

from mermaidseg.datasets.coralnet.preprocessing.manifest import build_manifest
from mermaidseg.datasets.coralnet.preprocessing.resize import (
    resize_and_upload_all_images,
    scan_for_missing_resized_images,
)


@pytest.fixture
def test_bucket():
    """Use a test bucket (set ENV var MERMAID_CORALNET_TEST_BUCKET)."""
    return os.getenv("MERMAID_CORALNET_TEST_BUCKET", "dev-datamermaid-sm-sources")


@pytest.fixture
def test_s3_client():
    """Boto3 S3 client."""
    return boto3.client("s3")


@pytest.mark.live
def test_resize_and_upload_end_to_end(test_bucket, test_s3_client, tmp_path):
    """Full pipeline: scan → resize/upload → manifest.

    This test:
    1. Creates a test image and uploads to S3
    2. Scans to identify resize candidates
    3. Resizes and uploads to resized/ prefix
    4. Verifies resized image exists on S3
    5. Builds manifest and validates schema
    """
    # Skip if no S3 credentials
    try:
        test_s3_client.head_bucket(Bucket=test_bucket)
    except Exception:
        pytest.skip("S3 credentials not available")

    test_prefix = "etl-test/preprocess-test"
    threshold = 1024

    # Create and upload a test image (2000x1500, longest edge > threshold)
    test_img = Image.new("RGB", (2000, 1500), color="red")
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format="JPEG")
    test_key = f"{test_prefix}/s1/images/test_img.jpg"
    test_s3_client.put_object(
        Bucket=test_bucket,
        Key=test_key,
        Body=img_bytes.getvalue(),
    )

    # Create images parquet with one image that needs resizing
    df_images = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["test_img"],
            "width": [2000],
            "height": [1500],
            "needs_resize": [True],
            "s3_key": [test_key],
        }
    )

    # Scan for resize candidates
    df_todo = scan_for_missing_resized_images(
        df_images=df_images,
        bucket=test_bucket,
        output_prefix=test_prefix,
        threshold=threshold,
        s3_client=test_s3_client,
        workers=2,
    )

    assert len(df_todo) == 1, "Scan should identify 1 image to resize"
    assert df_todo.iloc[0]["image_id"] == "test_img"

    # Resize and upload
    checkpoint_path = tmp_path / "checkpoint.parquet"
    num_resized, num_skipped, num_failed, num_corrupted = resize_and_upload_all_images(
        df_todo=df_todo,
        bucket=test_bucket,
        output_prefix=test_prefix,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        workers=2,
        s3_client=test_s3_client,
    )

    assert num_resized == 1, f"Expected 1 resized, got {num_resized}"
    assert num_failed == 0, f"Expected 0 failed, got {num_failed}"
    assert num_corrupted == 0, f"Expected 0 corrupted, got {num_corrupted}"

    # Verify resized image exists on S3
    resized_key = f"{test_prefix}/resized/s1/images/test_img.jpg"
    response = test_s3_client.head_object(Bucket=test_bucket, Key=resized_key)
    assert response["ContentLength"] > 0, "Resized image should exist on S3"

    # Verify checkpoint exists and has correct status
    df_checkpoint = pd.read_parquet(checkpoint_path)
    assert len(df_checkpoint) == 1
    assert df_checkpoint.loc[0, "status"] == "completed"
    assert pd.notna(df_checkpoint.loc[0, "resize_timestamp"])

    # Build and validate manifest
    df_manifest = build_manifest(
        df_images=df_images,
        df_checkpoint=df_checkpoint,
        output_prefix=test_prefix,
        threshold=threshold,
    )

    # Check manifest schema
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
        assert col in df_manifest.columns, f"Missing column: {col}"

    # Validate manifest data
    assert len(df_manifest) == 1
    assert df_manifest.loc[0, "source_id"] == 1
    assert df_manifest.loc[0, "image_id"] == "test_img"
    assert df_manifest.loc[0, "original_width"] == 2000
    assert df_manifest.loc[0, "original_height"] == 1500
    assert df_manifest.loc[0, "status"] == "completed"
    # Longest edge (2000) scaled to 1024: 1024 x (1500/2000) = 1024 x 750
    assert df_manifest.loc[0, "resized_width"] == 1024
    assert df_manifest.loc[0, "resized_height"] == 768  # 1500 * 1024 / 2000
    assert df_manifest.loc[0, "output_s3_key"] == resized_key


@pytest.mark.live
def test_resize_and_upload_multi_source(test_bucket, test_s3_client, tmp_path):
    """Test pipeline with multiple sources (source_id=1, source_id=2).

    This test verifies that the pipeline correctly handles images from different sources, with mixed
    resize needs.
    """
    # Skip if no S3 credentials
    try:
        test_s3_client.head_bucket(Bucket=test_bucket)
    except Exception:
        pytest.skip("S3 credentials not available")

    test_prefix = "etl-test/preprocess-multi-source"
    threshold = 2048

    # Create test images for two sources
    images_data = [
        {"source_id": 1, "image_id": "s1_large", "size": (3000, 2000), "needs_resize": True},
        {"source_id": 1, "image_id": "s1_small", "size": (800, 600), "needs_resize": False},
        {"source_id": 2, "image_id": "s2_large", "size": (4000, 3000), "needs_resize": True},
    ]

    for img_data in images_data:
        test_img = Image.new("RGB", img_data["size"], color="blue")
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format="JPEG")
        test_key = f"{test_prefix}/s{img_data['source_id']}/images/{img_data['image_id']}.jpg"
        test_s3_client.put_object(
            Bucket=test_bucket,
            Key=test_key,
            Body=img_bytes.getvalue(),
        )

    # Create images dataframe
    df_images = pd.DataFrame(
        {
            "source_id": [img["source_id"] for img in images_data],
            "image_id": [img["image_id"] for img in images_data],
            "width": [img["size"][0] for img in images_data],
            "height": [img["size"][1] for img in images_data],
            "needs_resize": [img["needs_resize"] for img in images_data],
            "s3_key": [
                f"{test_prefix}/s{img['source_id']}/images/{img['image_id']}.jpg"
                for img in images_data
            ],
        }
    )

    # Scan for resize candidates
    df_todo = scan_for_missing_resized_images(
        df_images=df_images,
        bucket=test_bucket,
        output_prefix=test_prefix,
        threshold=threshold,
        s3_client=test_s3_client,
        workers=2,
    )

    # Should identify 2 images (s1_large and s2_large, but not s1_small which has needs_resize=False)
    assert len(df_todo) == 2
    image_ids = set(df_todo["image_id"].tolist())
    assert "s1_large" in image_ids
    assert "s2_large" in image_ids
    assert "s1_small" not in image_ids

    # Resize and upload
    checkpoint_path = tmp_path / "checkpoint.parquet"
    num_resized, num_skipped, num_failed, num_corrupted = resize_and_upload_all_images(
        df_todo=df_todo,
        bucket=test_bucket,
        output_prefix=test_prefix,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        workers=2,
        s3_client=test_s3_client,
    )

    assert num_resized == 2, f"Expected 2 resized, got {num_resized}"
    assert num_failed == 0, f"Expected 0 failed, got {num_failed}"
    assert num_corrupted == 0, f"Expected 0 corrupted, got {num_corrupted}"

    # Verify both resized images exist on S3
    for image_id in ["s1_large", "s2_large"]:
        source_id = 1 if image_id.startswith("s1") else 2
        resized_key = f"{test_prefix}/resized/s{source_id}/images/{image_id}.jpg"
        response = test_s3_client.head_object(Bucket=test_bucket, Key=resized_key)
        assert response["ContentLength"] > 0, f"Resized image {image_id} should exist on S3"

    # Build manifest and verify
    df_checkpoint = pd.read_parquet(checkpoint_path)
    df_manifest = build_manifest(
        df_images=df_images,
        df_checkpoint=df_checkpoint,
        output_prefix=test_prefix,
        threshold=threshold,
    )

    assert len(df_manifest) == 3  # All 3 images
    completed = df_manifest[df_manifest["status"] == "completed"]
    assert len(completed) == 2, "2 images should be marked completed"


@pytest.mark.live
def test_scan_respects_already_resized(test_bucket, test_s3_client, tmp_path):
    """Test that scan skips images already resized on S3.

    This test verifies idempotency: if an image is already resized
    and on S3, scan should not include it in the todo list.
    """
    # Skip if no S3 credentials
    try:
        test_s3_client.head_bucket(Bucket=test_bucket)
    except Exception:
        pytest.skip("S3 credentials not available")

    test_prefix = "etl-test/preprocess-idempotent"
    threshold = 1024

    # Create and upload original image
    test_img = Image.new("RGB", (2000, 1500), color="green")
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format="JPEG")
    original_key = f"{test_prefix}/s1/images/test_img_idempotent.jpg"
    test_s3_client.put_object(
        Bucket=test_bucket,
        Key=original_key,
        Body=img_bytes.getvalue(),
    )

    # Pre-upload a resized version to simulate prior completion
    resized_img = Image.new("RGB", (1024, 768), color="green")
    resized_bytes = io.BytesIO()
    resized_img.save(resized_bytes, format="JPEG")
    resized_key = f"{test_prefix}/resized/s1/images/test_img_idempotent.jpg"
    test_s3_client.put_object(
        Bucket=test_bucket,
        Key=resized_key,
        Body=resized_bytes.getvalue(),
    )

    # Create images dataframe
    df_images = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["test_img_idempotent"],
            "width": [2000],
            "height": [1500],
            "needs_resize": [True],
            "s3_key": [original_key],
        }
    )

    # Phase 1 should skip this image (already resized)
    df_todo = scan_for_missing_resized_images(
        df_images=df_images,
        bucket=test_bucket,
        output_prefix=test_prefix,
        threshold=threshold,
        s3_client=test_s3_client,
        workers=2,
    )

    assert len(df_todo) == 0, "Phase 1 should skip already-resized images"


@pytest.mark.live
def test_checkpoint_recovery(test_bucket, test_s3_client, tmp_path):
    """Test that Phase 2 can resume from a checkpoint (fault tolerance).

    This test verifies that if Phase 2 is interrupted, it can resume by reading the checkpoint and
    only processing pending items.
    """
    # Skip if no S3 credentials
    try:
        test_s3_client.head_bucket(Bucket=test_bucket)
    except Exception:
        pytest.skip("S3 credentials not available")

    test_prefix = "etl-test/preprocess-checkpoint-recovery"
    threshold = 2048

    # Create two test images
    images_data = [
        {"image_id": "img1", "size": (3000, 2000)},
        {"image_id": "img2", "size": (2500, 1500)},
    ]

    for img_data in images_data:
        test_img = Image.new("RGB", img_data["size"], color="cyan")
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format="JPEG")
        test_key = f"{test_prefix}/s1/images/{img_data['image_id']}.jpg"
        test_s3_client.put_object(
            Bucket=test_bucket,
            Key=test_key,
            Body=img_bytes.getvalue(),
        )

    # Create images dataframe
    df_images = pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["img1", "img2"],
            "width": [3000, 2500],
            "height": [2000, 1500],
            "needs_resize": [True, True],
            "s3_key": [
                f"{test_prefix}/s1/images/{img_data['image_id']}.jpg" for img_data in images_data
            ],
        }
    )

    # Phase 1
    df_todo = scan_for_missing_resized_images(
        df_images=df_images,
        bucket=test_bucket,
        output_prefix=test_prefix,
        threshold=threshold,
        s3_client=test_s3_client,
        workers=2,
    )

    assert len(df_todo) == 2

    # Phase 2 - process all items
    checkpoint_path = tmp_path / "checkpoint_recovery.parquet"
    num_resized, num_skipped, num_failed, num_corrupted = resize_and_upload_all_images(
        df_todo=df_todo,
        bucket=test_bucket,
        output_prefix=test_prefix,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        workers=2,
        s3_client=test_s3_client,
    )

    assert num_resized == 2, f"Expected 2 resized in first pass, got {num_resized}"
    assert num_failed == 0
    assert num_corrupted == 0

    # Verify checkpoint shows both completed
    df_checkpoint = pd.read_parquet(checkpoint_path)
    assert len(df_checkpoint) == 2
    assert (df_checkpoint["status"] == "completed").all()

    # Run Phase 2 again with same checkpoint (no new work should happen)
    num_resized2, num_skipped2, num_failed2, num_corrupted2 = resize_and_upload_all_images(
        df_todo=df_todo,
        bucket=test_bucket,
        output_prefix=test_prefix,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        workers=2,
        s3_client=test_s3_client,
    )

    # All items already completed, should be skipped
    assert num_resized2 == 0, "No new resizes on second pass (all completed)"
    assert num_skipped2 == 2, f"Expected 2 skipped, got {num_skipped2}"
    assert num_corrupted2 == 0
