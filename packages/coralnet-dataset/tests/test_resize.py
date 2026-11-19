"""Tests for the resize stage: the resize itself, the S3 scan, the worker and the orchestrator."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from PIL import Image

from coralnet_dataset.resize import (
    _validate_and_download,
    resize_and_upload_all_images,
    resize_and_upload_image,
    resize_image_to_threshold,
    scan_for_missing_resized_images,
)


def test_resize_image_maintains_aspect_ratio():
    """Resizing longest edge > threshold maintains aspect ratio."""
    # 4000x1000: longest edge 4000 > 2048
    img = Image.new("RGB", (4000, 1000), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    # Width becomes 2048, height scales proportionally: 2048 * 1000 / 4000 = 512
    assert resized_img.width == 2048
    assert resized_img.height == 512


def test_resize_image_no_resize_if_below_threshold():
    """A JPEG below the threshold is returned as the original bytes, not re-encoded."""
    # 800x400: longest edge 800 < 2048
    img = Image.new("RGB", (800, 400), color="blue")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    original = img_bytes.getvalue()

    resized_bytes = resize_image_to_threshold(BytesIO(original))

    assert resized_bytes.getvalue() == original


def test_resize_image_scales_a_portrait_image_by_its_height():
    """1000x4000: the height is the longest edge, so it becomes 2048 and the width 512."""
    img = Image.new("RGB", (1000, 4000), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")

    resized = Image.open(resize_image_to_threshold(img_bytes))

    assert resized.size == (512, 2048)


def test_resize_converts_cmyk_jpeg_below_threshold():
    """CMYK cannot be shown as RGB downstream, so it is re-encoded even when small."""
    img_bytes = BytesIO()
    Image.new("CMYK", (800, 400), color=(0, 255, 255, 0)).save(img_bytes, format="JPEG")

    resized = Image.open(resize_image_to_threshold(img_bytes))

    assert (resized.format, resized.mode, resized.size) == ("JPEG", "RGB", (800, 400))


def test_resize_re_encodes_an_rgb_png_below_threshold():
    """An RGB PNG needs no mode change but is not a JPEG, so it is still re-encoded."""
    img_bytes = BytesIO()
    Image.new("RGB", (800, 400), color="blue").save(img_bytes, format="PNG")

    resized = Image.open(resize_image_to_threshold(img_bytes))

    assert (resized.format, resized.mode, resized.size) == ("JPEG", "RGB", (800, 400))


def test_resize_converts_rgba_below_threshold():
    """A small RGBA image is still re-encoded to RGB JPEG, not passed through as raw
    bytes."""
    img = Image.new("RGBA", (800, 400), color=(0, 255, 0, 64))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    assert resized_img.format == "JPEG"
    assert resized_img.mode == "RGB"
    assert resized_img.width == 800
    assert resized_img.height == 400


def _images_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["img_a", "img_b", "img_c"],
            "width": [3000, 800, 2500],
            "height": [2000, 600, 1500],
            "needs_resize": [True, False, True],
            "s3_key": [
                "coralnet-public-images/s1/images/img_a.jpg",
                "coralnet-public-images/s1/images/img_b.jpg",
                "coralnet-public-images/s2/images/img_c.jpg",
            ],
        }
    )


def test_scan_for_missing_resized_images_skips_images_already_resized(fake_s3):
    """img_a is already resized and img_b needs no resize, so only img_c is left to do."""
    fake_s3.put("test-bucket", "etl-outputs/coralnet/resized/s1/images/img_a.jpg", b"done")

    todo_df = scan_for_missing_resized_images(
        df_images=_images_frame(),
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        s3_client=fake_s3,
    )

    expected = pd.DataFrame(
        {
            "source_id": [2],
            "image_id": ["img_c"],
            "width": [2500],
            "height": [1500],
            "original_s3_key": ["coralnet-public-images/s2/images/img_c.jpg"],
            "output_s3_key": ["etl-outputs/coralnet/resized/s2/images/img_c.jpg"],
        }
    )
    pd.testing.assert_frame_equal(todo_df, expected)


def test_scan_for_missing_resized_images_lists_every_image_when_none_is_resized(fake_s3):
    """An empty resized prefix lists no ``Contents``; every needs_resize image is to do."""
    todo_df = scan_for_missing_resized_images(
        df_images=_images_frame(),
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        s3_client=fake_s3,
    )

    assert todo_df["image_id"].tolist() == ["img_a", "img_c"]


def test_resize_converts_and_uploads_rgba_png(rgba_png_bytes):
    """Worker converts an RGBA PNG to RGB JPEG, uploads it, and returns a completed
    result."""
    mock_s3 = MagicMock()

    # Mock GET to return RGBA PNG bytes
    mock_s3.get_object.return_value = {"Body": BytesIO(rgba_png_bytes)}
    mock_s3.head_object = MagicMock()

    todo_item = {
        "source_id": 1,
        "image_id": "rgba_img",
        "original_s3_key": "coralnet-public-images/s1/images/rgba_img.png",
        "output_s3_key": "etl-outputs/coralnet/resized/s1/images/rgba_img.jpg",
    }

    result = resize_and_upload_image(
        todo_item=todo_item,
        bucket="test-bucket",
        s3_client=mock_s3,
    )

    # Verify PUT was called with a valid RGB JPEG
    assert mock_s3.put_object.called
    uploaded = mock_s3.put_object.call_args.kwargs["Body"]
    out_img = Image.open(BytesIO(uploaded))
    out_img.load()
    assert out_img.format == "JPEG"
    assert out_img.mode == "RGB"

    # Worker returns a completed result row
    assert result["status"] == "completed"


class _PutFailsFakeS3:
    """Wraps a FakeS3 so ``put_object`` raises for one output key."""

    def __init__(self, fake_s3, failing_key: str):
        self._fake = fake_s3
        self._failing_key = failing_key

    def get_object(self, **kwargs):
        return self._fake.get_object(**kwargs)

    def put_object(self, **kwargs):
        if kwargs["Key"] == self._failing_key:
            raise ConnectionError("upload refused")
        return self._fake.put_object(**kwargs)


def test_resize_all_images_counts_completed_corrupted_and_failed(
    fake_s3, valid_rgb_jpeg_bytes, truncated_jpeg_bytes
):
    fake_s3.put("b", "orig/ok.jpg", valid_rgb_jpeg_bytes)
    fake_s3.put("b", "orig/bad.jpg", truncated_jpeg_bytes)
    fake_s3.put("b", "orig/upload.jpg", valid_rgb_jpeg_bytes)
    client = _PutFailsFakeS3(fake_s3, failing_key="out/upload.jpg")
    todo_df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "image_id": name,
                "original_s3_key": f"orig/{name}.jpg",
                "output_s3_key": f"out/{name}.jpg",
            }
            for name in ("ok", "bad", "upload")
        ]
    )

    counts = resize_and_upload_all_images(df_todo=todo_df, bucket="b", workers=2, s3_client=client)

    assert counts == (1, 1, 1)
    assert ("b", "out/ok.jpg") in fake_s3.objects
    assert ("b", "out/bad.jpg") not in fake_s3.objects


def test_validate_and_download_propagates_a_download_failure(fake_s3):
    with pytest.raises(ClientError):
        _validate_and_download("b", "orig/missing.jpg", 1, "x", fake_s3)


def test_resize_and_upload_image_fails_an_image_whose_download_fails(fake_s3):
    """A failed download is a retryable ``failed`` result, not a corrupted skip."""
    todo_item = {
        "source_id": 1,
        "image_id": "x",
        "original_s3_key": "orig/missing.jpg",
        "output_s3_key": "out/x.jpg",
    }

    result = resize_and_upload_image(todo_item=todo_item, bucket="b", s3_client=fake_s3)

    assert (result["status"], result["skip_reason"]) == ("failed", None)
    assert result["error_message"].startswith("ClientError: ")
    assert fake_s3.objects == {}
