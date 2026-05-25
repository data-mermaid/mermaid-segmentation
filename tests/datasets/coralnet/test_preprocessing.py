"""Unit tests for preprocessing module."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from mermaidseg.datasets.coralnet.preprocessing.resize import resize_image_to_threshold


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
