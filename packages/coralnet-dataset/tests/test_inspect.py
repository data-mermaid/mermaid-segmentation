"""Tests for the image inspection used to validate images before resizing."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from coralnet_dataset import inspect as inspect_module
from coralnet_dataset.inspect import IssueType, inspect_image


def _png_bytes(mode: str = "RGB") -> bytes:
    buf = BytesIO()
    Image.new(mode, (64, 48)).save(buf, format="PNG")
    return buf.getvalue()


def test_inspect_image_valid_rgb(valid_rgb_jpeg_bytes):
    """Inspection passes for valid RGB JPEG."""
    image_bytes = BytesIO(valid_rgb_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is True
    assert inspection.issue_type == IssueType.VALID
    assert inspection.channels == 3
    assert inspection.format == "JPEG"
    assert inspection.width == 2048
    assert inspection.height == 1536


def test_inspect_image_rgba_png(rgba_png_bytes):
    """Inspection passes for RGBA PNG; the resize step converts it to RGB before JPEG
    encoding."""
    image_bytes = BytesIO(rgba_png_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is True
    assert inspection.issue_type == IssueType.VALID
    assert inspection.channels == 4
    assert inspection.format == "PNG"


def test_inspect_image_grayscale(grayscale_jpeg_bytes):
    """Inspection passes for grayscale JPEG; the resize step encodes single-channel L
    directly."""
    image_bytes = BytesIO(grayscale_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is True
    assert inspection.issue_type == IssueType.VALID
    assert inspection.channels == 1
    assert inspection.format == "JPEG"


def test_inspect_image_truncated(truncated_jpeg_bytes):
    """Inspection reports a JPEG cut off mid-stream as truncated."""
    inspection = inspect_image(BytesIO(truncated_jpeg_bytes))

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.TRUNCATED
    assert inspection.error_message is not None


def test_inspect_image_corrupted_sof_marker_is_unsupported_format(corrupted_sof_jpeg_bytes):
    """A JPEG whose SOF marker is destroyed no longer identifies as an image."""
    inspection = inspect_image(BytesIO(corrupted_sof_jpeg_bytes))

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.UNSUPPORTED_FORMAT
    assert inspection.format is None


def test_inspect_image_two_band_image_has_invalid_channels():
    """An LA (grayscale + alpha) PNG has 2 bands, which the resize step cannot convert."""
    inspection = inspect_image(BytesIO(_png_bytes("LA")))

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.INVALID_CHANNELS
    assert (inspection.format, inspection.channels) == ("PNG", 2)
    assert inspection.error_message == "Expected 1 (L), 3 (RGB), or 4 (RGBA) channels, got 2"


def test_inspect_image_corrupt_pixel_data_is_decode_failure():
    """Zeroing the PNG's zlib stream breaks the decode, not the header."""
    png = _png_bytes()
    idat = png.index(b"IDAT")
    length = int.from_bytes(png[idat - 4 : idat], "big")
    broken = png[: idat + 4] + b"\x00" * length + png[idat + 4 + length :]

    inspection = inspect_image(BytesIO(broken))

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.DECODE_FAILURE
    assert inspection.error_message.startswith("PIL decode error: ")


def test_inspect_image_non_oserror_failure_is_unknown(monkeypatch):
    """A decompression bomb raises outside PIL's OSError family."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    inspection = inspect_image(BytesIO(_png_bytes()))

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.UNKNOWN
    assert inspection.error_message.startswith("DecompressionBombError: ")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("some other decoder error", IssueType.DECODE_FAILURE),
        ("unexpected EOF", IssueType.TRUNCATED),
    ],
)
def test_inspect_image_classifies_oserror_messages(monkeypatch, message, expected):
    """Stubbing ``Image.open`` reaches the message keywords real input rarely produces."""

    def _raise(_fp):
        raise OSError(message)

    monkeypatch.setattr(inspect_module.Image, "open", _raise)

    inspection = inspect_image(BytesIO(b"x"))

    assert inspection.issue_type == expected
    assert inspection.error_message == f"PIL decode error: {message}"


@pytest.mark.parametrize("image_bytes", [None, BytesIO(b"")], ids=["none", "empty"])
def test_inspect_image_empty(image_bytes):
    """Inspection fails for missing or empty data."""
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.ZERO_SIZE
    assert "Empty" in inspection.error_message
