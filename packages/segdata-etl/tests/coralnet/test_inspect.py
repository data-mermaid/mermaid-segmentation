"""Tests for the image inspection used to validate images before resizing."""

from __future__ import annotations

import os
from io import BytesIO

import pytest
from PIL import Image

from segdata_etl.common import images as inspect_module
from segdata_etl.common.images import IssueType, decode_image, inspect_image


def _png_bytes(mode: str = "RGB") -> bytes:
    buf = BytesIO()
    Image.new(mode, (64, 48)).save(buf, format="PNG")
    return buf.getvalue()


def _noise_jpeg_bytes(size: tuple[int, int]) -> bytes:
    """A JPEG of random pixels, so the entropy-coded data is large enough to truncate."""
    noise = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    buf = BytesIO()
    noise.save(buf, format="JPEG")
    return buf.getvalue()


def _jpeg_bytes(size: tuple[int, int]) -> BytesIO:
    buf = BytesIO()
    Image.new("RGB", size, color="red").save(buf, format="JPEG")
    return buf


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


def test_inspect_image_decompression_bomb_is_oversized(monkeypatch):
    """A decompression bomb is an oversized image, not a corrupted one."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    inspection = inspect_image(BytesIO(_png_bytes()))

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.OVERSIZED
    assert inspection.error_message.startswith("DecompressionBombError: ")


def test_inspect_image_non_oserror_failure_is_unknown(monkeypatch):
    """An exception outside PIL's OSError family and not a bomb is reported as unknown."""

    def _raise(_fp):
        raise ValueError("boom")

    monkeypatch.setattr(inspect_module.Image, "open", _raise)

    inspection = inspect_image(BytesIO(b"x"))

    assert inspection.issue_type == IssueType.UNKNOWN
    assert inspection.error_message == "ValueError: boom"


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        # A square (max_edge, max_edge) box would leave 1000x200 undrafted.
        ((1000, 200), (250, 50)),
        ((1000, 1000), (250, 250)),
        ((200, 1000), (50, 250)),
    ],
)
def test_decode_image_drafts_a_jpeg_by_its_longest_edge(original, expected):
    img, inspection = decode_image(_jpeg_bytes(original), max_edge=250)

    assert img.size == expected
    assert inspection.issue_type == IssueType.VALID
    assert (inspection.width, inspection.height) == original


def test_decode_image_without_max_edge_decodes_at_full_size():
    img, _ = decode_image(_jpeg_bytes((4096, 4096)))

    assert img.size == (4096, 4096)


def test_decode_image_truncated_jpeg_is_truncated_when_drafted():
    data = _noise_jpeg_bytes((1000, 1000))

    img, inspection = decode_image(BytesIO(data[: len(data) // 2]), max_edge=250)

    assert img is None
    assert inspection.issue_type == IssueType.TRUNCATED


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
