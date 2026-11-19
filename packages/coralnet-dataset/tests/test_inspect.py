"""Tests for the image inspection used to validate images before resizing."""

from __future__ import annotations

from io import BytesIO

import pytest

from coralnet_dataset.inspect import IssueType, inspect_image


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
    """Inspection fails for truncated JPEG."""
    image_bytes = BytesIO(truncated_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is False
    assert inspection.issue_type in (IssueType.TRUNCATED, IssueType.DECODE_FAILURE)
    assert inspection.error_message is not None


def test_inspect_image_corrupted_header(corrupted_header_jpeg_bytes):
    """Inspection fails for JPEG with corrupted header."""
    image_bytes = BytesIO(corrupted_header_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is False
    assert inspection.issue_type in (
        IssueType.DECODE_FAILURE,
        IssueType.CORRUPTED_HEADER,
        IssueType.UNSUPPORTED_FORMAT,
    )


@pytest.mark.parametrize("image_bytes", [None, BytesIO(b"")], ids=["none", "empty"])
def test_inspect_image_empty(image_bytes):
    """Inspection fails for missing or empty data."""
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.ZERO_SIZE
    assert "Empty" in inspection.error_message
