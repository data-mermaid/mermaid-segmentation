"""Tests for the JPEG SOF parser and ranged-GET driver."""

from __future__ import annotations

import pytest

from mermaidseg.datasets.coralnet.etl.jpeg_header import (
    _find_sof_dimensions,
    read_jpeg_size_from_s3,
)


def test_find_sof_dimensions_baseline(jpeg_small_bytes):
    height, width = _find_sof_dimensions(jpeg_small_bytes)
    assert (width, height) == (1024, 768)


def test_find_sof_dimensions_handles_padding(jpeg_extended_bytes):
    height, width = _find_sof_dimensions(jpeg_extended_bytes)
    assert (width, height) == (640, 480)


def test_find_sof_dimensions_returns_none_for_garbage(jpeg_corrupt_bytes):
    assert _find_sof_dimensions(jpeg_corrupt_bytes) is None


def test_read_jpeg_size_from_s3_ok(fake_s3, jpeg_small_bytes):
    fake_s3.put("b", "img.jpg", jpeg_small_bytes)
    w, h, size, status = read_jpeg_size_from_s3(fake_s3, "b", "img.jpg")
    assert (w, h) == (1024, 768)
    assert size == len(jpeg_small_bytes)
    assert status == "ok"


def test_read_jpeg_size_from_s3_extended(fake_s3, jpeg_extended_bytes):
    fake_s3.put("b", "img.jpg", jpeg_extended_bytes)
    w, h, _, status = read_jpeg_size_from_s3(fake_s3, "b", "img.jpg", first_chunk=4096)
    assert (w, h) == (640, 480)
    assert status == "sof_not_found_extended"


def test_read_jpeg_size_from_s3_not_found(fake_s3):
    w, h, size, status = read_jpeg_size_from_s3(fake_s3, "b", "missing.jpg")
    assert (w, h, size) == (None, None, None)
    assert status == "not_found"


def test_read_jpeg_size_from_s3_corrupt(fake_s3, jpeg_corrupt_bytes):
    fake_s3.put("b", "corrupt.jpg", jpeg_corrupt_bytes)
    w, h, _, status = read_jpeg_size_from_s3(fake_s3, "b", "corrupt.jpg")
    # PIL also rejects it; final status is "corrupt".
    assert w is None and h is None
    assert status in {"corrupt", "pil_fallback"}


@pytest.mark.parametrize("marker", [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC9])
def test_find_sof_dimensions_various_sof_markers(make_jpeg, marker):
    data = make_jpeg(800, 600, sof_marker=marker)
    height, width = _find_sof_dimensions(data)
    assert (width, height) == (800, 600)
