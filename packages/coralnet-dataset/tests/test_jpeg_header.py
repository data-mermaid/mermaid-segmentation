"""Tests for the JPEG SOF parser and ranged-GET driver."""

from __future__ import annotations

from io import BytesIO

import pytest
from botocore.exceptions import ClientError
from PIL import Image

from coralnet_dataset.jpeg_header import (
    _file_size_from_content_range,
    _find_sof_dimensions,
    read_jpeg_size_from_s3,
)


class _FailsOnCall:
    """Wraps a FakeS3 so the ``fail_on``-th ``get_object`` call raises a ClientError."""

    def __init__(self, fake_s3, fail_on: int, code: str):
        self._fake = fake_s3
        self._fail_on = fail_on
        self._code = code
        self._calls = 0

    def get_object(self, **kwargs):
        self._calls += 1
        if self._calls == self._fail_on:
            raise ClientError({"Error": {"Code": self._code, "Message": self._code}}, "GetObject")
        return self._fake.get_object(**kwargs)


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
    result = read_jpeg_size_from_s3(fake_s3, "b", "corrupt.jpg")

    assert result == (None, None, len(jpeg_corrupt_bytes), "corrupt")


def test_read_jpeg_size_from_s3_pil_fallback_for_a_header_without_sof(fake_s3):
    buf = BytesIO()
    Image.new("RGB", (64, 48)).save(buf, format="PNG")
    png = buf.getvalue()
    fake_s3.put("b", "img.png", png)

    result = read_jpeg_size_from_s3(fake_s3, "b", "img.png")

    assert result == (64, 48, len(png), "pil_fallback")


@pytest.mark.parametrize(
    ("fail_on", "expect_size"),
    [(1, False), (2, True)],
    ids=["first_read", "extended_read"],
)
def test_read_jpeg_size_from_s3_non_404_error_is_s3_error(
    fake_s3, jpeg_extended_bytes, fail_on, expect_size
):
    fake_s3.put("b", "img.jpg", jpeg_extended_bytes)
    client = _FailsOnCall(fake_s3, fail_on=fail_on, code="500")

    result = read_jpeg_size_from_s3(client, "b", "img.jpg", first_chunk=4096)

    # The extended read runs only after the first read returned the total size.
    size = len(jpeg_extended_bytes) if expect_size else None
    assert result == (None, None, size, "s3_error")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"ContentRange": "bytes 0-65535/4359133"}, 4359133),
        ({"contentRange": "bytes 0-9/77"}, 77),
        ({}, None),
        ({"ContentRange": "bytes 0-9"}, None),
        ({"ContentRange": "bytes 0-9/abc"}, None),
    ],
    ids=["total", "camel_case_key", "missing", "no_slash", "not_int"],
)
def test_file_size_from_content_range(response, expected):
    assert _file_size_from_content_range(response) == expected


@pytest.mark.parametrize("marker", [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC9])
def test_find_sof_dimensions_various_sof_markers(make_jpeg, marker):
    data = make_jpeg(800, 600, sof_marker=marker)
    height, width = _find_sof_dimensions(data)
    assert (width, height) == (800, 600)


@pytest.mark.parametrize("marker", [0xC4, 0xC8, 0xCC], ids=["DHT", "JPG", "DAC"])
def test_find_sof_dimensions_ignores_non_sof_markers_in_the_sof_range(make_jpeg, marker):
    assert _find_sof_dimensions(make_jpeg(800, 600, sof_marker=marker)) is None
