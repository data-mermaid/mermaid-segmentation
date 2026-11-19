from __future__ import annotations

import io

import pytest
from PIL import Image, JpegImagePlugin

from external_datasets.images import find_undecodable


def _jpeg_bytes(size: tuple[int, int] = (256, 256)) -> bytes:
    # 256x256 by default so the 1/8 draft scale applies.
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


def test_find_undecodable_reports_only_bad_files_with_error_text(tmp_path):
    good = tmp_path / "good.png"
    Image.new("RGB", (4, 4)).save(good)
    good_jpeg = tmp_path / "good.jpg"
    good_jpeg.write_bytes(_jpeg_bytes())
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(_jpeg_bytes()[:100])
    tail = tmp_path / "tail_truncated.jpg"
    tail.write_bytes(_jpeg_bytes()[:-200])
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    missing = tmp_path / "missing.jpg"

    bad = find_undecodable([good, good_jpeg, truncated, tail, empty, missing], workers=2)

    assert sorted(p.name for p, _ in bad) == [
        "empty.jpg",
        "missing.jpg",
        "tail_truncated.jpg",
        "truncated.jpg",
    ]
    assert all(err for _, err in bad)


@pytest.fixture
def draft_calls(monkeypatch):
    calls = []

    def spy(self, mode, size):
        calls.append((mode, size))

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy)
    monkeypatch.setattr(Image.Image, "draft", spy)
    return calls


@pytest.mark.parametrize(
    ("size", "expected_box"),
    [
        ((256, 256), (32, 32)),
        ((260, 250), (32, 31)),
    ],
)
def test_jpeg_is_drafted_to_one_eighth_and_png_is_not(tmp_path, draft_calls, size, expected_box):
    jpg = tmp_path / "a.jpg"
    jpg.write_bytes(_jpeg_bytes(size))
    png = tmp_path / "a.png"
    Image.new("RGB", (256, 256)).save(png)

    assert find_undecodable([jpg, png], workers=1) == []
    assert draft_calls == [("RGB", expected_box)]
