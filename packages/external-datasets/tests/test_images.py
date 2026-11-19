from __future__ import annotations

import io

from PIL import Image

from external_datasets.images import find_undecodable


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


def test_find_undecodable_reports_only_bad_files_with_error_text(tmp_path):
    good = tmp_path / "good.png"
    Image.new("RGB", (4, 4)).save(good)
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(_jpeg_bytes()[:100])
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    missing = tmp_path / "missing.jpg"

    bad = find_undecodable([good, truncated, empty, missing], workers=2)

    assert sorted(p.name for p, _ in bad) == ["empty.jpg", "missing.jpg", "truncated.jpg"]
    assert all(err for _, err in bad)
