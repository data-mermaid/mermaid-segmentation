"""Shared fixtures for the CoralNet dataset tests.

Image-byte and CSV builders for the scraped-source tests; ``fake_s3`` is in the root conftest.
"""

from __future__ import annotations

import struct

import pytest
from s3_fake import FakeS3

__all__ = ["populate_source"]


def _make_jpeg_bytes(width: int, height: int, *, sof_marker: int = 0xC0, padding: int = 0) -> bytes:
    """Synthesise a minimal JPEG byte sequence with a valid SOF marker.

    The bytes are not a fully decodable JPEG but the parser only looks at the SOF marker, which
    carries (height, width). ``padding`` inserts junk APP0 segments before the SOF so we can
    exercise the extended-read path. JPEG segments are capped at 65,533 bytes of payload (2-byte
    length field includes the 2 length bytes), so larger paddings are split across multiple
    segments.
    """
    out = bytearray(b"\xff\xd8")  # SOI
    remaining = padding
    _SEG_MAX = 65_533  # max payload per APP0 segment
    while remaining > 0:
        chunk = min(remaining, _SEG_MAX)
        out += b"\xff\xe0"
        out += struct.pack(">H", chunk + 2)
        out += b"\x00" * chunk
        remaining -= chunk
    # SOFn: marker(2) + length(2) + precision(1) + height(2) + width(2) + nf(1)
    out += bytes([0xFF, sof_marker])
    out += struct.pack(">H", 8)
    out += bytes([8])
    out += struct.pack(">H", height)
    out += struct.pack(">H", width)
    out += bytes([1])
    out += b"\xff\xd9"  # EOI
    return bytes(out)


def _csv_bytes(headers: list[str], rows: list[list]) -> bytes:
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture
def jpeg_small_bytes() -> bytes:
    return _make_jpeg_bytes(1024, 768)


@pytest.fixture
def jpeg_extended_bytes() -> bytes:
    """JPEG whose SOF is past the first 64 KB so it triggers the extended-read path."""
    return _make_jpeg_bytes(640, 480, padding=80_000)


@pytest.fixture
def jpeg_corrupt_bytes() -> bytes:
    return b"not really a jpeg" * 100


@pytest.fixture
def make_jpeg():
    """Factory for inline JPEG byte construction in tests."""
    return _make_jpeg_bytes


@pytest.fixture
def sample_csv_factory():
    """Returns a callable that builds (annotations, image_list) CSV bytes for a fake source."""

    def _factory(
        source_id: int = 1,
        n_images: int = 3,
        n_points_per_image: int = 5,
        labelset: bool = True,
    ) -> dict[str, bytes]:
        ann_headers = ["Name", "Row", "Column", "Label ID", "Status"]
        img_headers = ["Name", "Image Page"]
        ann_rows: list[list] = []
        img_rows: list[list] = []
        for img_idx in range(n_images):
            name = f"img_{source_id}_{img_idx}.jpg"
            image_id = 10_000 * source_id + img_idx
            img_rows.append([name, f"/image/{image_id}/view/"])
            for pt in range(n_points_per_image):
                ann_rows.append([name, 100 + pt, 200 + pt, 1000 + (pt % 7), "Confirmed"])
        files: dict[str, bytes] = {
            "annotations.csv": _csv_bytes(ann_headers, ann_rows),
            "image_list.csv": _csv_bytes(img_headers, img_rows),
        }
        if labelset:
            files["labelset.csv"] = _csv_bytes(
                ["Label ID", "Name"], [[1000 + i, f"label_{i}"] for i in range(7)]
            )
        return files

    return _factory


def populate_source(
    s3: FakeS3,
    bucket: str,
    prefix: str,
    source_id: int,
    files: dict[str, bytes],
    *,
    complete: bool = True,
) -> None:
    """Put a source's files, plus the scraper's ``_complete.json`` marker unless ``complete``
    is False."""
    base = f"{prefix}/s{source_id}/"
    for name, body in files.items():
        s3.put(bucket, base + name, body)
    if complete:
        s3.put(bucket, base + "_complete.json", b"{}")


@pytest.fixture
def populate():
    return populate_source
