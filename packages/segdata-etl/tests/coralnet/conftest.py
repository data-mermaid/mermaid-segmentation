"""Shared fixtures for the CoralNet dataset tests.

The in-memory ``FakeS3`` comes from ``s3_fake``; it keeps the suite free of network and AWS deps.
"""

from __future__ import annotations

import io
import struct

import pytest
from PIL import Image
from s3_fake import FakeS3

__all__ = ["FakeS3", "populate_source"]


@pytest.fixture(autouse=True)
def _fake_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give boto3 static credentials so it never queries the EC2 metadata endpoint."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


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
def fake_s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def jpeg_small_bytes() -> bytes:
    return _make_jpeg_bytes(1024, 768)


@pytest.fixture
def jpeg_large_bytes() -> bytes:
    """JPEG with longest_edge=4096 (> 2048 default threshold => needs_resize)."""
    return _make_jpeg_bytes(4096, 3000)


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


# ============================================================================
# Image corruption and format fixtures for preprocessing robustness tests
# ============================================================================


@pytest.fixture
def valid_rgb_jpeg_bytes() -> bytes:
    """Valid RGB JPEG (3 channels, 2048x1536)."""
    img = Image.new("RGB", (2048, 1536), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def rgba_png_bytes() -> bytes:
    """RGBA PNG (4 channels, 1024x768) — recoverable; resize converts it to RGB."""
    img = Image.new("RGBA", (1024, 768), color=(255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def grayscale_jpeg_bytes() -> bytes:
    """Grayscale JPEG (1 channel, L mode) — recoverable; resize encodes L directly."""
    img = Image.new("L", (1024, 768), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def truncated_jpeg_bytes() -> bytes:
    """Truncated JPEG (valid header but missing end marker and data)."""
    img = Image.new("RGB", (2048, 1536), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    jpeg_data = buf.getvalue()
    # Keep only first 50% of data, truncating mid-stream
    return jpeg_data[: len(jpeg_data) // 2]


@pytest.fixture
def corrupted_sof_jpeg_bytes() -> bytes:
    """Decodable JPEG (with scan data) whose SOF0 marker bytes are zeroed."""
    img = Image.new("RGB", (64, 48), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    sof = data.index(b"\xff\xc0")
    return data[:sof] + b"\x00\x00" + data[sof + 2 :]
