"""Shared fixtures for CoralNet ETL tests.

Provides a hand-rolled in-memory ``FakeS3`` that implements the boto3 client surface the ETL touches
(``head_object``, ``get_object`` with ``Range``, ``list_objects_v2``, ``get_paginator``). Keeps the
test suite free of network and AWS deps.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field

import pytest
from botocore.exceptions import ClientError


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


@dataclass
class _StubResponse:
    body: bytes
    content_length: int
    content_range: str | None = None

    def __getitem__(self, key):
        if key == "Body":
            return _Body(self.body)
        if key == "ContentLength":
            return self.content_length
        if key == "ContentRange" and self.content_range is not None:
            return self.content_range
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _client_error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


@dataclass
class FakeS3:
    """In-memory S3 client surface used by the ETL.

    Populate ``objects`` with ``{(bucket, key): bytes}`` to control what gets returned. ``Range``
    semantics follow ``bytes=start-end`` (inclusive).
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.objects[(bucket, key)] = data

    def head_object(self, *, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", "GetObject")
        full = self.objects[(Bucket, Key)]
        content_range: str | None = None
        data = full
        if Range:
            assert Range.startswith("bytes=")
            start_str, end_str = Range[len("bytes=") :].split("-")
            start = int(start_str)
            end = int(end_str) if end_str else len(full) - 1
            data = full[start : end + 1]
            content_range = f"bytes {start}-{end}/{len(full)}"
        return _StubResponse(body=data, content_length=len(data), content_range=content_range)

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str = "", MaxKeys: int | None = None, Delimiter: str = ""
    ):
        keys = [k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix)]
        if Delimiter:
            common_prefixes: set[str] = set()
            contents = []
            for k in keys:
                rest = k[len(Prefix) :]
                if Delimiter in rest:
                    common_prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
                else:
                    contents.append({"Key": k})
            return {
                "CommonPrefixes": [{"Prefix": p} for p in sorted(common_prefixes)],
                "Contents": contents,
                "KeyCount": len(common_prefixes) + len(contents),
            }
        if MaxKeys is not None:
            keys = keys[:MaxKeys]
        return {"Contents": [{"Key": k} for k in keys], "KeyCount": len(keys)}

    def get_paginator(self, name: str):
        if name != "list_objects_v2":
            raise NotImplementedError(name)
        return _Paginator(self)

    def upload_file(self, *args, **kwargs) -> None:  # pragma: no cover - not exercised
        raise NotImplementedError


class _Paginator:
    def __init__(self, client: FakeS3):
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str = "", Delimiter: str = ""):
        yield self._client.list_objects_v2(Bucket=Bucket, Prefix=Prefix, Delimiter=Delimiter)


def _csv_bytes(headers: list[str], rows: list[list]) -> bytes:
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def pandas_csv_reader():
    """Build a :data:`CsvReader` over a FakeS3 fixture.

    Tests inject this in place of the production ibis-on-DuckDB reader so the code-path matches what
    the ETL uses end-to-end without depending on a real DuckDB connection or AWS.
    """
    from mermaidseg.datasets.coralnet.etl.io import make_csv_reader_s3

    def _factory(fake_s3: FakeS3):
        return make_csv_reader_s3(fake_s3)

    return _factory


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
        metadata: bool = True,
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
        if metadata:
            files["metadata.csv"] = _csv_bytes(
                ["Name", "Date"],
                [[f"img_{source_id}_{i}.jpg", "2024-01-01"] for i in range(n_images)],
            )
        return files

    return _factory


def populate_source(
    s3: FakeS3, bucket: str, prefix: str, source_id: int, files: dict[str, bytes]
) -> None:
    base = f"{prefix}/s{source_id}/"
    for name, body in files.items():
        s3.put(bucket, base + name, body)
    # Provide at least one image so the images-folder check passes.
    s3.put(bucket, base + "images/img_marker.jpg", b"\xff\xd8\xff\xd9")


@pytest.fixture
def populate():
    return populate_source


@pytest.fixture
def bytes_io():
    """Lazy BytesIO factory for tests that need a streaming buffer."""
    return io.BytesIO
