import io

import pyarrow as pa
import pytest
from PIL import Image
from pydantic import ValidationError

from segdata_spec.resize import ResizeSpec, resize_image, resize_mask, scale_points, scaled_size


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _jpeg(size: tuple[int, int], mode: str = "RGB") -> bytes:
    return _encode(Image.new(mode, size), "JPEG")


def _decode(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def test_default_spec_id_is_a_pinned_hash():
    # sha256 of {"filter":"lanczos","jpeg_quality":95,"max_edge":2048}, first 12 hex chars.
    assert ResizeSpec(max_edge=2048).id == "b6015b32d687"


@pytest.mark.parametrize(
    "other",
    [
        ResizeSpec(max_edge=1024),
        ResizeSpec(max_edge=2048, filter="bicubic"),
        ResizeSpec(max_edge=2048, jpeg_quality=90),
    ],
)
def test_spec_id_changes_with_each_field(other):
    assert other.id != ResizeSpec(max_edge=2048).id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_edge": 0},
        {"max_edge": -1},
        {"max_edge": 10, "jpeg_quality": 0},
        {"max_edge": 10, "jpeg_quality": 96},
        {"max_edge": 10, "filter": "nearest"},
    ],
)
def test_spec_rejects_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        ResizeSpec(**kwargs)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((4000, 3000), (2048, 1536)),
        ((6000, 4000), (2048, 1365)),
        ((3000, 4000), (1536, 2048)),
        # int(2215 * (2048 / 2215)) is 2047 in float arithmetic, not 2048.
        ((2215, 1000), (2047, 924)),
        ((2048, 1536), (2048, 1536)),
        ((2049, 1000), (2048, 999)),
    ],
)
def test_scaled_size(size, expected):
    assert scaled_size(*size, ResizeSpec(max_edge=2048)) == expected


def test_resize_image_downsizes_a_large_jpeg():
    out = _decode(resize_image(_jpeg((4000, 3000)), ResizeSpec(max_edge=2048)))

    assert (out.format, out.size) == ("JPEG", (2048, 1536))


def test_resize_image_converts_rgba_png_to_rgb_jpeg_within_limit():
    data = _encode(Image.new("RGBA", (100, 50)), "PNG")

    out = _decode(resize_image(data, ResizeSpec(max_edge=2048)))

    assert (out.format, out.mode, out.size) == ("JPEG", "RGB", (100, 50))


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_resize_image_passes_through_jpeg_within_limit(mode):
    data = _jpeg((2048, 100), mode)

    assert resize_image(data, ResizeSpec(max_edge=2048)) == data


def test_resize_image_reencodes_cmyk_jpeg_within_limit():
    data = _jpeg((64, 32), "CMYK")

    out = _decode(resize_image(data, ResizeSpec(max_edge=2048)))

    assert (out.format, out.mode, out.size) == ("JPEG", "RGB", (64, 32))


def test_resize_image_raises_on_decompression_bomb(monkeypatch):
    data = _jpeg((100, 100))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(Image.DecompressionBombError):
        resize_image(data, ResizeSpec(max_edge=2048))


def test_resize_mask_uses_nearest_so_label_ids_survive():
    # 4x2 grid of 2x2 blocks; nearest halving must return exactly the block values.
    blocks = [[0, 3, 7, 0], [7, 0, 3, 7]]
    mask = Image.new("L", (8, 4))
    for by, row in enumerate(blocks):
        for bx, value in enumerate(row):
            mask.paste(value, (bx * 2, by * 2, bx * 2 + 2, by * 2 + 2))

    out = _decode(resize_mask(_encode(mask, "PNG"), (4, 2)))

    assert (out.format, out.mode, out.size) == ("PNG", "L", (4, 2))
    assert list(out.tobytes()) == [0, 3, 7, 0, 7, 0, 3, 7]


def test_resize_mask_returns_same_bytes_at_target_size():
    data = _encode(Image.new("L", (8, 4)), "PNG")

    assert resize_mask(data, (8, 4)) == data


def _images(*sizes: tuple[str, int, int]) -> pa.Table:
    return pa.table(
        {
            "image_id": [s[0] for s in sizes],
            "width": pa.array([s[1] for s in sizes], pa.int32()),
            "height": pa.array([s[2] for s in sizes], pa.int32()),
        }
    )


def _points(rows: list[tuple[str, int, int]]) -> pa.Table:
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "row": pa.array([r[1] for r in rows], pa.int32()),
            "col": pa.array([r[2] for r in rows], pa.int32()),
            "label_id": pa.array(range(len(rows)), pa.int32()),
            "annotator": ["x"] * len(rows),
        }
    )


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # 4000x3000 -> 2048x1536 (0.512); a point is (row, col) so row follows height.
        ([("big", 1500, 2000), ("big", 2999, 3999)], [(768, 1024), (1535, 2047)]),
        # Within the limit nothing moves.
        ([("small", 1199, 1599)], [(1199, 1599)]),
        # Out-of-range points clamp to size - 1, and below zero to 0.
        ([("big", 3000, 4000), ("big", -5, -1)], [(1535, 2047), (0, 0)]),
        # 2215x1000 -> 2047x924: no clamp, ETL float order.
        ([("odd", 999, 2214)], [(923, 2046)]),
    ],
)
def test_scale_points(rows, expected):
    images = _images(("big", 4000, 3000), ("small", 1600, 1200), ("odd", 2215, 1000))

    out = scale_points(_points(rows), images, ResizeSpec(max_edge=2048))

    assert list(zip(out["row"].to_pylist(), out["col"].to_pylist(), strict=True)) == expected


def test_scale_points_raises_on_image_id_missing_from_images():
    points = _points([("a", 1, 2), ("zzz", 3, 4)])
    images = _images(("a", 4000, 3000))

    with pytest.raises(ValueError, match="missing from images"):
        scale_points(points, images, ResizeSpec(max_edge=2048))


def test_scale_points_keeps_columns_order_and_types():
    points = _points([("small", 1, 2), ("big", 3, 4), ("small", 5, 6)])
    images = _images(("big", 4000, 3000), ("small", 1600, 1200))

    out = scale_points(points, images, ResizeSpec(max_edge=2048))

    assert out.schema.equals(points.schema)
    assert out["image_id"].to_pylist() == ["small", "big", "small"]
    assert out["label_id"].to_pylist() == [0, 1, 2]
