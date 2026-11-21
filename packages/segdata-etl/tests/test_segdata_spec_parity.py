"""segdata_spec.resize reproduces the legacy CoralNet resize path byte-for-byte and point-for-point."""

from __future__ import annotations

from io import BytesIO

import ibis
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from segdata_etl.common.images import decode_image
from segdata_etl.datasets.coralnet.config import RESIZE_THRESHOLD
from segdata_etl.datasets.coralnet.manifest import build_training_manifest, resized_existence_table
from segdata_etl.datasets.coralnet.resize import resize_image_to_threshold
from segdata_spec.resize import ResizeSpec, resize_image, scale_points

SPEC = ResizeSpec(max_edge=RESIZE_THRESHOLD)


def _textured_image(width: int, height: int, mode: str) -> Image.Image:
    """Gradient plus fixed-seed noise, so JPEG and Lanczos have texture to disagree on."""
    rng = np.random.default_rng(width * 10_000 + height)
    ramp = np.linspace(0, 200, width, dtype=np.float32)[None, :, None] + np.zeros((height, 1, 3))
    noise = rng.integers(0, 55, size=(height, width, 3))
    pixels = (ramp + noise).astype(np.uint8)
    if mode == "RGBA":
        alpha = np.full((height, width, 1), 128, dtype=np.uint8)
        pixels = np.concatenate([pixels, alpha], axis=2)
    return Image.fromarray(pixels, mode)


def _encode(width: int, height: int, mode: str, fmt: str) -> bytes:
    buffer = BytesIO()
    _textured_image(width, height, mode).save(buffer, format=fmt)
    return buffer.getvalue()


def _legacy_resize(data: bytes) -> bytes:
    img, inspection = decode_image(BytesIO(data), max_edge=RESIZE_THRESHOLD)
    assert inspection.width is not None and inspection.height is not None
    return resize_image_to_threshold(
        img, BytesIO(data), original_size=(inspection.width, inspection.height)
    ).getvalue()


@pytest.mark.parametrize(
    ("width", "height", "mode", "fmt", "expected_size"),
    [
        pytest.param(4000, 3000, "RGB", "JPEG", (2048, 1536), id="landscape-jpeg"),
        pytest.param(3000, 4000, "RGB", "JPEG", (1536, 2048), id="portrait-jpeg"),
        pytest.param(2215, 1000, "RGB", "JPEG", (2047, 924), id="truncated-2047"),
        pytest.param(1024, 768, "RGB", "JPEG", (1024, 768), id="jpeg-pass-through"),
        pytest.param(300, 200, "RGBA", "PNG", (300, 200), id="rgba-png-reencoded"),
        pytest.param(300, 200, "RGB", "PNG", (300, 200), id="rgb-png-reencoded"),
        # Longest edge >= 2 x max_edge, so libjpeg decodes at a reduced scale.
        pytest.param(6000, 4000, "RGB", "JPEG", (2048, 1365), id="draft-landscape"),
        pytest.param(4000, 6000, "RGB", "JPEG", (1365, 2048), id="draft-portrait"),
    ],
)
def test_resize_image_matches_legacy_bytes(width, height, mode, fmt, expected_size):
    data = _encode(width, height, mode, fmt)

    out = resize_image(data, SPEC)

    assert out == _legacy_resize(data)
    assert Image.open(BytesIO(out)).size == expected_size


def test_scale_points_matches_manifest_row_col():
    # (image_id, width, height, needs_resize); the 2215x1000 image truncates to 2047 wide.
    image_rows = [
        ("big", 4000, 3000, True),
        ("trunc", 2215, 1000, True),
        ("small", 1600, 1200, False),
    ]
    points = [  # (image_id, row, col): interior plus the last row/col of each image
        ("big", 0, 0),
        ("big", 1500, 2000),
        ("big", 2999, 3999),
        ("big", 2999, 0),
        ("trunc", 500, 1107),
        ("trunc", 999, 2214),
        ("small", 600, 800),
        ("small", 1199, 1599),
    ]
    images = pd.DataFrame(
        {
            "source_id": 1,
            "image_id": [r[0] for r in image_rows],
            "s3_key": [f"k/{r[0]}.jpg" for r in image_rows],
            "width": [r[1] for r in image_rows],
            "height": [r[2] for r in image_rows],
            "needs_resize": [r[3] for r in image_rows],
        }
    )
    annotations = pd.DataFrame(
        {
            "source_id": 1,
            "image_id": [p[0] for p in points],
            "row": [p[1] for p in points],
            "col": [p[2] for p in points],
            "coralnet_id": range(len(points)),  # unique, to rejoin after the manifest reorders
        }
    )
    resized = resized_existence_table(
        [f"dev/resized/s1/images/{name}.jpg" for name in ("big", "trunc")]
    )

    manifest = build_training_manifest(
        ibis.memtable(annotations), ibis.memtable(images), ibis.memtable(resized), "dev"
    ).to_pandas()
    legacy = {int(r.coralnet_id): (int(r.row), int(r.col)) for r in manifest.itertuples()}

    scaled = scale_points(
        pa.table(
            {
                "image_id": [p[0] for p in points],
                "row": pa.array([p[1] for p in points], pa.int64()),
                "col": pa.array([p[2] for p in points], pa.int64()),
            }
        ),
        pa.table(
            {
                "image_id": [r[0] for r in image_rows],
                "width": [r[1] for r in image_rows],
                "height": [r[2] for r in image_rows],
            }
        ),
        SPEC,
    )
    ours = dict(enumerate(zip(scaled["row"].to_pylist(), scaled["col"].to_pylist(), strict=True)))

    assert ours == legacy
