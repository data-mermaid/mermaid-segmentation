"""Property-based invariants for build_training_manifest coordinate scaling.

Uses hypothesis to fuzz image shapes / point positions / thresholds and assert invariants the
scaling must always satisfy. hypothesis is a dev-only convenience and is intentionally NOT in
pyproject.toml, so this module self-skips wherever it isn't installed.
"""

from __future__ import annotations

from datetime import datetime

import ibis
import pandas as pd
import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from mermaidseg.datasets.coralnet.preprocessing.manifest import (  # noqa: E402
    build_training_manifest,
)


@st.composite
def shape_point_threshold(draw):
    """Realistic image shape, an in-bounds point, and a resize threshold."""
    w = draw(st.integers(min_value=64, max_value=12000))
    h = draw(st.integers(min_value=64, max_value=12000))
    # Keep aspect ratios within a sane range so a tiny threshold can't floor an edge to 0.
    assume(0.1 <= w / h <= 10)
    col = draw(st.integers(min_value=0, max_value=w - 1))
    row = draw(st.integers(min_value=0, max_value=h - 1))
    threshold = draw(st.integers(min_value=256, max_value=4096))
    return w, h, row, col, threshold


@settings(max_examples=150, deadline=None)
@given(shape_point_threshold())
def test_scaling_invariants(params):
    w, h, row, col, threshold = params
    needs = max(w, h) > threshold

    images = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["x"],
            "s3_key": ["coralnet-public-images/s1/images/x.jpg"],
            "width": [w],
            "height": [h],
            "needs_resize": [needs],
        }
    )
    checkpoint = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["x"],
            "status": ["completed"],
            "resize_timestamp": [datetime.now()],
            "error_message": [None],
        }
    )
    annotations = pd.DataFrame(
        {"source_id": [1], "image_id": ["x"], "row": [row], "col": [col], "coralnet_id": [82]}
    )

    out = build_training_manifest(
        annotations=ibis.memtable(annotations),
        images=ibis.memtable(images),
        checkpoint=ibis.memtable(checkpoint),
        output_prefix="dev/images",
        threshold=threshold,
    ).to_pandas()

    # Image with valid dims is always kept (resized when over threshold, else original).
    assert len(out) == 1
    r = out.iloc[0]
    lw, lh = int(r["load_width"]), int(r["load_height"])
    out_row, out_col = int(r["row"]), int(r["col"])

    # Load dims never upscale and never exceed the threshold's longest edge.
    assert 1 <= lw <= w and 1 <= lh <= h
    assert max(lw, lh) <= threshold or not needs
    assert bool(r["uses_resized_image"]) == needs

    # Scaled point is always a valid index into the loaded image.
    assert 0 <= out_row < lh
    assert 0 <= out_col < lw

    # Downscaling never increases an index; a non-resized image is the identity.
    assert out_row <= row and out_col <= col
    if not needs:
        assert (out_row, out_col) == (row, col)
