"""Tests for the training-manifest builder."""

from __future__ import annotations

from io import BytesIO

import ibis
import pandas as pd
import pytest
from PIL import Image

from coralnet_dataset.manifest import build_training_manifest, resized_existence_table
from coralnet_dataset.resize import resize_image_to_threshold


def test_training_manifest_scales_coords_and_excludes():
    """Sub-threshold coords unchanged + original key; resized coords floored to load
    dims + resized key; needs_resize images with no resized object on S3 excluded."""
    images = pd.DataFrame(
        {
            "source_id": [1, 1, 1, 1, 1],
            "image_id": ["sub", "big", "fail", "missing", "nodims"],
            "s3_key": [
                "coralnet-public-images/s1/images/sub.jpg",
                "coralnet-public-images/s1/images/big.jpg",
                "coralnet-public-images/s1/images/fail.jpg",
                "coralnet-public-images/s1/images/missing.jpg",
                "coralnet-public-images/s1/images/nodims.jpg",
            ],
            # "nodims" has unknown dimensions (e.g. header_status="not_found").
            "width": [1000, 4000, 5000, 6000, None],
            "height": [800, 2000, 3000, 3000, None],
            "needs_resize": [False, True, True, True, False],
        }
    )
    # Only "big" has a resized object; "fail" and "missing" need resizing but have none.
    resized = pd.DataFrame({"source_id": [1], "image_id": ["big"], "status": ["completed"]})
    annotations = pd.DataFrame(
        {
            "source_id": [1, 1, 1, 1, 1, 1],
            "image_id": ["sub", "big", "big", "fail", "missing", "nodims"],
            "row": [400, 1000, 1999, 10, 10, 10],
            "col": [500, 2000, 3999, 10, 10, 10],
            "coralnet_id": [82, 91, 91, 7, 7, 7],
        }
    )

    out = build_training_manifest(
        annotations=ibis.memtable(annotations),
        images=ibis.memtable(images),
        resized=ibis.memtable(resized),
        output_prefix="dev/images",
    ).to_pandas()

    # Only sub-threshold + completed-resize images with known dims survive
    # ("fail"/"missing" excluded for a missing resized object, "nodims" for null dimensions).
    assert set(out["image_id"]) == {"sub", "big"}

    # Every surviving point is in bounds (no nulls, no out-of-range from null-dim collapse).
    assert out["load_width"].notna().all() and out["load_height"].notna().all()
    assert (out["row"] >= 0).all() and (out["row"] < out["load_height"]).all()
    assert (out["col"] >= 0).all() and (out["col"] < out["load_width"]).all()

    # Coords/dims are stored as int16 (bounded by the 2048 resize threshold) to keep the file small.
    for c in ("row", "col", "load_width", "load_height"):
        assert out[c].dtype == "int16", (c, out[c].dtype)

    # Sub-threshold image: original key, coords unchanged, source_label_name = str(coralnet_id).
    sub = out[out["image_id"] == "sub"].iloc[0]
    assert sub["image_s3_key"] == "coralnet-public-images/s1/images/sub.jpg"
    assert (int(sub["row"]), int(sub["col"])) == (400, 500)
    assert bool(sub["uses_resized_image"]) is False
    assert sub["source_label_name"] == "82"

    # Resized image: resized key, dims floored to threshold (4000 -> 2048, 2000 -> 1024).
    big = out[out["image_id"] == "big"]
    assert (big["image_s3_key"] == "dev/images/resized/s1/images/big.jpg").all()
    assert big["uses_resized_image"].all()
    assert (big["load_width"] == 2048).all() and (big["load_height"] == 1024).all()
    coords = set(zip(big["row"].astype(int), big["col"].astype(int), strict=True))
    # (1000,2000) -> (floor(1000*1024/2000), floor(2000*2048/4000)) = (512, 1024)
    assert (512, 1024) in coords
    # (1999,3999) -> (floor(1999*1024/2000), floor(3999*2048/4000)) = (1023, 2047), within bounds
    assert (1023, 2047) in coords


def _build_one_image_manifest(orig_w, orig_h, row, col):
    """Run build_training_manifest for a single image+point; return (out_df,
    needs_resize)."""
    needs = max(orig_w, orig_h) > 2048
    images = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["x"],
            "s3_key": ["coralnet-public-images/s1/images/x.jpg"],
            "width": [orig_w],
            "height": [orig_h],
            "needs_resize": [needs],
        }
    )
    resized = pd.DataFrame({"source_id": [1], "image_id": ["x"], "status": ["completed"]})
    annotations = pd.DataFrame(
        {"source_id": [1], "image_id": ["x"], "row": [row], "col": [col], "coralnet_id": [82]}
    )
    out = build_training_manifest(
        annotations=ibis.memtable(annotations),
        images=ibis.memtable(images),
        resized=ibis.memtable(resized),
        output_prefix="dev/images",
    ).to_pandas()
    return out, needs


@pytest.mark.parametrize(
    "orig_w,orig_h",
    [
        (4000, 2000),  # landscape, resized
        (2000, 4000),  # portrait, resized
        (3000, 3000),  # square, resized (non-power-of-2 ratio)
        (2049, 2049),  # just over threshold
        (800, 600),  # sub-threshold, not resized
        (2048, 1000),  # longest == threshold, not resized
    ],
)
def test_training_manifest_load_dims_and_coords_match_real_resize(orig_w, orig_h):
    """Builder load dims + scaled coords agree with the actual PIL resize (oracle), and
    a far-corner point stays in bounds after the floor+clip."""
    r, c = orig_h - 1, orig_w - 1  # far corner stresses the clip
    out, needs = _build_one_image_manifest(orig_w, orig_h, r, c)
    assert len(out) == 1
    rowd = out.iloc[0]

    buf = BytesIO()
    Image.new("RGB", (orig_w, orig_h)).save(buf, format="JPEG")
    buf.seek(0)
    resized = Image.open(resize_image_to_threshold(buf))
    resized.load()

    # Oracle: recorded load dims == dimensions the real resize produced.
    assert int(rowd["load_width"]) == resized.width
    assert int(rowd["load_height"]) == resized.height
    assert bool(rowd["uses_resized_image"]) == needs

    # Coords scale by load/orig (same factor the builder uses) and stay in bounds.
    assert int(rowd["col"]) == min(int(c * resized.width / orig_w), resized.width - 1)
    assert int(rowd["row"]) == min(int(r * resized.height / orig_h), resized.height - 1)
    assert 0 <= int(rowd["row"]) < resized.height
    assert 0 <= int(rowd["col"]) < resized.width


def test_resized_existence_table_parses_resized_keys_only():
    table = resized_existence_table(
        ["dev/images/resized/s5/images/9.jpg", "dev/images/resized/other.txt"]
    )
    assert table.to_dict("records") == [{"source_id": 5, "image_id": "9", "status": "completed"}]
