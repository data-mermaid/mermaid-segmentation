"""Unit tests for preprocessing module."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, Mock

import ibis
import pandas as pd
import pytest
from PIL import Image

from mermaidseg.datasets.coralnet.preprocessing.inspect import (
    IssueType,
    inspect_image,
)
from mermaidseg.datasets.coralnet.preprocessing.manifest import (
    build_manifest,
    build_training_manifest,
    combine_checkpoints,
)
from mermaidseg.datasets.coralnet.preprocessing.resize import (
    _resized_s3_key_for,
    get_pending_items,
    read_checkpoint,
    resize_and_upload_all_images,
    resize_and_upload_image,
    resize_image_to_threshold,
    scan_for_missing_resized_images,
    write_checkpoint,
)


def test_resize_image_maintains_aspect_ratio():
    """Resizing longest edge > threshold maintains aspect ratio."""
    # Create 2000x1000 image (longest edge = 2000 > threshold of 1024)
    img = Image.new("RGB", (2000, 1000), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Resize to threshold=1024
    resized_bytes = resize_image_to_threshold(img_bytes, threshold=1024)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    # New width should be 1024, height scaled proportionally: 1024 * 1000 / 2000 = 512
    assert resized_img.width == 1024
    assert resized_img.height == 512


def test_resize_image_no_resize_if_below_threshold():
    """Image below threshold is not resized."""
    # Create 800x400 image (longest edge = 800 < threshold of 1024)
    img = Image.new("RGB", (800, 400), color="blue")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes, threshold=1024)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    # Should return original dimensions
    assert resized_img.width == 800
    assert resized_img.height == 400


def test_resize_image_square():
    """Resizing square image maintains square aspect ratio."""
    # Create 3000x3000 image
    img = Image.new("RGB", (3000, 3000), color="green")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes, threshold=2048)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    assert resized_img.width == 2048
    assert resized_img.height == 2048


def test_resize_converts_rgba_to_rgb_jpeg():
    """RGBA images are converted to RGB so they can be JPEG-encoded (not RGBA-as-JPEG error)."""
    img = Image.new("RGBA", (3000, 2000), color=(255, 0, 0, 128))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes, threshold=2048)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    assert resized_img.format == "JPEG"
    assert resized_img.mode == "RGB"
    assert resized_img.width == 2048
    assert resized_img.height == 1365


def test_resize_converts_rgba_below_threshold():
    """A small RGBA image is still re-encoded to RGB JPEG, not passed through as raw bytes."""
    img = Image.new("RGBA", (800, 400), color=(0, 255, 0, 64))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    resized_bytes = resize_image_to_threshold(img_bytes, threshold=1024)
    resized_img = Image.open(resized_bytes)
    resized_img.load()

    assert resized_img.format == "JPEG"
    assert resized_img.mode == "RGB"
    assert resized_img.width == 800
    assert resized_img.height == 400


def test_scan_for_missing_resized_images_builds_todo_list():
    """Scan lists the resized prefix once and diffs in memory — no per-image head_object."""
    df_images = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["img_a", "img_b", "img_c"],
            "width": [3000, 800, 2500],
            "height": [2000, 600, 1500],
            "needs_resize": [True, False, True],
            "s3_key": [
                "coralnet-public-images/s1/images/img_a.jpg",
                "coralnet-public-images/s1/images/img_b.jpg",
                "coralnet-public-images/s2/images/img_c.jpg",
            ],
        }
    )

    # Mock S3 client: the resized prefix already contains img_a (img_b has needs_resize=False)
    mock_s3 = Mock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "etl-outputs/coralnet/resized/s1/images/img_a.jpg"}]},
        {},  # empty page (no Contents key) must not crash
    ]
    mock_s3.get_paginator.return_value = mock_paginator

    # Run scan
    todo_df = scan_for_missing_resized_images(
        df_images=df_images,
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        threshold=2048,
        s3_client=mock_s3,
        workers=2,
    )

    # Existence comes from one paginated LIST, never per-image HEADs
    mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
    mock_paginator.paginate.assert_called_once_with(
        Bucket="test-bucket", Prefix="etl-outputs/coralnet/resized/"
    )
    mock_s3.head_object.assert_not_called()

    assert len(todo_df) == 1
    assert todo_df.iloc[0]["image_id"] == "img_c"
    assert todo_df.iloc[0]["source_id"] == 2
    assert todo_df.iloc[0]["original_s3_key"] == "coralnet-public-images/s2/images/img_c.jpg"
    assert todo_df.iloc[0]["output_s3_key"] == "etl-outputs/coralnet/resized/s2/images/img_c.jpg"
    assert "original_s3_key" in todo_df.columns
    assert "output_s3_key" in todo_df.columns


def test_checkpoint_write_and_read(tmp_path):
    """Checkpoint parquet can be written and read back."""
    checkpoint_path = tmp_path / "checkpoint.parquet"

    # Write
    df_checkpoint = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["img_a", "img_b", "img_c"],
            "status": ["completed", "pending", "failed"],
            "resize_timestamp": [datetime.now(), None, datetime.now()],
            "error_message": [None, None, "PIL decode failed"],
        }
    )

    write_checkpoint(checkpoint_path, df_checkpoint)

    # Read
    df_read = read_checkpoint(checkpoint_path)

    assert len(df_read) == 3
    assert list(df_read["status"]) == ["completed", "pending", "failed"]
    assert pd.isna(df_read.loc[1, "resize_timestamp"])


def test_checkpoint_pending_items_extracted():
    """Only pending items are returned for processing."""
    df_checkpoint = pd.DataFrame(
        {
            "source_id": [1, 1, 2, 2],
            "image_id": ["a", "b", "c", "d"],
            "status": ["completed", "pending", "failed", "pending"],
            "resize_timestamp": [datetime.now(), None, datetime.now(), None],
            "error_message": [None, None, "error", None],
        }
    )

    df_pending = get_pending_items(df_checkpoint)

    assert len(df_pending) == 2
    assert list(df_pending["image_id"]) == ["b", "d"]


def test_resize_and_upload_image_returns_completed_result():
    """Worker downloads/resizes/uploads and RETURNS a checkpoint row; it does no checkpoint I/O."""
    # Create mock S3 client
    mock_s3 = Mock()

    # Mock GET (download): return a 3000x2000 JPEG
    original_img = Image.new("RGB", (3000, 2000), color="red")
    img_bytes = BytesIO()
    original_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    mock_s3.get_object.return_value = {"Body": BytesIO(img_bytes.getvalue())}

    # Mock PUT (upload): track calls
    mock_s3.put_object = MagicMock()
    mock_s3.head_object = MagicMock()

    # Create todo item
    todo_item = {
        "source_id": 1,
        "image_id": "img_a",
        "width": 3000,
        "height": 2000,
        "original_s3_key": "coralnet-public-images/s1/images/img_a.jpg",
        "output_s3_key": "etl-outputs/coralnet/resized/s1/images/img_a.jpg",
    }

    # Run single image resize/upload — no checkpoint path involved
    result = resize_and_upload_image(
        todo_item=todo_item,
        bucket="test-bucket",
        threshold=2048,
        s3_client=mock_s3,
    )

    # Verify PUT was called
    assert mock_s3.put_object.called
    call_args = mock_s3.put_object.call_args
    assert call_args[1]["Bucket"] == "test-bucket"
    assert call_args[1]["Key"] == "etl-outputs/coralnet/resized/s1/images/img_a.jpg"

    # No post-upload head_object verify — S3 PUT is strongly consistent
    mock_s3.head_object.assert_not_called()

    # Verify the returned result row, not any on-disk checkpoint
    assert result["source_id"] == 1
    assert result["image_id"] == "img_a"
    assert result["status"] == "completed"
    assert result["resize_timestamp"] is not None
    assert result["skip_reason"] is None


def _checkpoint_table(df: pd.DataFrame) -> ibis.Table:
    """Memtable with the dtypes the checkpoint parquet schema guarantees in production."""
    return ibis.memtable(
        df.astype(
            {
                "resize_timestamp": "datetime64[ns]",
                "error_message": "string",
                "skip_reason": "string",
            }
        )
    )


def test_combine_checkpoints_later_run_wins():
    """Rows reprocessed in a later checkpoint override the earlier run's status."""
    ckpt_full = pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["a", "b"],
            "status": ["completed", "skipped"],
            "resize_timestamp": [datetime(2026, 5, 29), None],
            "error_message": [None, None],
            "skip_reason": [None, "corrupted_invalid_channels"],
        }
    )
    ckpt_rgba = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["b"],
            "status": ["completed"],
            "resize_timestamp": [datetime(2026, 6, 3)],
            "error_message": [None],
            "skip_reason": [None],
        }
    )

    combined = combine_checkpoints(
        [_checkpoint_table(ckpt_full), _checkpoint_table(ckpt_rgba)]
    ).to_pandas()

    assert len(combined) == 2
    by_id = combined.set_index("image_id")
    assert by_id.loc["a", "status"] == "completed"
    # 'b' was skipped in the first run but completed in the rgba rerun — later wins
    assert by_id.loc["b", "status"] == "completed"
    assert by_id.loc["b", "resize_timestamp"] == datetime(2026, 6, 3)


def test_combine_checkpoints_most_recent_timestamp_wins_regardless_of_order():
    """The freshest resize_timestamp wins even when checkpoints are passed out of order."""
    ckpt_newer = pd.DataFrame(
        {
            "source_id": [1],
            "image_id": ["a"],
            "status": ["completed"],
            "resize_timestamp": [datetime(2026, 6, 3)],
            "error_message": [None],
            "skip_reason": [None],
        }
    )
    ckpt_older = pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["a", "b"],
            "status": ["failed", "skipped"],
            "resize_timestamp": [None, None],
            "error_message": ["timeout", None],
            "skip_reason": [None, "corrupted_invalid_channels"],
        }
    )

    # Newer checkpoint listed first — positional keep="last" would wrongly pick the failed row
    combined = combine_checkpoints(
        [_checkpoint_table(ckpt_newer), _checkpoint_table(ckpt_older)]
    ).to_pandas()

    assert len(combined) == 2
    by_id = combined.set_index("image_id")
    assert by_id.loc["a", "status"] == "completed"
    assert by_id.loc["a", "resize_timestamp"] == datetime(2026, 6, 3)
    # 'b' has no timestamped row anywhere, so it keeps the later checkpoint's status
    assert by_id.loc["b", "status"] == "skipped"


def test_combine_checkpoints_empty_raises():
    """An empty checkpoint list is a caller error, not a silent empty table."""
    with pytest.raises(ValueError, match="at least one checkpoint"):
        combine_checkpoints([])


def test_combine_checkpoints_single_input_dedups():
    """A single checkpoint passes through, still deduped by latest timestamp per image."""
    ckpt = pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["a", "a"],
            "status": ["failed", "completed"],
            "resize_timestamp": [None, datetime(2026, 6, 3)],
            "error_message": ["timeout", None],
            "skip_reason": [None, None],
        }
    )

    combined = combine_checkpoints([_checkpoint_table(ckpt)]).to_pandas()

    assert len(combined) == 1
    assert combined.iloc[0]["status"] == "completed"


def test_manifest_schema_is_correct():
    """Manifest has required columns."""
    df_images = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["a", "b", "c"],
            "width": [3000, 800, 2500],
            "height": [2000, 600, 1500],
        }
    )

    checkpoint_df = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "image_id": ["a", "b", "c"],
            "status": ["completed", "failed", "completed"],
            "resize_timestamp": [datetime.now(), None, datetime.now()],
            "error_message": [None, "decode failed", None],
        }
    )

    output_prefix = "etl-outputs/coralnet"
    threshold = 2048

    manifest_df = build_manifest(
        images=ibis.memtable(df_images),
        checkpoint=ibis.memtable(checkpoint_df),
        output_prefix=output_prefix,
        threshold=threshold,
    ).to_pandas()

    # Check schema
    required_columns = [
        "source_id",
        "image_id",
        "original_width",
        "original_height",
        "resized_width",
        "resized_height",
        "output_s3_key",
        "resize_timestamp",
        "status",
    ]
    for col in required_columns:
        assert col in manifest_df.columns, f"Missing column: {col}"

    # Check data
    assert len(manifest_df) == 3
    assert manifest_df.loc[0, "original_width"] == 3000


def test_manifest_dimensions_and_keys_match_resize():
    """Below threshold keeps original dims; above threshold floors like the resize worker, and
    output_s3_key matches _resized_s3_key_for."""
    threshold = 2048
    df_images = pd.DataFrame(
        {
            "source_id": [1, 2],
            "image_id": ["below", "above"],
            "width": [800, 3000],
            "height": [600, 2000],
        }
    )
    checkpoint_df = pd.DataFrame(
        {
            "source_id": [1, 2],
            "image_id": ["below", "above"],
            "status": ["completed", "completed"],
            "resize_timestamp": [datetime.now(), datetime.now()],
            "error_message": [None, None],
        }
    )

    manifest = (
        build_manifest(
            images=ibis.memtable(df_images),
            checkpoint=ibis.memtable(checkpoint_df),
            output_prefix="etl-outputs/coralnet",
            threshold=threshold,
        )
        .to_pandas()
        .set_index("image_id")
    )

    # Below threshold: dimensions unchanged
    assert manifest.loc["below", "resized_width"] == 800
    assert manifest.loc["below", "resized_height"] == 600

    # Above threshold: same int() truncation the resize worker applies
    scale = threshold / 3000
    assert manifest.loc["above", "resized_width"] == int(3000 * scale)
    assert manifest.loc["above", "resized_height"] == int(2000 * scale)

    # Keys match the single source of truth in resize.py
    assert manifest.loc["above", "output_s3_key"] == _resized_s3_key_for(
        "etl-outputs/coralnet", 2, "above", threshold
    )


def test_training_manifest_scales_coords_and_excludes():
    """Sub-threshold coords unchanged + original key; resized coords floored to load dims + resized
    key; needs_resize-but-not-completed images (failed / absent from checkpoint) excluded."""
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
    # "missing" needs_resize but has no checkpoint row at all -> must be excluded.
    checkpoint = pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["big", "fail"],
            "status": ["completed", "failed"],
            "resize_timestamp": [datetime.now(), None],
            "error_message": [None, "decode failed"],
        }
    )
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
        checkpoint=ibis.memtable(checkpoint),
        output_prefix="dev/images",
        threshold=2048,
    ).to_pandas()

    # Only sub-threshold + completed-resize images with known dims survive
    # ("fail"/"missing" excluded for incomplete resize, "nodims" for null dimensions).
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


def _build_one_image_manifest(orig_w, orig_h, row, col, *, threshold=2048):
    """Run build_training_manifest for a single image+point; return (out_df, needs_resize)."""
    needs = max(orig_w, orig_h) > threshold
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
    """Builder load dims + scaled coords agree with the actual PIL resize (oracle), and a far-corner
    point stays in bounds after the floor+clip."""
    threshold = 2048
    r, c = orig_h - 1, orig_w - 1  # far corner stresses the clip
    out, needs = _build_one_image_manifest(orig_w, orig_h, r, c, threshold=threshold)
    assert len(out) == 1
    rowd = out.iloc[0]

    buf = BytesIO()
    Image.new("RGB", (orig_w, orig_h)).save(buf, format="JPEG")
    buf.seek(0)
    resized = Image.open(resize_image_to_threshold(buf, threshold=threshold))
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


# ============================================================================
# Image inspection and robustness tests
# ============================================================================


def test_inspect_image_valid_rgb(valid_rgb_jpeg_bytes):
    """Inspection passes for valid RGB JPEG."""
    image_bytes = BytesIO(valid_rgb_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is True
    assert inspection.issue_type == IssueType.VALID
    assert inspection.channels == 3
    assert inspection.format == "JPEG"
    assert inspection.width == 2048
    assert inspection.height == 1536


def test_inspect_image_rgba_png(rgba_png_bytes):
    """Inspection passes for RGBA PNG; the resize step converts it to RGB before JPEG encoding."""
    image_bytes = BytesIO(rgba_png_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is True
    assert inspection.issue_type == IssueType.VALID
    assert inspection.channels == 4
    assert inspection.format == "PNG"


def test_inspect_image_grayscale(grayscale_jpeg_bytes):
    """Inspection passes for grayscale JPEG; the resize step encodes single-channel L directly."""
    image_bytes = BytesIO(grayscale_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is True
    assert inspection.issue_type == IssueType.VALID
    assert inspection.channels == 1
    assert inspection.format == "JPEG"


def test_inspect_image_truncated(truncated_jpeg_bytes):
    """Inspection fails for truncated JPEG."""
    image_bytes = BytesIO(truncated_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is False
    assert inspection.issue_type in (IssueType.TRUNCATED, IssueType.DECODE_FAILURE)
    assert inspection.error_message is not None


def test_inspect_image_corrupted_header(corrupted_header_jpeg_bytes):
    """Inspection fails for JPEG with corrupted header."""
    image_bytes = BytesIO(corrupted_header_jpeg_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is False
    assert inspection.issue_type in (
        IssueType.DECODE_FAILURE,
        IssueType.CORRUPTED_HEADER,
        IssueType.UNSUPPORTED_FORMAT,
    )


def test_inspect_image_empty(empty_bytes):
    """Inspection fails for empty file."""
    image_bytes = BytesIO(empty_bytes)
    inspection = inspect_image(image_bytes)

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.ZERO_SIZE
    assert "Empty" in inspection.error_message


def test_inspect_image_none():
    """Inspection fails for None input."""
    inspection = inspect_image(None)

    assert inspection.is_valid is False
    assert inspection.issue_type == IssueType.ZERO_SIZE


def test_resize_converts_and_uploads_rgba_png(rgba_png_bytes):
    """Worker converts an RGBA PNG to RGB JPEG, uploads it, and returns a completed result."""
    mock_s3 = MagicMock()

    # Mock GET to return RGBA PNG bytes
    mock_s3.get_object.return_value = {"Body": BytesIO(rgba_png_bytes)}
    mock_s3.head_object = MagicMock()

    todo_item = {
        "source_id": 1,
        "image_id": "rgba_img",
        "original_s3_key": "coralnet-public-images/s1/images/rgba_img.png",
        "output_s3_key": "etl-outputs/coralnet/resized/s1/images/rgba_img.jpg",
    }

    result = resize_and_upload_image(
        todo_item=todo_item,
        bucket="test-bucket",
        threshold=2048,
        s3_client=mock_s3,
    )

    # Verify PUT was called with a valid RGB JPEG
    assert mock_s3.put_object.called
    uploaded = mock_s3.put_object.call_args.kwargs["Body"]
    out_img = Image.open(BytesIO(uploaded))
    out_img.load()
    assert out_img.format == "JPEG"
    assert out_img.mode == "RGB"

    # Worker returns a completed result row
    assert result["status"] == "completed"


def test_resize_mixed_batch(valid_rgb_jpeg_bytes, truncated_jpeg_bytes, tmp_path):
    """Batch with valid + corrupted images: valid resized, corrupted skipped."""
    # Create mock S3 with two images
    mock_s3 = MagicMock()

    def get_object_side_effect(Bucket, Key):
        if "valid" in Key:
            return {"Body": BytesIO(valid_rgb_jpeg_bytes)}
        if "bad" in Key:
            return {"Body": BytesIO(truncated_jpeg_bytes)}
        raise Exception("NotFound")

    mock_s3.get_object.side_effect = get_object_side_effect
    mock_s3.put_object = MagicMock()
    mock_s3.head_object = MagicMock()

    checkpoint_path = tmp_path / "checkpoint.parquet"

    todo_df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "image_id": "valid",
                "original_s3_key": "coralnet-public-images/s1/images/valid.jpg",
                "output_s3_key": "etl-outputs/coralnet/resized/s1/images/valid.jpg",
            },
            {
                "source_id": 1,
                "image_id": "bad",
                "original_s3_key": "coralnet-public-images/s1/images/bad.jpg",
                "output_s3_key": "etl-outputs/coralnet/resized/s1/images/bad.jpg",
            },
        ]
    )

    num_resized, num_skipped, num_failed, num_corrupted = resize_and_upload_all_images(
        df_todo=todo_df,
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        checkpoint_path=checkpoint_path,
        threshold=2048,
        workers=2,
        s3_client=mock_s3,
    )

    assert num_resized == 1, f"Expected 1 resized, got {num_resized}"
    assert num_corrupted == 1, f"Expected 1 corrupted, got {num_corrupted}"
    assert num_failed == 0, f"Expected 0 failed, got {num_failed}"

    # Verify checkpoint
    df_checkpoint = read_checkpoint(checkpoint_path)
    assert len(df_checkpoint) == 2

    completed = df_checkpoint[df_checkpoint["status"] == "completed"]
    skipped = df_checkpoint[df_checkpoint["status"] == "skipped"]

    assert len(completed) == 1
    assert len(skipped) == 1
    assert completed.iloc[0]["image_id"] == "valid"
    assert skipped.iloc[0]["image_id"] == "bad"
    assert "corrupted_" in str(skipped.iloc[0]["skip_reason"])


def test_resize_resumes_only_pending_items(valid_rgb_jpeg_bytes, tmp_path):
    """An existing checkpoint with completed rows is resumed: completed images are not re-
    downloaded."""
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": BytesIO(valid_rgb_jpeg_bytes)}
    mock_s3.put_object = MagicMock()
    mock_s3.head_object = MagicMock()

    checkpoint_path = tmp_path / "checkpoint.parquet"

    # Pre-existing checkpoint: 'a' already completed, 'b' still pending.
    write_checkpoint(
        checkpoint_path,
        pd.DataFrame(
            {
                "source_id": [1, 1],
                "image_id": ["a", "b"],
                "status": ["completed", "pending"],
                "resize_timestamp": [datetime.now(), None],
                "error_message": [None, None],
                "skip_reason": [None, None],
            }
        ),
    )

    todo_df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "image_id": "a",
                "original_s3_key": "coralnet-public-images/s1/images/a.jpg",
                "output_s3_key": "etl-outputs/coralnet/resized/s1/images/a.jpg",
            },
            {
                "source_id": 1,
                "image_id": "b",
                "original_s3_key": "coralnet-public-images/s1/images/b.jpg",
                "output_s3_key": "etl-outputs/coralnet/resized/s1/images/b.jpg",
            },
        ]
    )

    resize_and_upload_all_images(
        df_todo=todo_df,
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        checkpoint_path=checkpoint_path,
        threshold=2048,
        workers=2,
        s3_client=mock_s3,
    )

    # Only the pending image 'b' should have been downloaded.
    assert mock_s3.get_object.call_count == 1
    assert mock_s3.get_object.call_args.kwargs["Key"].endswith("/b.jpg")

    # Both rows end up completed in the checkpoint.
    df_final = read_checkpoint(checkpoint_path)
    assert set(df_final[df_final["status"] == "completed"]["image_id"]) == {"a", "b"}


def test_resume_summary_counts_never_negative(valid_rgb_jpeg_bytes, tmp_path):
    """Resuming with a todo smaller than the checkpoint reports counts from the checkpoint
    itself."""
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": BytesIO(valid_rgb_jpeg_bytes)}

    checkpoint_path = tmp_path / "checkpoint.parquet"

    # Old run: 'a' completed, 'c' corrupted; this run's todo only contains pending 'b'.
    write_checkpoint(
        checkpoint_path,
        pd.DataFrame(
            {
                "source_id": [1, 1, 1],
                "image_id": ["a", "b", "c"],
                "status": ["completed", "pending", "skipped"],
                "resize_timestamp": [datetime.now(), None, None],
                "error_message": [None, None, None],
                "skip_reason": [None, None, "corrupted_truncated"],
            }
        ),
    )

    todo_df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "image_id": "b",
                "original_s3_key": "coralnet-public-images/s1/images/b.jpg",
                "output_s3_key": "etl-outputs/coralnet/resized/s1/images/b.jpg",
            }
        ]
    )

    num_resized, num_skipped, num_failed, num_corrupted = resize_and_upload_all_images(
        df_todo=todo_df,
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        checkpoint_path=checkpoint_path,
        threshold=2048,
        workers=2,
        s3_client=mock_s3,
    )

    # Old formula derived num_skipped from len(df_todo) and went negative on resume.
    assert num_resized == 2  # 'a' from the old run + 'b' from this one
    assert num_skipped == 0  # no non-corrupted skips anywhere in the checkpoint
    assert num_failed == 0
    assert num_corrupted == 1  # 'c'


def test_resize_all_images_builds_one_pool_sized_client(
    monkeypatch, valid_rgb_jpeg_bytes, tmp_path
):
    """With no client injected, the orchestrator builds ONE shared client with pool >= workers."""
    from mermaidseg.datasets.coralnet.preprocessing import resize as resize_module

    created_pools: list[int] = []

    def fake_factory(*, max_pool_connections: int = 10):
        created_pools.append(max_pool_connections)
        client = MagicMock()
        client.get_object.side_effect = lambda Bucket, Key: {"Body": BytesIO(valid_rgb_jpeg_bytes)}
        return client

    monkeypatch.setattr(resize_module, "make_resize_s3_client", fake_factory)

    todo_df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "image_id": f"img_{i}",
                "original_s3_key": f"coralnet-public-images/s1/images/img_{i}.jpg",
                "output_s3_key": f"etl-outputs/coralnet/resized/s1/images/img_{i}.jpg",
            }
            for i in range(4)
        ]
    )

    resize_module.resize_and_upload_all_images(
        df_todo=todo_df,
        bucket="test-bucket",
        output_prefix="etl-outputs/coralnet",
        checkpoint_path=tmp_path / "checkpoint.parquet",
        threshold=2048,
        workers=4,
        s3_client=None,
    )

    # Exactly one client, sized for the worker count — not one default-pool client per thread.
    assert len(created_pools) == 1
    assert created_pools[0] >= 4


def test_checkpoint_tracks_skip_reason(tmp_path):
    """Checkpoint parquet correctly marks corrupted vs already-existing skips."""
    checkpoint_path = tmp_path / "checkpoint.parquet"

    # Simulate mixed statuses: completed, failed, corrupted skip, already-exists skip
    df_checkpoint = pd.DataFrame(
        {
            "source_id": [1, 1, 1, 1],
            "image_id": ["a", "b", "c", "d"],
            "status": ["completed", "failed", "skipped", "pending"],
            "resize_timestamp": [datetime.now(), None, None, None],
            "error_message": [None, "PIL error", None, None],
            "skip_reason": [None, None, "corrupted_invalid_channels", None],
        }
    )

    write_checkpoint(checkpoint_path, df_checkpoint)
    df_read = read_checkpoint(checkpoint_path)

    # Verify columns exist and values preserved
    assert "skip_reason" in df_read.columns
    corrupted_row = df_read[df_read["image_id"] == "c"]
    assert corrupted_row.iloc[0]["status"] == "skipped"
    assert "corrupted_" in str(corrupted_row.iloc[0]["skip_reason"])
