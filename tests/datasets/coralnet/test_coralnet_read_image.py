"""read_image key resolution: parquet-provided key vs templated original path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mermaidseg.datasets.coralnet import coralnet_dataset as cd


def _make_ds():
    """A CoralNetDataset instance without the S3-reading __init__ (just read_image
    deps)."""
    ds = object.__new__(cd.CoralNetDataset)
    ds.source_bucket = "test-bucket"
    ds.source_s3_prefix = "coralnet-public-images"
    ds.s3 = MagicMock()
    return ds


def _captured_key(**read_image_kwargs):
    ds = _make_ds()
    fake_img = MagicMock()
    fake_img.convert.return_value = [[[0, 0, 0]]]  # np.array-able (1x1x3)
    with patch.object(cd, "get_image_s3", return_value=fake_img) as get_img:
        ds.read_image(**read_image_kwargs)
    return get_img.call_args.kwargs


def test_templates_original_key_when_no_parquet_key():
    kwargs = _captured_key(image_id="1472493", source_id="1645")
    assert kwargs["key"] == "coralnet-public-images/s1645/images/1472493.jpg"
    assert kwargs["bucket"] == "test-bucket"


def test_uses_parquet_key_when_present():
    kwargs = _captured_key(
        image_id="1472493",
        source_id="1645",
        image_s3_key="dev/images/resized/s1645/images/1472493.jpg",
    )
    assert kwargs["key"] == "dev/images/resized/s1645/images/1472493.jpg"


def test_falls_back_to_template_on_null_key():
    # A parquet column present but null (pandas NaN → float) must not become the S3 key.
    kwargs = _captured_key(image_id="1472493", source_id="1645", image_s3_key=float("nan"))
    assert kwargs["key"] == "coralnet-public-images/s1645/images/1472493.jpg"
