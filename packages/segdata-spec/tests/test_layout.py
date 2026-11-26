from pathlib import Path

import pytest

from segdata_spec.layout import (
    cache_path,
    feature_key,
    feature_set_prefix,
    feature_store_prefix,
    image_prefix,
)


@pytest.mark.parametrize("key", ["../x", "/x"])
def test_cache_path_refuses_a_key_that_leaves_the_cache(key):
    with pytest.raises(ValueError, match="outside the cache"):
        cache_path(Path("cache"), "bucket", key)


@pytest.mark.parametrize("bucket", ["", ".", "..", "/etc", "a/b"])
def test_cache_path_refuses_a_bucket_that_leaves_the_cache(bucket):
    with pytest.raises(ValueError, match="bucket"):
        cache_path(Path("cache"), bucket, "key")


def test_image_prefix_is_the_images_folder_then_the_dataset() -> None:
    assert image_prefix("coralnet") == "images/coralnet/"


def test_feature_layout_helpers_return_the_published_keys():
    assert (
        feature_store_prefix("coralnet", "efficientnet-b0") == "features/coralnet/efficientnet-b0/"
    )
    assert feature_key("coralnet", "efficientnet-b0", "img1") == (
        "features/coralnet/efficientnet-b0/img1.featurevector"
    )
    assert feature_key("coralnet", "efficientnet-b0", "img1", "p2") == (
        "features/coralnet/efficientnet-b0/img1~p2.featurevector"
    )
    assert feature_set_prefix("ds", "20261120-abcdef1", "efficientnet-b0") == (
        "datasets/ds/20261120-abcdef1/features/efficientnet-b0/"
    )


@pytest.mark.parametrize("bad", ["", "a/b", "..", "a..b", "Upper", "-x"])
@pytest.mark.parametrize(
    "call",
    [
        lambda bad: feature_store_prefix(bad, "fam"),
        lambda bad: feature_store_prefix("store", bad),
        lambda bad: feature_key("store", "fam", "img", bad),
        lambda bad: feature_set_prefix("ds", "20261120-abcdef1", bad),
    ],
    ids=["store", "family", "suffix", "set-family"],
)
def test_feature_layout_helpers_refuse_an_unsafe_segment(call, bad):
    with pytest.raises(ValueError, match="segment"):
        call(bad)
