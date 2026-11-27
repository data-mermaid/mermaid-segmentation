from pathlib import Path

import pytest

from segdata_spec.layout import (
    cache_path,
    feature_key,
    feature_set_prefix,
    feature_store_prefix,
)


@pytest.mark.parametrize("key", ["../x", "/x"])
def test_cache_path_refuses_a_key_that_leaves_the_cache(key):
    with pytest.raises(ValueError, match="outside the cache"):
        cache_path(Path("cache"), "bucket", key)


def test_cache_path_refuses_a_bucket_that_leaves_the_cache():
    with pytest.raises(ValueError, match="bucket"):
        cache_path(Path("cache"), "..", "key")


@pytest.mark.parametrize(
    "call",
    [
        lambda: feature_store_prefix("", "fam"),
        lambda: feature_key("store", "fam", "img", "a..b"),
        lambda: feature_set_prefix("ds", "20261120-abcdef1", "Upper"),
    ],
    ids=["empty-store", "dotdot-suffix", "uppercase-set-family"],
)
def test_feature_layout_helpers_refuse_an_unsafe_segment(call):
    with pytest.raises(ValueError, match="segment"):
        call()
