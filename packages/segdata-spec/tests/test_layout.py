from pathlib import Path

import pytest

from segdata_spec.layout import cache_path, image_prefix


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
