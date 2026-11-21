import re
from pathlib import Path

import pytest
from conftest import (
    BUCKET,
    IMG_BUCKET,
    VERSION,
    jpeg_bytes,
    mask_bytes,
    publish_taxonomy_version,
    publish_version,
)

from segdata_spec.errors import PinMismatch
from segdata_spec.manifest import Pin
from segdata_spec.read import open_taxonomy, open_version
from segdata_spec.resize import ResizeSpec
from segdata_spec.sync import sync
from segdata_spec.taxonomy import hierarchy_from_table

PREFIX = f"datasets/ds/{VERSION}/"


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def test_open_version_reads_synced_tables_and_resolves_image_paths(fake_s3, tmp_path, cache):
    manifest, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    version = open_version(cache, BUCKET, pin)

    assert version.manifest == manifest
    assert version.images.column("image_id").to_pylist() == ["a", "b", "c"]
    assert version.points is not None
    assert version.points.column("row").to_pylist() == [1, 2]
    assert version.labels.column("name").to_pylist() == ["x", "y"]
    assert version.label_counts.num_rows == 1
    path = version.image_path(version.images.to_pylist()[1])
    assert path == cache / IMG_BUCKET / "images/ds/b.jpg"
    assert path.read_bytes() == b"bbbbbbbb"


def test_open_version_rejects_a_pin_that_differs_from_the_cache(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    with pytest.raises(PinMismatch):
        open_version(cache, BUCKET, Pin(name="ds", version=VERSION, manifest_sha256="0" * 64))


def test_open_version_raises_naming_a_missing_table(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    missing = cache / BUCKET / PREFIX / "labels.parquet"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        open_version(cache, BUCKET, pin)


def test_open_taxonomy_feeds_the_hierarchy(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    tax = publish_taxonomy_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, taxonomy=tax)

    taxonomy = open_taxonomy(cache, BUCKET, tax)

    assert taxonomy.manifest.version == tax.version
    assert taxonomy.growth_forms.column("name").to_pylist() == ["Branching"]
    assert hierarchy_from_table(taxonomy.benthic_attributes) == {
        "hard coral": None,
        "acropora": "hard coral",
    }


def test_mask_path_resolves_a_dense_row(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3, tmp_path, images={"a": (b"aaaa", "g1")}, points=None, masks={"a": b"ma"}
    )
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    version = open_version(cache, BUCKET, pin)

    assert version.mask_path(version.images.to_pylist()[0]) == (
        cache / IMG_BUCKET / "images/ds/a.png"
    )


def test_mask_path_refuses_a_points_row(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    version = open_version(cache, BUCKET, pin)

    with pytest.raises(ValueError, match="has no mask"):
        version.mask_path(version.images.to_pylist()[0])


SPEC = ResizeSpec(max_edge=32)  # 64x48 -> 32x24


def test_open_version_with_resize_scales_sizes_points_and_paths(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images={"a": (jpeg_bytes(64, 48), "g1")},
        points=[("a", 47, 63, 5), ("a", 10, 21, 6)],
        sizes={"a": (64, 48)},
        media_types={"a": "image/png"},
    )
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, resize={"ds": SPEC})

    version = open_version(cache, BUCKET, pin, resize=SPEC)

    row = version.images.to_pylist()[0]
    assert (row["width"], row["height"], row["media_type"]) == (32, 24, "image/jpeg")
    assert version.points is not None
    assert version.points.column("row").to_pylist() == [23, 5]
    assert version.points.column("col").to_pylist() == [31, 10]
    assert version.resize == SPEC
    path = version.image_path(row)
    assert path == cache / "resized" / SPEC.id / IMG_BUCKET / "images/ds/a.jpg"
    assert path.is_file()


def test_mask_path_with_resize_resolves_into_the_spec_folder(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images={"a": (jpeg_bytes(64, 48), "g1")},
        points=None,
        masks={"a": mask_bytes(64, 48, 3, 7)},
        sizes={"a": (64, 48)},
    )
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, resize={"ds": SPEC})
    version = open_version(cache, BUCKET, pin, resize=SPEC)

    path = version.mask_path(version.images.to_pylist()[0])

    assert path == cache / "resized" / SPEC.id / IMG_BUCKET / "images/ds/a.png"
    assert path.is_file()
