import hashlib
import io
import re
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from conftest import (
    BUCKET,
    IMG_BUCKET,
    VERSION,
    jpeg_bytes,
    mask_bytes,
    publish_feature_set_version,
    publish_taxonomy_version,
    publish_version,
)

from segdata_spec.errors import NotSynced, PinMismatch, SegdataError
from segdata_spec.manifest import FileEntry, Pin, dump_manifest, manifest_sha256
from segdata_spec.read import open_feature_set, open_tables, open_taxonomy, open_version
from segdata_spec.resize import ResizeSpec
from segdata_spec.sync import sync, sync_feature_set, sync_tables
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


@pytest.mark.parametrize("resize", [None, SPEC], ids=["plain", "resized"])
def test_open_version_keeps_source_names(fake_s3, tmp_path, cache, resize):
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images={"a": (jpeg_bytes(64, 48), "g1"), "b": (jpeg_bytes(64, 48), "g1")},
        sizes={"a": (64, 48), "b": (64, 48)},
        source_names={"a": "reef-1.jpg", "b": "reef-2.jpg"},
    )
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, resize={"ds": resize})

    version = open_version(cache, BUCKET, pin, resize=resize)

    assert version.images["source_name"].to_pylist() == ["reef-1.jpg", "reef-2.jpg"]


def _publish_without_source_name(fake_s3, tmp_path) -> Pin:
    """Rewrite a published version's images table without ``source_name``, as an older ETL did.

    ``publish_dataset`` refuses such a table, so the table and manifest go straight into S3.
    """
    manifest, pin = publish_version(
        fake_s3, tmp_path, images={"a": (jpeg_bytes(64, 48), "g1")}, sizes={"a": (64, 48)}
    )
    key = (BUCKET, f"{PREFIX}images.parquet")
    table = pq.read_table(io.BytesIO(fake_s3.objects[key])).drop_columns(["source_name"])
    sink = io.BytesIO()
    pq.write_table(table, sink)
    fake_s3.put(*key, sink.getvalue())
    entry = FileEntry(
        sha256=hashlib.sha256(sink.getvalue()).hexdigest(),
        bytes=len(sink.getvalue()),
        rows=table.num_rows,
    )
    legacy = manifest.model_copy(update={"files": {**manifest.files, "images.parquet": entry}})
    manifest_bytes = dump_manifest(legacy)
    fake_s3.put(BUCKET, f"{PREFIX}manifest.json", manifest_bytes)
    return Pin(name="ds", version=VERSION, manifest_sha256=manifest_sha256(manifest_bytes))


def test_a_version_without_a_source_name_column_syncs_and_opens(fake_s3, tmp_path, cache):
    pin = _publish_without_source_name(fake_s3, tmp_path)

    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    version = open_version(cache, BUCKET, pin)

    assert "source_name" not in version.images.column_names
    assert version.images["image_id"].to_pylist() == ["a"]


def test_open_version_exposes_the_synced_groups(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, selection={"ds": {"g1"}})

    assert open_version(cache, BUCKET, pin).synced_groups == frozenset({"g1"})


def test_open_version_reports_all_groups_as_none(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert open_version(cache, BUCKET, pin).synced_groups is None


@pytest.mark.parametrize(
    ("syncs", "expected"),
    [
        ([None, set()], None),
        ([{"g1"}, set()], frozenset({"g1"})),
        ([{"g1"}, {"g2"}], frozenset({"g1", "g2"})),
        ([{"g1"}, None], None),
        ([set()], frozenset()),
    ],
    ids=[
        "all-then-tables-only",
        "subset-then-tables-only",
        "two-subsets",
        "subset-then-all",
        "tables-only",
    ],
)
def test_open_version_unions_the_groups_of_every_sync(fake_s3, tmp_path, cache, syncs, expected):
    _, pin = publish_version(fake_s3, tmp_path)
    for groups in syncs:
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, selection={"ds": groups})

    assert open_version(cache, BUCKET, pin).synced_groups == expected


def test_open_version_refuses_a_spec_that_was_not_synced(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3, tmp_path, images={"a": (jpeg_bytes(64, 48), "g1")}, sizes={"a": (64, 48)}
    )
    spec_b = ResizeSpec(max_edge=16)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, resize={"ds": SPEC})

    with pytest.raises(NotSynced) as excinfo:
        open_version(cache, BUCKET, pin, resize=spec_b)

    message = str(excinfo.value)
    assert spec_b.id in message
    assert SPEC.id in message
    assert "ds" in message


def test_open_version_refuses_a_cache_with_no_marker(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    shutil.rmtree(cache / "synced")

    with pytest.raises(NotSynced, match="original"):
        open_version(cache, BUCKET, pin)


def test_open_version_names_an_unparseable_marker_file(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    bad = cache / "synced" / "original" / BUCKET / "ds" / VERSION / "bad.json"
    bad.write_text("{not json")

    with pytest.raises(SegdataError, match=re.escape(str(bad))):
        open_version(cache, BUCKET, pin)


def test_open_tables_reads_a_tables_only_sync_that_open_version_refuses(fake_s3, tmp_path, cache):
    manifest, pin = publish_version(fake_s3, tmp_path)
    sync_tables(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    tables = open_tables(cache, BUCKET, pin)

    assert tables.manifest == manifest
    assert tables.images.column("image_id").to_pylist() == ["a", "b", "c"]
    assert tables.points is not None
    assert tables.points.column("image_id").to_pylist() == ["a", "b"]
    assert tables.labels.column("name").to_pylist() == ["x", "y"]
    assert tables.label_counts.column("n").to_pylist() == [1]
    with pytest.raises(NotSynced, match="resize ids synced for it: none"):
        open_version(cache, BUCKET, pin)


def test_open_tables_refuses_a_cache_with_no_marker(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync_tables(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    shutil.rmtree(cache / "synced")

    with pytest.raises(NotSynced):
        open_tables(cache, BUCKET, pin)


def test_open_feature_set_reads_a_synced_set(fake_s3, tmp_path, cache):
    fpin = publish_feature_set_version(fake_s3, tmp_path)
    sync_feature_set(fake_s3, fpin, cache, bucket=BUCKET)

    features = open_feature_set(cache, BUCKET, fpin)

    assert features.index.column("image_id").to_pylist() == ["a", "b"]
    assert features.path("a") == cache / BUCKET / "features/ds/effnet/a.featurevector"
    assert features.path("a").read_bytes() == b"feature-a"


def test_open_feature_set_refuses_an_unsynced_set(fake_s3, tmp_path, cache):
    fpin = publish_feature_set_version(fake_s3, tmp_path)

    with pytest.raises(NotSynced):
        open_feature_set(cache, BUCKET, fpin)


def test_open_feature_set_refuses_another_set_sha(fake_s3, tmp_path, cache):
    fpin = publish_feature_set_version(fake_s3, tmp_path)
    sync_feature_set(fake_s3, fpin, cache, bucket=BUCKET)
    wrong = fpin.model_copy(update={"manifest_sha256": "0" * 64})

    with pytest.raises(PinMismatch):
        open_feature_set(cache, BUCKET, wrong)


def test_open_tables_reads_a_version_synced_with_images(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path)
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    tables = open_tables(cache, BUCKET, pin)

    assert tables.images.column("image_id").to_pylist() == ["a", "b", "c"]


def test_open_feature_set_refuses_a_set_built_for_another_dataset_sha(fake_s3, tmp_path, cache):
    fpin = publish_feature_set_version(fake_s3, tmp_path)
    sync_feature_set(fake_s3, fpin, cache, bucket=BUCKET)
    other = fpin.model_copy(
        update={"dataset": fpin.dataset.model_copy(update={"manifest_sha256": "0" * 64})}
    )

    with pytest.raises(PinMismatch, match="0" * 64):
        open_feature_set(cache, BUCKET, other)
