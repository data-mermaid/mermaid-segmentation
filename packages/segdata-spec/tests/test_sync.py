import re
from pathlib import Path

import pytest
from conftest import (
    BUCKET,
    IMG_BUCKET,
    IMG_PREFIX,
    TAX_VERSION,
    VERSION,
    FakeS3,
    publish_taxonomy_version,
    publish_version,
)

from segdata_spec.errors import ChecksumMismatch, ImageChanged, ImageMissing, PinMismatch
from segdata_spec.manifest import Pin
from segdata_spec.sync import DatasetSyncResult, sync

PREFIX = f"datasets/ds/{VERSION}/"
TABLES = ["images.parquet", "points.parquet", "labels.parquet", "stats/label_counts.parquet"]


def _gets(client: FakeS3) -> list[tuple[str, str]]:
    return [(bucket, key) for op, bucket, key, _ in client.calls if op == "get"]


def _image_gets(client: FakeS3) -> list[str]:
    return [key for bucket, key in _gets(client) if bucket == IMG_BUCKET]


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def pin(fake_s3: FakeS3, tmp_path: Path) -> Pin:
    return publish_version(fake_s3, tmp_path)[1]


def test_sync_caches_tables_and_images_under_bucket_and_key(fake_s3, pin, cache):
    result = sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    for name in ["manifest.json", *TABLES]:
        cached = cache / BUCKET / PREFIX / name
        assert cached.read_bytes() == fake_s3.objects[(BUCKET, PREFIX + name)]
    assert (cache / IMG_BUCKET / "images/ds/a.jpg").read_bytes() == b"aaaa"
    assert (cache / IMG_BUCKET / "images/ds/b.jpg").read_bytes() == b"bbbbbbbb"
    assert (cache / IMG_BUCKET / "images/ds/c.jpg").read_bytes() == b"cc"
    assert result.datasets == {
        "ds": DatasetSyncResult(
            version=VERSION,
            manifest_sha256=pin.manifest_sha256,
            images_dropped=0,
            dropped_image_ids=(),
        )
    }
    assert result.taxonomy is None


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ({"g1"}, ["images/ds/a.jpg", "images/ds/b.jpg"]),
        (None, ["images/ds/a.jpg", "images/ds/b.jpg", "images/ds/c.jpg"]),
    ],
)
def test_sync_downloads_only_selected_groups(fake_s3, pin, cache, groups, expected):
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, selection={"ds": groups})

    assert sorted(_image_gets(fake_s3)) == expected
    assert (cache / IMG_BUCKET / "images/ds/c.jpg").exists() == (groups is None)


def test_pin_mismatch_stops_before_tables_and_images(fake_s3, pin, cache):
    wrong = Pin(name="ds", version=VERSION, manifest_sha256="0" * 64)

    with pytest.raises(PinMismatch):
        sync(fake_s3, {"ds": wrong}, cache, bucket=BUCKET)

    assert _gets(fake_s3) == [(BUCKET, PREFIX + "manifest.json")]
    assert not (cache / BUCKET / PREFIX / "manifest.json").exists()


def test_tampered_table_raises_naming_it_before_any_image(fake_s3, pin, cache):
    fake_s3.put(BUCKET, PREFIX + "labels.parquet", b"tampered")

    with pytest.raises(ChecksumMismatch, match="labels.parquet"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert _image_gets(fake_s3) == []


def test_check_split_runs_after_tables_and_before_images(fake_s3, pin, cache):
    seen = []

    def check_split(name: str, groups: set[str]) -> None:
        seen.append((name, groups, {key for _, key in _gets(fake_s3)}))

    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, check_split=check_split)

    assert seen == [
        ("ds", {"g1", "g2"}, {PREFIX + name for name in ["manifest.json", *TABLES]}),
    ]
    assert len(_image_gets(fake_s3)) == 3


def test_check_split_failure_aborts_before_images(fake_s3, pin, cache):
    def check_split(name: str, groups: set[str]) -> None:
        raise ValueError("split does not cover g2")

    with pytest.raises(ValueError, match="split does not cover g2"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, check_split=check_split)

    assert _image_gets(fake_s3) == []


@pytest.mark.parametrize("mode", ["fail", "drop"])
def test_changed_image_fails_in_both_modes(fake_s3, pin, cache, mode):
    fake_s3.put(IMG_BUCKET, IMG_PREFIX + "b.jpg", b"BBBBBBBB")

    with pytest.raises(ImageChanged, match="images/ds/b.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, on_missing_images={"ds": mode})


@pytest.mark.parametrize(
    "on_missing",
    [{"ds": "fail"}, None, {"other": "drop"}],
    ids=["fail", "default", "other-dataset-drop"],
)
def test_missing_image_fails_unless_dropped(fake_s3, pin, cache, on_missing):
    del fake_s3.objects[(IMG_BUCKET, IMG_PREFIX + "c.jpg")]

    with pytest.raises(ImageMissing, match="images/ds/c.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, on_missing_images=on_missing)


def test_missing_image_is_dropped_and_counted(fake_s3, pin, cache):
    del fake_s3.objects[(IMG_BUCKET, IMG_PREFIX + "c.jpg")]

    result = sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, on_missing_images={"ds": "drop"})

    assert result.datasets["ds"].images_dropped == 1
    assert result.datasets["ds"].dropped_image_ids == ("c",)
    assert (cache / IMG_BUCKET / "images/ds/a.jpg").read_bytes() == b"aaaa"
    assert (cache / IMG_BUCKET / "images/ds/b.jpg").read_bytes() == b"bbbbbbbb"


def test_second_sync_downloads_nothing(fake_s3, pin, cache):
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    fake_s3.calls.clear()

    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert _gets(fake_s3) == []


@pytest.mark.parametrize(
    ("relative", "content", "error"),
    [
        (f"{IMG_BUCKET}/images/ds/a.jpg", b"aaaX", ChecksumMismatch),
        (f"{IMG_BUCKET}/images/ds/a.jpg", b"aa", ChecksumMismatch),
        (f"{BUCKET}/{PREFIX}labels.parquet", b"altered", ChecksumMismatch),
        (f"{BUCKET}/{PREFIX}manifest.json", b"{}", PinMismatch),
    ],
    ids=["image-md5", "image-size", "table", "manifest"],
)
def test_corrupt_cached_file_fails_naming_it_without_redownload(
    fake_s3, pin, cache, relative, content, error
):
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    path = cache / relative
    path.write_bytes(content)
    fake_s3.calls.clear()

    with pytest.raises(error, match=re.escape(str(path))):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert path.read_bytes() == content
    assert relative.split("/", 1)[1] not in [key for _, key in _gets(fake_s3)]


@pytest.mark.parametrize(
    ("content", "error"),
    [(b"zzzz", None), (b"zz", ChecksumMismatch)],
    ids=["right-size", "wrong-size"],
)
def test_multipart_etag_checks_cached_size_only(fake_s3, tmp_path, cache, content, error):
    _, pin = publish_version(fake_s3, tmp_path, etags={"a": "abc-2"})
    cached = cache / IMG_BUCKET / "images/ds/a.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(content)

    if error is None:
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
        assert sorted(_image_gets(fake_s3)) == ["images/ds/b.jpg", "images/ds/c.jpg"]
    else:
        with pytest.raises(error, match=re.escape(str(cached))):
            sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)
    assert cached.read_bytes() == content


def test_taxonomy_pin_is_synced_under_taxonomy_prefix(fake_s3, pin, tmp_path, cache):
    tax = publish_taxonomy_version(fake_s3, tmp_path)

    result = sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, taxonomy=tax)

    prefix = f"taxonomy/mermaid-benthic/{TAX_VERSION}/"
    for name in ["manifest.json", "benthic_attributes.parquet", "growth_forms.parquet"]:
        cached = cache / BUCKET / prefix / name
        assert cached.read_bytes() == fake_s3.objects[(BUCKET, prefix + name)]
    assert result.taxonomy == tax


def test_taxonomy_pin_mismatch_fails_before_any_image(fake_s3, pin, tmp_path, cache):
    tax = publish_taxonomy_version(fake_s3, tmp_path)
    wrong = Pin(name=tax.name, version=tax.version, manifest_sha256="0" * 64)

    with pytest.raises(PinMismatch):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, taxonomy=wrong)

    assert _image_gets(fake_s3) == []


def test_short_download_body_fails_and_is_not_cached(fake_s3, pin, cache):
    fake_s3.body_overrides[(IMG_BUCKET, IMG_PREFIX + "a.jpg")] = b"aa"

    with pytest.raises(ChecksumMismatch, match="images/ds/a.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert not (cache / IMG_BUCKET / "images/ds/a.jpg").exists()


DENSE_IMAGES = {"a": (b"aaaa", "g1"), "b": (b"bbbbbbbb", "g1")}
MASKS = {"a": b"ma", "b": b"mbmb"}


def test_dense_masks_are_downloaded_with_images(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, images=DENSE_IMAGES, points=None, masks=MASKS)

    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert (cache / IMG_BUCKET / "images/ds/a.png").read_bytes() == b"ma"
    assert (cache / IMG_BUCKET / "images/ds/b.png").read_bytes() == b"mbmb"
    assert (cache / IMG_BUCKET / "images/ds/b.jpg").read_bytes() == b"bbbbbbbb"


def test_missing_mask_drops_its_image(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, images=DENSE_IMAGES, points=None, masks=MASKS)
    del fake_s3.objects[(IMG_BUCKET, IMG_PREFIX + "b.png")]

    result = sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, on_missing_images={"ds": "drop"})

    assert result.datasets["ds"].images_dropped == 1
    assert result.datasets["ds"].dropped_image_ids == ("b",)


def test_changed_mask_fails(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, images=DENSE_IMAGES, points=None, masks=MASKS)
    fake_s3.put(IMG_BUCKET, IMG_PREFIX + "b.png", b"XXXX")

    with pytest.raises(ImageChanged, match="images/ds/b.png"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)


def test_missing_mask_fails_by_default(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, images=DENSE_IMAGES, points=None, masks=MASKS)
    del fake_s3.objects[(IMG_BUCKET, IMG_PREFIX + "b.png")]

    with pytest.raises(ImageMissing, match="images/ds/b.png"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)


@pytest.mark.parametrize(
    ("selection", "named"),
    [({"other": None}, "other"), ({"ds": {"G1"}}, "G1")],
    ids=["unknown-dataset", "unknown-group"],
)
def test_selection_naming_an_unknown_dataset_or_group_fails_before_any_image(
    fake_s3, pin, cache, selection, named
):
    with pytest.raises(ValueError, match=named):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, selection=selection)

    assert _image_gets(fake_s3) == []


def test_published_key_escaping_the_cache_raises_and_writes_no_image(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3, tmp_path, images={"../evil": (b"xx", "g1")}, points=[("../evil", 1, 1, 5)]
    )

    with pytest.raises(ValueError, match="evil"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert _image_gets(fake_s3) == []
    assert not (cache / IMG_BUCKET).exists()
