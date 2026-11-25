import hashlib
import io
import json
import os
import re
import stat
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError
from conftest import (
    BUCKET,
    IMG_BUCKET,
    IMG_PREFIX,
    TAX_VERSION,
    VERSION,
    FakeS3,
    jpeg_bytes,
    mask_bytes,
    publish_taxonomy_version,
    publish_version,
)
from PIL import Image

from segdata_spec.errors import (
    ChecksumMismatch,
    ImageChanged,
    ImageMissing,
    ImageUndecodable,
    ManifestNotFound,
    PinMismatch,
)
from segdata_spec.manifest import Pin
from segdata_spec.resize import ResizeSpec
from segdata_spec.sync import (
    _IN_FLIGHT_PER_WORKER,
    DatasetSyncResult,
    check_published_manifest,
    sync,
)

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
def umask_027() -> Iterator[None]:
    previous = os.umask(0o027)
    try:
        yield
    finally:
        os.umask(previous)


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
        )
    }
    assert result.taxonomy is None


def test_sync_writes_cached_files_with_umask_applied_mode(fake_s3, pin, cache, umask_027):
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    image = cache / IMG_BUCKET / "images/ds/a.jpg"
    table = cache / BUCKET / PREFIX / "images.parquet"
    assert stat.S_IMODE(image.stat().st_mode) == 0o640
    assert stat.S_IMODE(table.stat().st_mode) == 0o640


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


@pytest.mark.parametrize("max_workers", [1, 4])
def test_check_split_failure_aborts_before_images(fake_s3, pin, cache, max_workers):
    def check_split(name: str, groups: set[str]) -> None:
        raise ValueError("split does not cover g2")

    with pytest.raises(ValueError, match="split does not cover g2"):
        sync(
            fake_s3,
            {"ds": pin},
            cache,
            bucket=BUCKET,
            check_split=check_split,
            max_workers=max_workers,
        )

    assert _image_gets(fake_s3) == []


def test_changed_image_fails(fake_s3, pin, cache):
    fake_s3.put(IMG_BUCKET, IMG_PREFIX + "b.jpg", b"BBBBBBBB")

    with pytest.raises(ImageChanged, match="images/ds/b.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)


def test_missing_image_fails(fake_s3, pin, cache):
    del fake_s3.objects[(IMG_BUCKET, IMG_PREFIX + "c.jpg")]

    with pytest.raises(ImageMissing, match="images/ds/c.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)


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


DENSE_IMAGES = {"a": (b"aaaa", "g1"), "b": (b"bbbbbbbb", "g1")}
MASKS = {"a": b"ma", "b": b"mbmb"}


HELLO_CRC = "jSnVw/bqjr4="  # crc64nvme(b"hello world"), base64 of the big-endian 8 bytes


def test_cached_multipart_image_with_wrong_crc_fails_without_download(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, etags={"a": "abc-2"}, crcs={"a": "AAAAAAAAAAA="})
    cached = cache / IMG_BUCKET / "images/ds/a.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"zzzz")

    with pytest.raises(ChecksumMismatch, match=re.escape(str(cached))):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert "images/ds/a.jpg" not in _image_gets(fake_s3)


def test_cached_multipart_image_with_matching_crc_syncs_without_download(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images={"a": (b"hello world", "g1")},
        points=[("a", 1, 1, 5)],
        etags={"a": "abc-2"},
        crcs={"a": HELLO_CRC},
    )
    cached = cache / IMG_BUCKET / "images/ds/a.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"hello world")

    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert _image_gets(fake_s3) == []


def test_downloaded_image_with_wrong_crc_fails_and_is_not_cached(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, crcs={"a": "AAAAAAAAAAA="})

    with pytest.raises(ChecksumMismatch, match="images/ds/a.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert not (cache / IMG_BUCKET / "images/ds/a.jpg").exists()


def test_cached_mask_with_wrong_crc_fails(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images=DENSE_IMAGES,
        points=None,
        masks=MASKS,
        mask_crcs={"a": "AAAAAAAAAAA="},
    )
    cached = cache / IMG_BUCKET / "images/ds/a.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"ma")

    with pytest.raises(ChecksumMismatch, match=re.escape(str(cached))):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)


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


def test_dense_masks_are_downloaded_with_images(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, images=DENSE_IMAGES, points=None, masks=MASKS)

    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert (cache / IMG_BUCKET / "images/ds/a.png").read_bytes() == b"ma"
    assert (cache / IMG_BUCKET / "images/ds/b.png").read_bytes() == b"mbmb"
    assert (cache / IMG_BUCKET / "images/ds/b.jpg").read_bytes() == b"bbbbbbbb"


def test_changed_mask_fails(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, images=DENSE_IMAGES, points=None, masks=MASKS)
    fake_s3.put(IMG_BUCKET, IMG_PREFIX + "b.png", b"XXXX")

    with pytest.raises(ImageChanged, match="images/ds/b.png"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)


def test_missing_mask_fails(fake_s3, tmp_path, cache):
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


SPEC = ResizeSpec(max_edge=32)  # 64x48 -> 32x24
SMALL_SPEC = ResizeSpec(max_edge=16)  # 64x48 -> 16x12
JPEG_A = jpeg_bytes(64, 48)
RESIZE_IMAGES = {"a": (JPEG_A, "g1")}
RESIZE_SIZES = {"a": (64, 48)}


def _resized(cache: Path, spec: ResizeSpec, key: str = "a.jpg") -> Path:
    return cache / "resized" / spec.id / IMG_BUCKET / IMG_PREFIX / key


def _sidecar(cache: Path, spec: ResizeSpec, key: str = "a.jpg") -> Path:
    return cache / "resized" / spec.id / ".index" / IMG_BUCKET / IMG_PREFIX / f"{key}.json"


def _decoded(path: Path) -> Image.Image:
    img = Image.open(io.BytesIO(path.read_bytes()))
    img.load()
    return img


@pytest.fixture
def resize_pin(fake_s3: FakeS3, tmp_path: Path) -> Pin:
    return publish_version(
        fake_s3, tmp_path, images=RESIZE_IMAGES, points=[("a", 1, 1, 5)], sizes=RESIZE_SIZES
    )[1]


def _sync_resized(fake_s3, pin, cache, spec=SPEC):
    return sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, resize={"ds": spec})


def test_resized_sync_writes_output_and_sidecar_but_not_the_original(fake_s3, resize_pin, cache):
    _sync_resized(fake_s3, resize_pin, cache)

    output = _resized(cache, SPEC)
    assert (_decoded(output).format, _decoded(output).size) == ("JPEG", (32, 24))
    data = output.read_bytes()
    assert json.loads(_sidecar(cache, SPEC).read_text()) == {
        "source_etag": hashlib.md5(JPEG_A).hexdigest(),  # noqa: S324
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": 32,
        "height": 24,
    }
    assert not (cache / IMG_BUCKET).exists()


def _rewrite_sidecar(cache: Path, **changes: Any) -> None:
    path = _sidecar(cache, SPEC)
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def _drop_from_sidecar(cache: Path, name: str) -> None:
    path = _sidecar(cache, SPEC)
    entry = json.loads(path.read_text())
    del entry[name]
    path.write_text(json.dumps(entry))


@pytest.mark.parametrize(
    "tamper",
    [
        lambda cache: _rewrite_sidecar(cache, size=_resized(cache, SPEC).stat().st_size + 1),
        lambda cache: _resized(cache, SPEC).write_bytes(
            b"x" * _resized(cache, SPEC).stat().st_size
        ),
        lambda cache: _rewrite_sidecar(cache, width=31),
        lambda cache: _rewrite_sidecar(cache, height=25),
        lambda cache: _drop_from_sidecar(cache, "width"),
        lambda cache: _drop_from_sidecar(cache, "height"),
        lambda cache: _rewrite_sidecar(cache, source_etag="0" * 32),
        lambda cache: _sidecar(cache, SPEC).unlink(),
        lambda cache: _sidecar(cache, SPEC).write_text("{trunc"),
        lambda cache: _sidecar(cache, SPEC).write_text("[]"),
    ],
    ids=[
        "indexed-size-only",
        "output-bytes-same-length",
        "indexed-width",
        "indexed-height",
        "missing-width",
        "missing-height",
        "stale-source-etag",
        "missing-sidecar",
        "malformed-sidecar",
        "non-object-sidecar",
    ],
)
def test_unverifiable_resized_output_fails_without_download(fake_s3, resize_pin, cache, tamper):
    _sync_resized(fake_s3, resize_pin, cache)
    tamper(cache)
    fake_s3.calls.clear()

    with pytest.raises(ChecksumMismatch, match=re.escape(str(_resized(cache, SPEC)))):
        _sync_resized(fake_s3, resize_pin, cache)

    assert _image_gets(fake_s3) == []


def test_cached_output_fails_when_a_later_version_records_another_resized_size(
    fake_s3, resize_pin, cache, tmp_path
):
    _sync_resized(fake_s3, resize_pin, cache)
    # Same image bytes and ETag, but the row now says 96x48, which scales to 32x16, not 32x24.
    _, later = publish_version(
        fake_s3,
        tmp_path,
        version="20261121-abcdef2",
        images=RESIZE_IMAGES,
        points=[("a", 1, 1, 5)],
        sizes={"a": (96, 48)},
    )
    fake_s3.calls.clear()

    with pytest.raises(ChecksumMismatch, match=re.escape(str(_resized(cache, SPEC)))):
        _sync_resized(fake_s3, later, cache)

    assert _image_gets(fake_s3) == []


def test_original_whose_row_size_disagrees_with_its_header_fails_and_caches_nothing(
    fake_s3, tmp_path, cache
):
    _, pin = publish_version(
        fake_s3, tmp_path, images=RESIZE_IMAGES, points=[("a", 1, 1, 5)], sizes={"a": (96, 48)}
    )

    with pytest.raises(ChecksumMismatch, match=r"images/ds/a\.jpg.*32x24.*expected 32x16"):
        _sync_resized(fake_s3, pin, cache)

    assert not (cache / "resized").exists()


def test_undecodable_original_raises_naming_it_and_caches_nothing(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images={"a": (b"not an image", "g1")},
        points=[("a", 1, 1, 5)],
        sizes=RESIZE_SIZES,
    )

    with pytest.raises(ImageUndecodable, match=f"s3://{IMG_BUCKET}/{IMG_PREFIX}a.jpg"):
        _sync_resized(fake_s3, pin, cache)

    assert not (cache / "resized").exists()


def test_decompression_bomb_original_propagates_unwrapped(fake_s3, resize_pin, cache, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(Image.DecompressionBombError):
        _sync_resized(fake_s3, resize_pin, cache)

    assert not (cache / "resized").exists()


def test_failed_sidecar_write_leaves_no_output(fake_s3, resize_pin, cache, monkeypatch):
    real_replace = os.replace
    output = _resized(cache, SPEC)
    sidecar = _sidecar(cache, SPEC)

    def replace(src: Any, dst: Any) -> None:
        if Path(dst) == sidecar:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError, match="disk full"):
        _sync_resized(fake_s3, resize_pin, cache)
    assert not output.exists()
    assert not list(sidecar.parent.glob("*.tmp"))


def test_sidecar_without_output_is_redone(fake_s3, resize_pin, cache):
    _sync_resized(fake_s3, resize_pin, cache)
    _resized(cache, SPEC).unlink()
    _sidecar(cache, SPEC).write_text("{}")
    fake_s3.calls.clear()

    _sync_resized(fake_s3, resize_pin, cache)

    data = _resized(cache, SPEC).read_bytes()
    assert _decoded(_resized(cache, SPEC)).size == (32, 24)
    assert json.loads(_sidecar(cache, SPEC).read_text())["sha256"] == (
        hashlib.sha256(data).hexdigest()
    )
    assert _image_gets(fake_s3) == [IMG_PREFIX + "a.jpg"]


def test_two_specs_share_one_cache_without_interfering(fake_s3, resize_pin, cache):
    _sync_resized(fake_s3, resize_pin, cache, SPEC)
    _sync_resized(fake_s3, resize_pin, cache, SMALL_SPEC)
    fake_s3.calls.clear()

    _sync_resized(fake_s3, resize_pin, cache, SPEC)
    _sync_resized(fake_s3, resize_pin, cache, SMALL_SPEC)

    assert _decoded(_resized(cache, SPEC)).size == (32, 24)
    assert _decoded(_resized(cache, SMALL_SPEC)).size == (16, 12)
    assert _image_gets(fake_s3) == []


def test_resize_naming_an_unpinned_dataset_fails_before_any_image(fake_s3, resize_pin, cache):
    with pytest.raises(ValueError, match="other"):
        sync(fake_s3, {"ds": resize_pin}, cache, bucket=BUCKET, resize={"other": SPEC})

    assert _image_gets(fake_s3) == []


def test_dense_mask_is_resized_to_the_image_scaled_size(fake_s3, tmp_path, cache):
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images=RESIZE_IMAGES,
        points=None,
        masks={"a": mask_bytes(64, 48, 3, 7)},
        sizes=RESIZE_SIZES,
    )

    _sync_resized(fake_s3, pin, cache)

    mask = _decoded(_resized(cache, SPEC, "a.png"))
    assert (mask.format, mask.size) == ("PNG", (32, 24))
    assert (mask.getpixel((0, 0)), mask.getpixel((31, 23))) == (3, 7)
    assert set(mask.tobytes()) == {3, 7}
    assert _sidecar(cache, SPEC, "a.png").exists()


def test_resize_none_for_a_dataset_caches_originals(fake_s3, resize_pin, cache):
    sync(fake_s3, {"ds": resize_pin}, cache, bucket=BUCKET, resize={"ds": None})

    assert (cache / IMG_BUCKET / IMG_PREFIX / "a.jpg").read_bytes() == JPEG_A
    assert not (cache / "resized").exists()


def test_sync_result_records_the_spec_applied_per_dataset(fake_s3, resize_pin, cache):
    pins = {"ds": resize_pin, "plain": resize_pin}

    result = sync(fake_s3, pins, cache, bucket=BUCKET, resize={"ds": SPEC})

    assert result.datasets["ds"].resize == SPEC
    assert result.datasets["plain"].resize is None


def test_max_workers_below_one_fails_before_any_s3_call(fake_s3, resize_pin, cache):
    with pytest.raises(ValueError, match="max_workers"):
        sync(fake_s3, {"ds": resize_pin}, cache, bucket=BUCKET, max_workers=0)

    assert fake_s3.calls == []


def test_resized_original_failing_verification_writes_nothing(fake_s3, resize_pin, cache):
    fake_s3.body_overrides[(IMG_BUCKET, IMG_PREFIX + "a.jpg")] = JPEG_A[:-4]

    with pytest.raises(ChecksumMismatch, match="images/ds/a.jpg"):
        _sync_resized(fake_s3, resize_pin, cache)

    assert not (cache / "resized").exists()


POOL_IMAGES = {i: (i.encode() * 4, "g1") for i in "abcde"}


def _hold_first_until_second_returns(fake_s3: FakeS3, first: str, second: str) -> None:
    """Finish the GET of ``first`` only after the GET of ``second`` returns or raises, so a pool
    that consumed results in completion order would see ``second`` first."""
    released = threading.Event()
    real_get = fake_s3.get_object

    def get_object(*, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        if Key == IMG_PREFIX + first:
            assert released.wait(timeout=5), f"{second} was never fetched concurrently"
            return real_get(Bucket=Bucket, Key=Key, **kwargs)
        try:
            return real_get(Bucket=Bucket, Key=Key, **kwargs)
        finally:
            if Key == IMG_PREFIX + second:
                released.set()

    fake_s3.get_object = get_object  # type: ignore[method-assign]


def test_earliest_failing_row_raises_whatever_the_completion_order(fake_s3, tmp_path, cache):
    _, pin = publish_version(fake_s3, tmp_path, images=POOL_IMAGES, points=[("a", 1, 1, 5)])
    fake_s3.put(IMG_BUCKET, IMG_PREFIX + "b.jpg", b"BBBB")
    del fake_s3.objects[(IMG_BUCKET, IMG_PREFIX + "d.jpg")]
    _hold_first_until_second_returns(fake_s3, "b.jpg", "d.jpg")

    with pytest.raises(ImageChanged, match="images/ds/b.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, max_workers=4)


def test_failure_on_the_first_row_bounds_the_rows_submitted_and_cancels_the_rest(
    fake_s3, tmp_path, cache, monkeypatch
):
    ids = [f"i{n:02d}" for n in range(20)]
    images = {i: (i.encode() * 4, "g1") for i in ids}
    _, pin = publish_version(fake_s3, tmp_path, images=images, points=[("i00", 1, 1, 5)])
    fake_s3.put(IMG_BUCKET, IMG_PREFIX + "i00.jpg", b"CHANGED!")
    futs: list[Any] = []
    real_get = fake_s3.get_object

    class CountingPool(ThreadPoolExecutor):
        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            futs.append(super().submit(fn, *args, **kwargs))
            return futs[-1]

    def get_object(*, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        # Every image but row 0 holds its worker until the queued rows are cancelled.
        if Key.startswith(IMG_PREFIX) and Key != IMG_PREFIX + "i00.jpg":
            deadline = time.monotonic() + 10
            while not futs[-1].cancelled() and time.monotonic() < deadline:
                time.sleep(0.005)
        return real_get(Bucket=Bucket, Key=Key, **kwargs)

    fake_s3.get_object = get_object  # type: ignore[method-assign]
    monkeypatch.setattr("segdata_spec.sync.ThreadPoolExecutor", CountingPool)

    with pytest.raises(ImageChanged, match="i00.jpg"):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, max_workers=2)

    assert len(futs) == 2 * _IN_FLIGHT_PER_WORKER
    assert len(_image_gets(fake_s3)) <= 3  # max_workers + 1: rows 0, 1 and at most one more


def _marker_files(cache: Path, resize_id: str, name: str = "ds") -> list[Path]:
    return sorted((cache / "synced" / resize_id / BUCKET / name / VERSION).glob("*.json"))


def _markers(cache: Path, resize_id: str, name: str = "ds") -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in _marker_files(cache, resize_id, name)]


def test_sync_with_a_spec_writes_one_marker_file_under_the_spec_id(fake_s3, tmp_path, cache):
    spec = ResizeSpec(max_edge=32)
    jpeg = jpeg_bytes(64, 48)
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images={"a": (jpeg, "g1"), "b": (jpeg, "g1"), "c": (jpeg, "g2")},
        sizes={"a": (64, 48), "b": (64, 48), "c": (64, 48)},
    )

    sync(
        fake_s3,
        {"ds": pin},
        cache,
        bucket=BUCKET,
        selection={"ds": {"g1", "g2"}},
        resize={"ds": spec},
    )

    assert _markers(cache, spec.id) == [
        {
            "resize": {"max_edge": 32, "filter": "lanczos", "jpeg_quality": 95},
            "groups": ["g1", "g2"],
        }
    ]


def test_sync_without_a_spec_writes_one_marker_file_under_original(fake_s3, pin, cache):
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, selection={"ds": {"g2"}})

    assert _markers(cache, "original") == [{"resize": None, "groups": ["g2"]}]


def test_sync_without_a_selection_records_all_groups(fake_s3, pin, cache):
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert _markers(cache, "original")[0]["groups"] == "all"


def test_each_sync_adds_a_marker_file_and_keeps_the_earlier_ones(fake_s3, pin, cache):
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, selection={"ds": {"g1"}})
    first = _marker_files(cache, "original")
    sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET, selection={"ds": {"g2"}})

    files = _marker_files(cache, "original")
    assert len(files) == 2
    assert set(first) < set(files)
    assert sorted(m["groups"] for m in _markers(cache, "original")) == [["g1"], ["g2"]]


def test_a_failed_sync_leaves_no_marker(fake_s3, pin, cache):
    del fake_s3.objects[(IMG_BUCKET, IMG_PREFIX + "c.jpg")]

    with pytest.raises(ImageMissing):
        sync(fake_s3, {"ds": pin}, cache, bucket=BUCKET)

    assert not (cache / "synced").exists()


def test_check_published_manifest_returns_none_for_a_matching_sha(fake_s3, pin):
    assert check_published_manifest(fake_s3, BUCKET, pin) is None


def test_check_published_manifest_names_the_key_of_a_missing_manifest(fake_s3, pin):
    del fake_s3.objects[(BUCKET, PREFIX + "manifest.json")]

    with pytest.raises(ManifestNotFound, match=re.escape(f"s3://{BUCKET}/{PREFIX}manifest.json")):
        check_published_manifest(fake_s3, BUCKET, pin)


def test_check_published_manifest_gives_both_shas_on_mismatch(fake_s3, pin):
    fake_s3.objects[(BUCKET, PREFIX + "manifest.json")] = b"tampered"
    actual = hashlib.sha256(b"tampered").hexdigest()

    with pytest.raises(PinMismatch) as raised:
        check_published_manifest(fake_s3, BUCKET, pin)

    assert actual in str(raised.value)
    assert pin.manifest_sha256 in str(raised.value)


def test_check_published_manifest_propagates_a_non_404_client_error(fake_s3, pin):
    denied = ClientError(
        {
            "Error": {"Code": "AccessDenied", "Message": "no"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "GetObject",
    )

    def get_object(**kwargs: Any) -> Any:
        raise denied

    fake_s3.get_object = get_object  # type: ignore[method-assign]

    with pytest.raises(ClientError) as raised:
        check_published_manifest(fake_s3, BUCKET, pin)

    assert raised.value is denied


def test_check_published_manifest_taxonomy_flag_selects_the_taxonomy_prefix(fake_s3, tmp_path):
    tax = publish_taxonomy_version(fake_s3, tmp_path)

    assert check_published_manifest(fake_s3, BUCKET, tax, taxonomy=True) is None
    with pytest.raises(ManifestNotFound):
        check_published_manifest(fake_s3, BUCKET, tax, taxonomy=False)
