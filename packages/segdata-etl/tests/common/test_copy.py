"""copy_images issues conditional server-side copies and verifies a pre-existing destination."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError
from s3_fake import FakeS3

from segdata_etl.common.copy import CopiedObject, CopyItem, copy_images
from segdata_etl.common.local_store import LocalStore
from segdata_spec.errors import ObjectExists

# MD5 of b"abc", and S3's base64 big-endian CRC64NVME of b"abc".
ABC_ETAG = '"900150983cd24fb0d6963f7d28e17f72"'
ABC_CRC = "BeXKuz/B+us="
SRC = "src-bucket"
DST = "dst-bucket"


def _item(key: str = "a.jpg", dest: str = "out/a.jpg") -> CopyItem:
    return CopyItem(source_bucket=SRC, source_key=key, dest_key=dest)


def test_each_item_is_one_conditional_checksummed_copy(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"

    result = copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)

    assert fake_s3.copy_calls == [
        {
            "CopySource": {"Bucket": SRC, "Key": "a.jpg"},
            "Bucket": DST,
            "Key": "out/a.jpg",
            "IfNoneMatch": "*",
            "ChecksumAlgorithm": "CRC64NVME",
            "TaggingDirective": "REPLACE",
        }
    ]
    assert "Tagging" not in fake_s3.copy_calls[0]
    assert result == {"out/a.jpg": CopiedObject(etag=ABC_ETAG, size=3, crc64nvme=ABC_CRC)}


def test_storage_class_is_passed_to_the_copy(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"

    copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1, storage_class="INTELLIGENT_TIERING")

    assert fake_s3.copy_calls[0]["StorageClass"] == "INTELLIGENT_TIERING"


def test_existing_dest_with_the_sources_size_and_crc_is_accepted(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.objects[(DST, "out/a.jpg")] = b"abc"

    result = copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)

    assert result == {"out/a.jpg": CopiedObject(etag=ABC_ETAG, size=3, crc64nvme=ABC_CRC)}


def test_existing_dest_with_different_bytes_raises_object_exists(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.objects[(DST, "out/a.jpg")] = b"abd"

    with pytest.raises(ObjectExists, match="out/a.jpg"):
        copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)


def test_source_without_crc_falls_back_to_md5_of_both_objects(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.no_crc.add((SRC, "a.jpg"))
    fake_s3.objects[(DST, "out/a.jpg")] = b"abc"

    result = copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)

    assert result["out/a.jpg"].etag == ABC_ETAG


@pytest.mark.parametrize("dest_bytes", [b"abd", b"abcd"], ids=["same-size", "different-size"])
def test_source_without_crc_and_a_different_dest_raises_object_exists(
    fake_s3: FakeS3, dest_bytes: bytes
) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.no_crc.add((SRC, "a.jpg"))
    fake_s3.objects[(DST, "out/a.jpg")] = dest_bytes

    with pytest.raises(ObjectExists, match="out/a.jpg"):
        copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)


def test_a_non_412_client_error_is_reraised(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.copy_error = "AccessDenied"

    with pytest.raises(ClientError) as exc:
        copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)

    assert exc.value.response["Error"]["Code"] == "AccessDenied"


def test_every_item_is_copied_once_across_workers(fake_s3: FakeS3) -> None:
    items = []
    for i in range(20):
        fake_s3.objects[(SRC, f"{i}.jpg")] = f"img{i}".encode()
        items.append(_item(f"{i}.jpg", f"out/{i}.jpg"))

    result = copy_images(fake_s3, items, dest_bucket=DST, workers=4)

    assert set(result) == {f"out/{i}.jpg" for i in range(20)}
    assert sorted(c["Key"] for c in fake_s3.copy_calls) == sorted(f"out/{i}.jpg" for i in range(20))
    for i in range(20):
        expected = hashlib.md5(f"img{i}".encode(), usedforsecurity=False).hexdigest()
        assert result[f"out/{i}.jpg"].etag == f'"{expected}"'
        assert result[f"out/{i}.jpg"].size == len(f"img{i}")


def test_dest_without_crc_but_the_sources_bytes_is_accepted(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.objects[(DST, "out/a.jpg")] = b"abc"
    fake_s3.no_crc.add((DST, "out/a.jpg"))

    result = copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)

    assert result == {"out/a.jpg": CopiedObject(etag=ABC_ETAG, size=3, crc64nvme=None)}


def test_dest_without_crc_and_different_bytes_raises_object_exists(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.objects[(DST, "out/a.jpg")] = b"abd"
    fake_s3.no_crc.add((DST, "out/a.jpg"))

    with pytest.raises(ObjectExists, match="out/a.jpg"):
        copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)


def test_a_source_that_changes_during_the_copy_raises(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.source_changes_on_copy[(SRC, "a.jpg")] = b"abcdef"

    with pytest.raises(RuntimeError, match="a.jpg"):
        copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)


def test_size_comes_from_the_destination_when_the_source_has_no_crc(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "a.jpg")] = b"abc"
    fake_s3.no_crc.add((SRC, "a.jpg"))
    fake_s3.source_changes_on_copy[(SRC, "a.jpg")] = b"abcdef"

    result = copy_images(fake_s3, [_item()], dest_bucket=DST, workers=1)

    assert result["out/a.jpg"].size == 6


def test_a_localstore_destination_copies_from_a_separate_source_client(tmp_path: Path) -> None:
    src = LocalStore(tmp_path / "src")
    src.put_object(Bucket=SRC, Key="a.jpg", Body=b"abc")
    dst = LocalStore(tmp_path / "dst", source_client=src)

    result = copy_images(dst, [_item()], dest_bucket=DST, workers=1, source_client=src)

    assert result == {"out/a.jpg": CopiedObject(etag=ABC_ETAG, size=3, crc64nvme=ABC_CRC)}
    assert dst.get_object(Bucket=DST, Key="out/a.jpg")["Body"].read() == b"abc"


class _CountingItems(list[CopyItem]):
    """A list that counts how many items iteration has handed out."""

    handed_out = 0

    def __iter__(self) -> Iterator[CopyItem]:
        for item in super().__iter__():
            self.handed_out += 1
            yield item


def test_only_a_bounded_window_of_items_is_submitted_ahead(fake_s3: FakeS3) -> None:
    items = _CountingItems()
    for i in range(100):
        fake_s3.objects[(SRC, f"{i}.jpg")] = b"abc"
        items.append(_item(f"{i}.jpg", f"out/{i}.jpg"))
    handed_out_at_first_head: list[int] = []
    real_head = fake_s3.head_object

    def head(**kwargs: Any) -> dict[str, Any]:
        if not handed_out_at_first_head:
            # Gives an unbounded submitter time to hand out every item before the first finishes.
            deadline = time.monotonic() + 0.3
            while items.handed_out < len(items) and time.monotonic() < deadline:
                time.sleep(0.005)
            handed_out_at_first_head.append(items.handed_out)
        return real_head(**kwargs)

    fake_s3.head_object = head  # type: ignore[method-assign]

    copy_images(fake_s3, items, dest_bucket=DST, workers=1)

    assert handed_out_at_first_head[0] < 20


def test_the_first_failure_raises_and_leaves_later_items_uncopied(fake_s3: FakeS3) -> None:
    fake_s3.objects[(SRC, "0.jpg")] = b"abc"
    fake_s3.objects[(DST, "out/0.jpg")] = b"abd"
    items = [_item("0.jpg", "out/0.jpg")]
    for i in range(1, 100):
        fake_s3.objects[(SRC, f"{i}.jpg")] = b"abc"
        items.append(_item(f"{i}.jpg", f"out/{i}.jpg"))

    with pytest.raises(ObjectExists, match="out/0.jpg"):
        copy_images(fake_s3, items, dest_bucket=DST, workers=1)

    assert len(fake_s3.copy_calls) < 20
