from __future__ import annotations

import pytest

from segdata_etl.common.s3_objects import (
    ObjectConflict,
    SourceChanged,
    copy_new,
    image_size,
    put_new,
    sniff_media_type,
)

from .conftest import FakeS3, crc64nvme_b64, jpeg_bytes, md5_etag, png_bytes

SRC = "src-bucket"
DST = "dst-bucket"
SRC_KEY = "raw/a.jpg"
DST_KEY = "datasets/ds/v1/images/a.jpg"


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\xff\xd8\xff\xe0rest", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\nrest", "image/png"),
        (b"GIF89a\x01\x00\x01\x00", None),
        (b"", None),
    ],
)
def test_sniff_media_type(head, expected):
    assert sniff_media_type(head) == expected


def test_image_size_reads_jpeg_and_png():
    assert image_size(jpeg_bytes(37, 23)) == (37, 23)
    assert image_size(png_bytes(41, 19)) == (41, 19)


@pytest.mark.parametrize(
    "head", [b"", b"not an image at all", jpeg_bytes(37, 23)[:12], png_bytes(41, 19)[:20]]
)
def test_image_size_unreadable_is_none(head):
    assert image_size(head) is None


def _seed_source(fake: FakeS3) -> tuple[bytes, str]:
    data = jpeg_bytes(37, 23)
    fake.put(SRC, SRC_KEY, data)
    return data, md5_etag(data)


def _copy(fake: FakeS3, data: bytes, etag: str, **overrides):
    kwargs = {
        "expect_etag": md5_etag(data),
        "src_bucket": SRC,
        "src_key": SRC_KEY,
        "src_etag": etag,
        "dst_bucket": DST,
        "dst_key": DST_KEY,
        "expect_size": len(data),
        "expect_crc64nvme": crc64nvme_b64(data),
    }
    return copy_new(fake, **{**kwargs, **overrides})


def test_copy_new_absent_destination_copies_conditionally(fake_s3):
    data, etag = _seed_source(fake_s3)
    result = _copy(fake_s3, data, etag)
    (call,) = fake_s3.ops("copy")
    assert call[1:3] == (DST, DST_KEY)
    assert call[3]["IfNoneMatch"] == "*"
    assert call[3]["CopySourceIfMatch"] == etag
    assert call[3]["ChecksumAlgorithm"] == "CRC64NVME"
    assert call[3]["CopySource"] == {"Bucket": SRC, "Key": SRC_KEY}
    assert result.etag == etag
    assert result.size == len(data)
    assert result.crc64nvme == crc64nvme_b64(data)
    assert result.already_present is False
    assert fake_s3.objects[(DST, DST_KEY)] == data


def test_copy_new_existing_identical_destination_is_accepted(fake_s3):
    data, etag = _seed_source(fake_s3)
    fake_s3.put(DST, DST_KEY, data)
    result = _copy(fake_s3, data, etag)
    assert result.already_present is True
    assert result.etag == etag
    assert result.size == len(data)
    assert result.crc64nvme == crc64nvme_b64(data)
    (head,) = fake_s3.ops("head")
    assert head[3]["ChecksumMode"] == "ENABLED"


def test_copy_new_existing_destination_with_other_crc_conflicts(fake_s3):
    data, etag = _seed_source(fake_s3)
    other = bytes(reversed(data))
    fake_s3.put(DST, DST_KEY, other)
    with pytest.raises(ObjectConflict):
        _copy(fake_s3, data, etag)
    assert fake_s3.objects[(DST, DST_KEY)] == other


def test_copy_new_existing_destination_with_other_size_conflicts(fake_s3):
    data, etag = _seed_source(fake_s3)
    fake_s3.put(DST, DST_KEY, data + b"x")
    with pytest.raises(ObjectConflict):
        _copy(fake_s3, data, etag)


def test_copy_new_changed_source_etag_is_distinct_error(fake_s3):
    data, _ = _seed_source(fake_s3)
    with pytest.raises(SourceChanged) as info:
        _copy(fake_s3, data, '"deadbeef"')
    assert not isinstance(info.value, ObjectConflict)
    assert (DST, DST_KEY) not in fake_s3.objects


def test_copy_new_multipart_source_lands_with_md5_etag(fake_s3):
    data = jpeg_bytes(37, 23)
    fake_s3.put(SRC, SRC_KEY, data, etag='"abc123-2"')
    result = _copy(fake_s3, data, '"abc123-2"')
    assert result.etag == md5_etag(data)


def test_copy_new_destination_etag_differing_from_expected_conflicts(fake_s3):
    data, etag = _seed_source(fake_s3)
    with pytest.raises(ObjectConflict):
        _copy(fake_s3, data, etag, expect_etag='"abc123-2"')


def test_copy_new_existing_destination_with_other_etag_conflicts(fake_s3):
    data, etag = _seed_source(fake_s3)
    fake_s3.put(DST, DST_KEY, data, etag='"abc123-2"')
    with pytest.raises(ObjectConflict):
        _copy(fake_s3, data, etag)


def test_copy_new_response_without_crc_raises(fake_s3):
    data, etag = _seed_source(fake_s3)
    real_copy = fake_s3.copy_object

    def copy_without_crc(**kwargs):
        response = real_copy(**kwargs)
        del response["CopyObjectResult"]["ChecksumCRC64NVME"]
        return response

    fake_s3.copy_object = copy_without_crc
    with pytest.raises(ObjectConflict):
        _copy(fake_s3, data, etag)


def test_copy_new_existing_destination_without_crc_raises(fake_s3):
    data, etag = _seed_source(fake_s3)
    fake_s3.put(DST, DST_KEY, data, crc=False)
    with pytest.raises(ObjectConflict):
        _copy(fake_s3, data, etag)


def test_put_new_absent_key_writes_conditionally(fake_s3):
    result = put_new(fake_s3, bucket=DST, key="k.json", body=b"{}", content_type="application/json")
    (call,) = fake_s3.ops("put")
    assert call[3]["IfNoneMatch"] == "*"
    assert call[3]["ChecksumAlgorithm"] == "CRC64NVME"
    assert call[3]["ContentType"] == "application/json"
    assert fake_s3.objects[(DST, "k.json")] == b"{}"
    assert result.etag == md5_etag(b"{}")
    assert result.size == 2
    assert result.crc64nvme == crc64nvme_b64(b"{}")
    assert result.already_present is False


def test_put_new_identical_existing_bytes_is_accepted(fake_s3):
    fake_s3.put(DST, "k.json", b"{}")
    result = put_new(fake_s3, bucket=DST, key="k.json", body=b"{}", content_type="application/json")
    assert result.already_present is True
    assert result.etag == '"99914b932bd37a50b983c5e7c90ae93b"'
    assert result.crc64nvme == "Sv1wpwepjTw="


def test_put_new_response_without_crc_raises(fake_s3):
    real_put = fake_s3.put_object

    def put_without_crc(**kwargs):
        response = real_put(**kwargs)
        del response["ChecksumCRC64NVME"]
        return response

    fake_s3.put_object = put_without_crc
    with pytest.raises(ObjectConflict):
        put_new(fake_s3, bucket=DST, key="k.json", body=b"{}", content_type="application/json")


def test_put_new_identical_existing_bytes_without_crc_raises(fake_s3):
    fake_s3.put(DST, "k.json", b"{}", crc=False)
    with pytest.raises(ObjectConflict):
        put_new(fake_s3, bucket=DST, key="k.json", body=b"{}", content_type="application/json")


def test_put_new_different_existing_bytes_conflicts(fake_s3):
    fake_s3.put(DST, "k.json", b"[]")
    with pytest.raises(ObjectConflict):
        put_new(fake_s3, bucket=DST, key="k.json", body=b"{}", content_type="application/json")
    assert fake_s3.objects[(DST, "k.json")] == b"[]"
