"""LocalStore must behave like S3 for exactly the calls publish, sync and copy_images make."""

from __future__ import annotations

from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from segdata_etl.common.local_store import LocalStore

# MD5 of b"abc", and S3's base64 big-endian CRC64NVME of b"abc"
# (base64(awscrt.checksums.crc64nvme(b"abc").to_bytes(8, "big"))).
ABC_MD5_ETAG = '"900150983cd24fb0d6963f7d28e17f72"'
ABC_CRC64NVME = "BeXKuz/B+us="


def _code(exc: pytest.ExceptionInfo[ClientError]) -> tuple[str, int]:
    response = exc.value.response
    return response["Error"]["Code"], response["ResponseMetadata"]["HTTPStatusCode"]


def test_put_object_round_trips_at_bucket_key_path_with_md5_etag(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)

    store.put_object(Bucket="b", Key="datasets/x/m.json", Body=b"abc")

    assert (tmp_path / "b" / "datasets" / "x" / "m.json").read_bytes() == b"abc"
    assert store.get_object(Bucket="b", Key="datasets/x/m.json")["Body"].read() == b"abc"
    head = store.head_object(Bucket="b", Key="datasets/x/m.json")
    assert head["ContentLength"] == 3
    assert head["ETag"] == ABC_MD5_ETAG


def test_put_object_if_none_match_on_existing_key_is_412_and_keeps_bytes(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.put_object(Bucket="b", Key="k", Body=b"old")

    with pytest.raises(ClientError) as exc:
        store.put_object(Bucket="b", Key="k", Body=b"new", IfNoneMatch="*")

    assert _code(exc) == ("PreconditionFailed", 412)
    assert (tmp_path / "b" / "k").read_bytes() == b"old"


@pytest.mark.parametrize("op", ["head_object", "get_object"])
def test_missing_key_is_a_404_client_error(tmp_path: Path, op: str) -> None:
    store = LocalStore(tmp_path)

    with pytest.raises(ClientError) as exc:
        getattr(store, op)(Bucket="b", Key="missing")

    code, status = _code(exc)
    assert status == 404
    assert code in {"404", "NoSuchKey"}


def test_copy_object_writes_source_bytes_and_returns_s3_etag_and_crc(tmp_path: Path) -> None:
    source = LocalStore(tmp_path / "src")
    source.put_object(Bucket="sb", Key="a.jpg", Body=b"abc")
    store = LocalStore(tmp_path / "dst", source_client=source)

    result = store.copy_object(
        CopySource={"Bucket": "sb", "Key": "a.jpg"},
        Bucket="db",
        Key="out/a.jpg",
        IfNoneMatch="*",
        ChecksumAlgorithm="CRC64NVME",
        StorageClass="INTELLIGENT_TIERING",
    )

    assert (tmp_path / "dst" / "db" / "out" / "a.jpg").read_bytes() == b"abc"
    assert result == {
        "CopyObjectResult": {"ETag": ABC_MD5_ETAG, "ChecksumCRC64NVME": ABC_CRC64NVME}
    }


def test_copy_object_if_none_match_on_existing_key_is_412_and_keeps_bytes(tmp_path: Path) -> None:
    source = LocalStore(tmp_path / "src")
    source.put_object(Bucket="sb", Key="a.jpg", Body=b"abc")
    store = LocalStore(tmp_path / "dst", source_client=source)
    store.put_object(Bucket="db", Key="a.jpg", Body=b"old")

    with pytest.raises(ClientError) as exc:
        store.copy_object(
            CopySource={"Bucket": "sb", "Key": "a.jpg"}, Bucket="db", Key="a.jpg", IfNoneMatch="*"
        )

    assert _code(exc) == ("PreconditionFailed", 412)
    assert (tmp_path / "dst" / "db" / "a.jpg").read_bytes() == b"old"


def test_list_objects_v2_returns_keys_and_immediate_child_prefixes(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    for key in ["d/n/v1/a", "d/n/v1/sub/b", "d/n/v2/a", "d/other"]:
        store.put_object(Bucket="b", Key=key, Body=b"x")

    flat = store.list_objects_v2(Bucket="b", Prefix="d/n/v1/")
    delimited = store.list_objects_v2(Bucket="b", Prefix="d/", Delimiter="/")

    assert [o["Key"] for o in flat["Contents"]] == ["d/n/v1/a", "d/n/v1/sub/b"]
    assert [o["Key"] for o in delimited["Contents"]] == ["d/other"]
    assert [p["Prefix"] for p in delimited["CommonPrefixes"]] == ["d/n/"]


def test_a_real_key_ending_in_tmp_is_listed(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.put_object(Bucket="b", Key="x.tmp", Body=b"abc")

    keys = [o["Key"] for o in store.list_objects_v2(Bucket="b")["Contents"]]

    assert keys == ["x.tmp"]
