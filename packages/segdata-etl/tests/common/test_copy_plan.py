from __future__ import annotations

from typing import Any

from s3_fake import FakeS3

from segdata_etl.common.copy_plan import probe_image, read_image_header

_HEAD = {"ETag": '"e1"', "ContentLength": 10, "ChecksumCRC64NVME": "crc"}


def _probe(parent_fingerprint: str, client: Any) -> Any:
    def build_row(*_: Any) -> dict[str, Any]:
        raise AssertionError("an unchanged image must not be probed")

    parent = {"i1": {"image_id": "i1", "source_fingerprint": parent_fingerprint}}
    return probe_image(client, "src", "i1", "raw/i1.jpg", _HEAD, parent, build_row)


def test_probe_image_reuses_a_parent_row_whose_fingerprint_matches_the_bare_etag():
    client = FakeS3()

    outcome = _probe("e1", client)

    assert outcome.reused is True
    assert client.calls == []


def _header_of(client: FakeS3, data: bytes) -> Any:
    client.put("src", "raw/i1", data)
    return read_image_header(client, "src", "raw/i1", client.etag("src", "raw/i1"), len(data))


def test_read_image_header_returns_none_for_a_jpeg_without_a_sof_after_both_reads():
    client = FakeS3()
    com = b"\xff\xfe" + (65_000).to_bytes(2, "big") + b"\x00" * (65_000 - 2)
    data = b"\xff\xd8" + com + com + b"\xff\xd9"

    assert _header_of(client, data) is None
    assert [call[3]["Range"] for call in client.ops("get")] == ["bytes=0-65535", "bytes=0-1048575"]
