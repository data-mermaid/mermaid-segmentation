from __future__ import annotations

from typing import Any

from s3_fake import FakeS3

from segdata_etl.common.copy_plan import probe_image

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
