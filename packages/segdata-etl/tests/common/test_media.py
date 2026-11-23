"""sniff_media_type names an image's media type from its leading bytes."""

from __future__ import annotations

import pytest

from segdata_etl.common.media import sniff_media_type


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image/png"),
    ],
)
def test_magic_bytes_name_the_media_type(head: bytes, expected: str) -> None:
    assert sniff_media_type(head) == expected


def test_unknown_bytes_raise() -> None:
    with pytest.raises(ValueError, match="GIF89a"):
        sniff_media_type(b"GIF89a\x01\x00")
