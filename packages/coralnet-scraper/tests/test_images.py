"""Offline tests for ImageFetcher and the image validators."""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest
import requests
from conftest import (
    _JPEG,
    _JPEG_HEADERS,
    Reply,
    ScriptedServer,
    _http_response,
    _refs,
    _site_downloader,
    _view_page,
)
from PIL import Image

from coralnet_scraper import images as images_mod
from coralnet_scraper.downloader import CoralNetDownloader
from coralnet_scraper.models import ImageFailures
from coralnet_scraper.walker import ImageRef


def test_download_images_separates_missing_url_from_upload_failures():
    dl = CoralNetDownloader("u", "p", max_workers=2)
    # Image 1 uploads, 2 has no original on its view page, 3 fails to upload.

    def resolve(rel: str) -> str:
        if rel == "/image/2/view/":
            raise ValueError("private")
        return "http://x" + rel

    def upload(url: str, bucket: str, key: str) -> int:
        if url.endswith("/3/view/"):
            raise RuntimeError("boom")
        return 1

    with (
        patch.object(dl.images, "_resolve_one_image_url", side_effect=resolve),
        patch.object(dl.images, "download_image_to_s3", side_effect=upload),
    ):
        result = dl.images.download_images(_refs({1: [1, 2, 3]}), 1, "b", "p")
    assert result == ImageFailures(missing_url=1, upload_failed=1)


def _outage_run(
    failing: Callable[[int], bool], count: int, *, await_stop: bool = False
) -> tuple[ImageFailures, list[str]]:
    """Run ``count`` single-worker image downloads whose view page fails when ``failing(id)``.

    ``await_stop`` makes image 21 wait for the collecting thread to set ``stop`` first.
    """
    dl = CoralNetDownloader("u", "p", max_workers=1)
    resolved: list[str] = []

    def resolve(rel: str) -> str:
        resolved.append(rel)
        if failing(int(rel.split("/")[2])):
            raise requests.HTTPError("503")
        return "http://x" + rel

    real = dl.images._download_image

    def paced(ref: ImageRef, *args: Any) -> str:
        if await_stop and ref.id == images_mod._IMAGE_OUTAGE_STREAK + 1:
            assert args[-1].wait(5)  # stop is the last positional argument
        return real(ref, *args)

    with (
        patch.object(dl.images, "_resolve_one_image_url", side_effect=resolve),
        patch.object(dl.images, "download_image_to_s3", return_value=1),
        patch.object(dl.images, "_download_image", side_effect=paced),
    ):
        failures = dl.images.download_images(_refs({1: list(range(1, count + 1))}), 1, "b", "p")
    return failures, resolved


def test_a_run_of_failed_images_stops_the_stage_and_skips_the_rest():
    failures, resolved = _outage_run(lambda _i: True, 25, await_stop=True)

    assert failures == ImageFailures(missing_url=20, skipped=5)
    assert len(resolved) < 25 * images_mod._IMAGE_ATTEMPTS


def test_a_success_resets_the_image_failure_streak():
    # Images 1-19 fail, 20 succeeds, 21-39 fail: 19 consecutive failures twice.
    failures, _ = _outage_run(lambda i: i != 20, 39)

    assert failures.skipped == 0


_ORIGINAL_IMAGE_PAGE = (
    '<html><body><div id="original_image_container">'
    '<img src="https://bucket.s3.amazonaws.com/original/abc.jpg"/></div></body></html>'
)


def test_resolve_one_image_url_returns_the_original_image_src():
    dl = CoralNetDownloader("u", "p")

    with patch.object(
        requests.Session, "get", return_value=_http_response(_ORIGINAL_IMAGE_PAGE)
    ) as get:
        url = dl.images._resolve_one_image_url("/image/5/view/")

    assert url == "https://bucket.s3.amazonaws.com/original/abc.jpg"
    assert get.call_args.args == ("https://coralnet.ucsd.edu/image/5/view/",)


def test_resolve_one_image_url_raises_when_the_page_has_no_original_image():
    dl = CoralNetDownloader("u", "p")
    private_page = "<html><body>You don't have permission to view this image.</body></html>"

    with (
        patch.object(requests.Session, "get", return_value=_http_response(private_page)),
        pytest.raises(ValueError, match="original image container"),
    ):
        dl.images._resolve_one_image_url("/image/5/view/")


def _download_image_one(server: ScriptedServer, fake_s3) -> ImageFailures:
    return _site_downloader(server, fake_s3).images.download_images(
        [ImageRef(1, "img1.jpg", 1)], 1, "b", "p"
    )


_IMAGE_KEY = ("b", "p/s1/images/1.jpg")


def test_an_image_body_cut_short_is_downloaded_again(http_server, fake_s3):
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, "/media/1.jpg")))
    short = {**_JPEG_HEADERS, "Content-Length": str(len(_JPEG))}
    http_server.script(
        "/media/1.jpg", Reply(200, _JPEG[:100], short), Reply(200, _JPEG, _JPEG_HEADERS)
    )

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == _JPEG
    assert len(http_server.requests_to("/media/1.jpg")) == 2


def test_a_failed_image_is_logged_without_its_media_url_query(
    http_server, fake_s3, caplog: pytest.LogCaptureFixture
):
    signed = "/media/1.jpg?AWSAccessKeyId=A&Signature=S"
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, signed)))
    caplog.set_level(logging.DEBUG)

    assert _download_image_one(http_server, fake_s3) == ImageFailures(upload_failed=1)

    failed = [
        r.getMessage() for r in caplog.records if "Image 1 failed (upload_failed)" in r.getMessage()
    ]
    assert failed
    assert all("?<redacted>" in m for m in failed)
    assert not any("Signature" in r.getMessage() for r in caplog.records)


def test_an_expired_image_url_is_resolved_again_from_the_view_page(http_server, fake_s3):
    http_server.script(
        "/image/1/view/",
        Reply(200, _view_page(http_server, "/media/old.jpg")),
        Reply(200, _view_page(http_server, "/media/new.jpg")),
    )
    expired = b"<Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>"
    http_server.script("/media/old.jpg", Reply(403, expired, {"Content-Type": "application/xml"}))
    http_server.script("/media/new.jpg", Reply(200, _JPEG, _JPEG_HEADERS))

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == _JPEG
    assert len(http_server.requests_to("/image/1/view/")) == 2


def test_a_jpeg_with_bytes_after_its_end_marker_is_uploaded_with_its_format(http_server, fake_s3):
    body = _JPEG + b"\x00trailing"
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, "/media/1.jpg")))
    http_server.script("/media/1.jpg", Reply(200, body, _JPEG_HEADERS))

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == body
    assert fake_s3.headers[_IMAGE_KEY] == {
        "ContentType": "image/jpeg",
        "Metadata": {"source-format": "JPEG", "source-bytes": str(len(body))},
    }


def _noise_image_bytes(fmt: str) -> bytes:
    """A 256x256 random-pixel image, large enough that cutting it in half drops scan data."""
    pixels = random.Random(0).randbytes(256 * 256 * 3)
    image = Image.frombytes("RGB", (256, 256), pixels)
    buf = io.BytesIO()
    if fmt == "MPO":  # a single-frame MPO save is written as a plain JPEG
        image.save(buf, fmt, save_all=True, append_images=[image])
    else:
        image.save(buf, fmt)
    return buf.getvalue()


_NOISE_JPEG = _noise_image_bytes("JPEG")
_NOISE_PNG = _noise_image_bytes("PNG")


def _serve_image(server: ScriptedServer, body: bytes, headers: dict[str, str]) -> None:
    server.script("/image/1/view/", Reply(200, _view_page(server, "/media/1.jpg")))
    server.script("/media/1.jpg", Reply(200, body, headers))


def test_an_octet_stream_jpeg_is_stored_as_image_jpeg(http_server, fake_s3):
    _serve_image(http_server, _NOISE_JPEG, {"Content-Type": "binary/octet-stream"})

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.headers[_IMAGE_KEY]["ContentType"] == "image/jpeg"


def test_a_png_served_as_image_jpeg_is_stored_with_the_detected_format(http_server, fake_s3):
    _serve_image(http_server, _NOISE_PNG, _JPEG_HEADERS)

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == _NOISE_PNG
    assert fake_s3.headers[_IMAGE_KEY] == {
        "ContentType": "image/png",
        "Metadata": {"source-format": "PNG", "source-bytes": str(len(_NOISE_PNG))},
    }


# Content-Encoding makes the downloader skip its Content-Length check, so the body's own
# decodability is the only thing that can reject the cut image.


class _NeverStreamed:
    """A response declaring a body of 11 bytes whose body must not be read."""

    headers = {"Content-Length": "11"}  # noqa: RUF012

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        raise AssertionError("body read despite a declared length over the cap")


def test_pinned_images_are_not_downloaded(fake_s3):
    dl = CoralNetDownloader("u", "p", max_workers=2, s3=fake_s3)
    fake_s3.put("b", "p/s1/images/1.jpg", b"JPEG!")

    with patch.object(dl.images, "download_images", return_value=ImageFailures()) as download:
        dl.images.download_missing_images(_refs({1: [1, 2, 3]}), 1, "b", "p", pinned=frozenset({2}))

    assert [r.id for r in download.call_args.args[0]] == [3]
