"""Original-image downloads: resolve, fetch, validate and upload each image."""

from __future__ import annotations

import concurrent
import io
import logging
import threading
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import requests
from bs4 import BeautifulSoup
from PIL import Image

from coralnet_scraper import s3_io
from coralnet_scraper.config import DEFAULT_PREFIX
from coralnet_scraper.errors import ImageInvalid, UrlExpired, _retrying
from coralnet_scraper.http import redact_url_query
from coralnet_scraper.models import ImageFailures
from coralnet_scraper.session import _PAGE_GET_TIMEOUT, CoralNetSession
from coralnet_scraper.walker import ImageRef

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # survey images legitimately exceed Pillow's bomb limit

_IMAGE_ATTEMPTS = 4
_IMAGE_OUTAGE_STREAK = 20  # consecutive failed images that stop the image stage
_IMAGE_TIMEOUT = (10, 60)  # connect, idle read
_IMAGE_DEADLINE_S = 300.0  # wall-clock budget for one image body
_IMAGE_CHUNK_BYTES = 64 * 1024
_IMAGE_MAX_BYTES = 256 * 1024 * 1024
_IMAGE_FORMATS = (
    "JPEG",
    "PNG",
)  # the JPEG opener also yields MPO files; "MPO" itself is not an opener
_IMAGE_CONTENT_TYPES = ("image/", "binary/octet-stream", "application/octet-stream")


def validate_image_bytes(data: io.BytesIO) -> str:
    """Return the image format of ``data``, such as ``JPEG``, leaving it positioned at 0.

    Only JPEG, MPO (opened as JPEG) and PNG are accepted. The image is verified, then fully decoded, so a
    truncated body fails here.

    Raises:
        ImageInvalid: If ``data`` is empty or Pillow cannot open, verify and decode it as one
            of the accepted formats.
    """
    if data.getbuffer().nbytes == 0:
        raise ImageInvalid("empty body")
    try:
        data.seek(0)
        with Image.open(data, formats=_IMAGE_FORMATS) as im:
            fmt = im.format or ""
            im.verify()
        data.seek(0)
        with Image.open(data, formats=_IMAGE_FORMATS) as im:
            im.load()
    except Exception as e:  # noqa: BLE001 - Pillow raises many types for bad data
        raise ImageInvalid(f"not a decodable image: {type(e).__name__}") from e
    finally:
        data.seek(0)
    return fmt


def _check_image_response(resp: requests.Response) -> str:
    """Return the content type of an original-image response that may carry an image.

    Raises:
        UrlExpired: If the signed URL has expired.
        requests.HTTPError: If the status is 4xx or 5xx.
        ImageInvalid: If the status is not 2xx or the content type is not an image one.
    """
    if resp.status_code == 403 and b"expired" in resp.content.lower():
        raise UrlExpired(resp.url)
    resp.raise_for_status()
    if not 200 <= resp.status_code < 300:
        raise ImageInvalid(f"status {resp.status_code}")
    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith(_IMAGE_CONTENT_TYPES):
        raise ImageInvalid(f"content type {content_type!r}")
    return content_type


def _read_image_body(resp: requests.Response, deadline: float) -> io.BytesIO:
    """Return the streamed body of ``resp``, positioned at its end.

    Raises:
        ImageInvalid: If the body is still arriving at the monotonic time ``deadline``, it
            exceeds :data:`_IMAGE_MAX_BYTES` (by an unencoded ``Content-Length``, before any
            read, or by the bytes read), or its length differs from an unencoded
            ``Content-Length``.
    """
    declared = resp.headers.get("Content-Length")
    unencoded = "Content-Encoding" not in resp.headers
    if declared is not None and unencoded and int(declared) > _IMAGE_MAX_BYTES:
        raise ImageInvalid(f"declared {declared} bytes, over the {_IMAGE_MAX_BYTES} byte cap")
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=_IMAGE_CHUNK_BYTES):
        buf.write(chunk)
        if buf.tell() > _IMAGE_MAX_BYTES:
            raise ImageInvalid(f"body over the {_IMAGE_MAX_BYTES} byte cap")
        if time.monotonic() > deadline:
            raise ImageInvalid(f"image body took longer than {_IMAGE_DEADLINE_S:.0f} s")
    if declared is not None and unencoded and buf.tell() != int(declared):
        raise ImageInvalid(f"received {buf.tell()} of {declared} bytes")
    return buf


def _next_image_streak(streak: int, outcome: str) -> int:
    """Return the failed-image streak after ``outcome``: reset by ok, unchanged by skipped."""
    if outcome == "ok":
        return 0
    return streak if outcome == "skipped" else streak + 1


class ImageFetcher:
    """Download a source's original images to S3."""

    def __init__(self, auth: CoralNetSession, s3: Any, *, max_workers: int) -> None:
        """Create the fetcher; no network call is made.

        Args:
            auth: The CoralNet session whose sessions resolve and fetch the images.
            s3: S3 client the images are uploaded to.
            max_workers: Concurrent image download workers.
        """
        self.auth = auth
        self.s3 = s3
        self.max_workers = max_workers

    def _resolve_one_image_url(self, rel: str) -> str:
        """Return the original-image URL found on the image page at the relative link ``rel``.

        Raises:
            ValueError: If the page has no original image, e.g. a private source.
        """
        session = self.auth._get_thread_session()
        url = urllib.parse.urljoin(self.auth.CORALNET_URL + "/", rel)
        r = session.get(url, timeout=_PAGE_GET_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        originals = soup.select("div#original_image_container > img")
        src = originals[0].attrs.get("src") if originals else None
        if not src:
            raise ValueError(
                f"CoralNet image {url}: couldn't find original image container (private source?)."
            )
        return cast(str, src)

    def download_image_to_s3(self, url: str, bucket_name: str, s3_key: str) -> int:
        """Download the original image at ``url``, validate it, and upload it to ``s3_key``.

        The body must arrive within :data:`_IMAGE_DEADLINE_S` seconds and fit in
        :data:`_IMAGE_MAX_BYTES`. The object's content type is the MIME type of the format
        Pillow detects, whatever the server declared; the format and byte count are
        metadata.

        Returns:
            The number of bytes uploaded.

        Raises:
            UrlExpired: If the signed URL has expired.
            ImageInvalid: If the response or its body is not a whole, decodable image;
                nothing is uploaded.
            requests.RequestException: On a transport error or an error status.
        """
        deadline = time.monotonic() + _IMAGE_DEADLINE_S
        with self.auth.media.get(url, stream=True, timeout=_IMAGE_TIMEOUT) as response:
            _check_image_response(response)
            body = _read_image_body(response, deadline)
        fmt = validate_image_bytes(body)
        size = body.getbuffer().nbytes
        self.s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=body,
            ContentType=Image.MIME.get(fmt, "application/octet-stream"),
            Metadata={"source-format": fmt, "source-bytes": str(size)},
        )
        return size

    def _download_image(
        self, ref: ImageRef, bucket_name: str, s3_key: str, stop: threading.Event
    ) -> str:
        """Resolve and download one image as one retried unit, so an expired URL is resolved again.

        ``stop`` is checked before each attempt: set before the first, the image is not tried;
        set later, it is not retried.

        Returns:
            ``"ok"``, ``"skipped"`` when ``stop`` was set before the first attempt, or the
            failure kind of the last attempt: ``"missing_url"`` when its view page gave no
            URL, ``"upload_failed"`` when the download or upload failed.
        """
        stage = "missing_url"
        tried = False
        try:
            for attempt in _retrying(_IMAGE_ATTEMPTS):
                if stop.is_set():
                    if not tried:
                        return "skipped"
                    logger.warning("Image %s failed (%s), stage stopped", ref.id, stage)
                    return stage
                tried = True
                with attempt:
                    stage = "missing_url"
                    url = self._resolve_one_image_url(ref.view_path)
                    stage = "upload_failed"
                    self.download_image_to_s3(url, bucket_name, s3_key)
        except Exception as e:  # noqa: BLE001 - counted as the image's failure kind
            logger.warning("Image %s failed (%s): %s", ref.id, stage, redact_url_query(str(e)))
            return stage
        return "ok"

    def download_images(
        self,
        refs: list[ImageRef],
        source_id: int,
        bucket_name: str,
        s3_prefix: str = DEFAULT_PREFIX,
    ) -> ImageFailures:
        """Download every image of ``refs`` to S3 across :attr:`max_workers` threads.

        After :data:`_IMAGE_OUTAGE_STREAK` consecutive failed images the stage stops: images
        not yet started are skipped and running ones are not retried. A skipped image neither
        extends nor resets the streak.

        Args:
            refs: Images to download, whatever is already on S3.
            source_id: CoralNet source id.
            bucket_name: Destination S3 bucket.
            s3_prefix: Destination key prefix.

        Returns:
            Failure counts by kind, as :meth:`_download_image` reports them.

        Raises:
            KeyboardInterrupt: After cancelling the images not yet started.
        """
        outcomes: Counter[str] = Counter()
        stop = threading.Event()
        streak = 0
        logger.info(
            "Image upload for source %s: %d images, %d workers",
            source_id,
            len(refs),
            self.max_workers,
        )
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._download_image,
                    ref,
                    bucket_name,
                    s3_io.object_key_image(s3_prefix, source_id, str(ref.id)),
                    stop,
                )
                for ref in refs
            ]
            try:
                for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                    outcome = fut.result()
                    outcomes[outcome] += 1
                    streak = _next_image_streak(streak, outcome)
                    if streak >= _IMAGE_OUTAGE_STREAK and not stop.is_set():
                        stop.set()
                        logger.warning(
                            "Image upload for source %s stopped after %d consecutive failures",
                            source_id,
                            streak,
                        )
                    if i % 50 == 0 or i == len(futures):
                        logger.info(
                            "Image upload progress source %s: %s/%s", source_id, i, len(refs)
                        )
            except KeyboardInterrupt:
                stop.set()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        logger.info("Uploaded %s/%s images for source %s", outcomes["ok"], len(refs), source_id)
        return ImageFailures(
            missing_url=outcomes["missing_url"],
            upload_failed=outcomes["upload_failed"],
            skipped=outcomes["skipped"],
        )

    def _download_missing_images(
        self, refs: list[ImageRef], source_id: int, bucket_name: str, s3_prefix: str
    ) -> ImageFailures:
        """Download every image of ``refs`` that has no non-empty object on S3 yet.

        Returns:
            The failure counts of :meth:`download_images`.
        """
        images_prefix = f"{s3_io.source_prefix(s3_prefix, source_id)}images/"
        stored = {
            key
            for key, size in s3_io.object_sizes(self.s3, bucket_name, images_prefix).items()
            if size > 0
        }
        todo = [
            r for r in refs if s3_io.object_key_image(s3_prefix, source_id, str(r.id)) not in stored
        ]
        logger.info(
            "Source %s: %d of %d images already on S3", source_id, len(refs) - len(todo), len(refs)
        )
        failures = self.download_images(todo, source_id, bucket_name, s3_prefix)
        if failures != ImageFailures():
            logger.warning(
                "Source %s image failures: %d missing URL, %d upload failed",
                source_id,
                failures.missing_url,
                failures.upload_failed,
            )
        return failures
