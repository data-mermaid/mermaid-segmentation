"""CoralNet session client: login, probe, and S3-backed downloads."""

from __future__ import annotations

import concurrent
import importlib.metadata
import io
import logging
import re
import threading
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from queue import Queue
from typing import Any, cast

import boto3
import pandas as pd
import requests
import stamina
from botocore.config import Config as BotoConfig
from bs4 import BeautifulSoup
from PIL import Image

from coralnet_scraper import s3_io
from coralnet_scraper.config import (
    get_export_workers,
    get_requests_per_second,
    get_scraper_workers,
    validate_requests_per_second,
    validate_workers,
)
from coralnet_scraper.http import SessionFactory
from coralnet_scraper.models import DownloadResult, ImageFailures
from coralnet_scraper.parsers import (
    classifier_plot_data_from_source_html,
    extract_csrf_token,
    parse_export_annotations_prep_form,
    parse_image_status_counts,
    parse_source_access,
    parse_total_images_from_source_html,
)
from coralnet_scraper.verify import (
    Verification,
    _strip_status,
    orphan_image_keys,
    verify_source,
)

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # survey images legitimately exceed Pillow's bomb limit

_SOURCE_INACCESSIBLE_STATUSES = frozenset({403, 404})  # missing or forbidden source overview
_PAGE_GET_TIMEOUT = 90  # simple page GETs; CoralNet can be slow under load
_EXPORT_PREP_TIMEOUT_MIN = 300
_EXPORT_PREP_TIMEOUT_MAX = 7200  # 2 hours
_EXPORT_SERVE_TIMEOUT_MIN = 120
_EXPORT_SERVE_TIMEOUT_MAX = 1800
_EXPORT_PREP_TIMEOUT_STEP = 300  # export_prep read timeout added per further attempt
_EXPORT_PREP_TIMEOUT_RETRY_WAIT_S = (
    60  # pause after an export_prep read timeout, before the next POST
)
_EXPORT_PREP_HEARTBEAT_S = 60  # log while a long export_prep POST is still running
_PAGINATED_EXPORT_THRESHOLD = 5000  # sources above this use per-page export
_PAGE_EXPORT_TIMEOUT = 180  # CoralNet session setup can take >60s even for 20 images
_PAGE_DELAY_SECONDS = 2.0  # politeness delay between page export requests
_WALK_CHECKPOINT_EVERY_PAGES = 50  # save the walk's progress to _partial/walk.json every N pages
_BROWSE_PROGRESS_EVERY = 10  # log walk progress every N pages
_EXPORT_ATTEMPTS = 4
_WALK_ATTEMPTS = 3
_IMAGE_ATTEMPTS = 4
_IMAGE_OUTAGE_STREAK = 20  # consecutive failed images that stop the image stage
_LOGIN_ATTEMPTS = 4
_IMAGE_TIMEOUT = (10, 60)  # connect, idle read
_IMAGE_DEADLINE_S = 300.0  # wall-clock budget for one image body
_IMAGE_CHUNK_BYTES = 64 * 1024
_IMAGE_MAX_BYTES = 256 * 1024 * 1024
_IMAGE_FORMATS = (
    "JPEG",
    "PNG",
)  # the JPEG opener also yields MPO files; "MPO" itself is not an opener
_IMAGE_CONTENT_TYPES = ("image/", "binary/octet-stream", "application/octet-stream")
# stamina's default 45 s total timeout would end the retries of an export running for minutes.
_RETRY_WAITS: dict[str, Any] = {
    "timeout": None,
    "wait_initial": 5.0,
    "wait_max": 120.0,
    "wait_jitter": 5.0,
}
_EXPORT_OPTIONS: dict[str, Any] = {
    "browse_action": "export_annotations",
    "label_format": "both",
    "optional_columns": ["annotator_info", "metadata_date_aux", "metadata_other"],
}
_STATE_FILE = "state.json"
_WALK_FILE = "walk.json"
_SCRAPER_VERSION = importlib.metadata.version("coralnet-scraper")

_IMAGE_HREF_PK = re.compile(r"/image/(\d+)(?:/|$)")
_CHUNK_PAGE = re.compile(r"/page-(\d+)\.csv$")


class ContentError(Exception):
    """A successful response whose content is not what the request should return."""


class WalkIncomplete(ContentError):
    """The browse walk breaks a page rule or does not match the overview's image count."""


class SessionExpired(Exception):
    """CoralNet treated a logged-in session's request as logged out."""


class ImageInvalid(Exception):
    """A downloaded image body failed validation."""


class UrlExpired(Exception):
    """The signed original-image URL has expired."""


class _ExportPrepTimeout(requests.ReadTimeout):
    """The ``export_prep`` POST, not a GET, timed out reading its response."""


class _TotalChanged(WalkIncomplete):
    """The overview's image count changed while the walk ran."""

    def __init__(self, total: int) -> None:
        """Record the overview's new image count ``total``."""
        super().__init__(f"overview image count changed to {total} during the walk")
        self.total = total


def _is_retryable(exc: BaseException) -> bool:
    """Return whether an attempt that raised ``exc`` is worth repeating.

    Content, session and image failures, transport errors and 5xx statuses are; a 4xx
    status and any other error are not.
    """
    if isinstance(exc, requests.HTTPError):
        return exc.response is None or exc.response.status_code >= 500
    return isinstance(
        exc,
        ContentError | SessionExpired | ImageInvalid | UrlExpired | requests.RequestException,
    )


def _retrying(attempts: int) -> Any:
    """Return a stamina retry context of ``attempts`` attempts with the scraper's waits."""
    return stamina.retry_context(on=_is_retryable, attempts=attempts, **_RETRY_WAITS)


def decode_export_csv(resp: requests.Response) -> pd.DataFrame:
    """Return an ``export/serve`` response as an annotations frame.

    CoralNet serves ``text/csv`` without a charset, which ``requests`` decodes as
    ISO-8859-1; the body is UTF-8, so it is decoded from the bytes.

    Raises:
        SessionExpired: If the request landed on the login page.
        ContentError: If the body is not ``text/csv``, not a UTF-8 CSV, has a column
            holding HTML or an unnamed one, lacks a ``Name`` column, or has a row without
            a name.
    """
    if "/accounts/login/" in resp.url:
        raise SessionExpired("export/serve landed on the login page")
    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("text/csv"):
        raise ContentError(f"export/serve returned {content_type or 'no content type'}")
    try:
        df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ContentError(f"export/serve body is not a UTF-8 CSV: {type(e).__name__}") from e
    if any("<" in str(c) or str(c).startswith("Unnamed") for c in df.columns):
        raise ContentError("export CSV has an HTML or unnamed column")
    if "Name" not in df.columns:
        raise ContentError("export CSV has no Name column")
    if df["Name"].isna().any():
        raise ContentError("export CSV has rows without a Name")
    return df


def _export_timestamp(resp: requests.Response) -> str:
    """Return the ``session_data_timestamp`` of an ``export_prep`` response.

    Raises:
        SessionExpired: If CoralNet answers that the user must be signed in.
        ContentError: If the body is not JSON, reports another error, or has no timestamp.
    """
    try:
        body = resp.json()
    except ValueError as e:
        raise ContentError("export_prep did not return JSON") from e
    if not isinstance(body, dict):
        raise ContentError("export_prep did not return a JSON object")
    if "error" in body:
        if "signed in" in str(body["error"]):
            raise SessionExpired(str(body["error"]))
        raise ContentError(f"export_prep error: {body['error']}")
    if "session_data_timestamp" not in body:
        raise ContentError("export_prep returned no session_data_timestamp")
    return str(body["session_data_timestamp"])


def _export_names(refs: list[ImageRef]) -> set[str]:
    """Return the image names an export of ``refs`` may contain: titles without a status."""
    return {_strip_status(r.name) for r in refs}


def _check_export_names(df: pd.DataFrame, allowed: set[str]) -> None:
    """Raise :class:`ContentError` if the export has a row for an image outside ``allowed``."""
    foreign = set(df["Name"].astype(str)) - allowed
    if foreign:
        raise ContentError(
            f"export has rows for {len(foreign)} images not requested, e.g. {sorted(foreign)[:2]}"
        )


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


@dataclass(frozen=True)
class ImageRef:
    """One image found by the browse walk.

    Attributes:
        id: CoralNet image id.
        name: Thumbnail title, as listed in ``image_list.csv``.
        page: Browse page the image is listed on, counted from 1.
    """

    id: int
    name: str
    page: int

    @property
    def view_path(self) -> str:
        """Return the relative link of the image's view page."""
        return f"/image/{self.id}/view/"


class _WalkCountMismatch(WalkIncomplete):
    """The browse walk found a different number of images than the overview lists."""

    def __init__(self, found: int, expected: int) -> None:
        super().__init__(f"walk_count_mismatch:{found}!={expected}")


@dataclass
class _WalkProgress:
    """The images a browse walk has found so far, as saved in ``_partial/walk.json``."""

    refs: dict[int, ImageRef] = field(default_factory=dict)
    last_page: int = 0
    page_size: int = 0
    complete: bool = False

    def add_page(self, thumbs: list[tuple[int, str]]) -> None:
        """Record the next page's thumbnails; an id already seen keeps its first page."""
        self.last_page += 1
        self.page_size = self.page_size or len(thumbs)
        for image_id, name in thumbs:
            self.refs.setdefault(image_id, ImageRef(image_id, name, self.last_page))

    def to_json(self) -> dict[str, Any]:
        """Return the JSON document of this progress."""
        return {
            "last_page": self.last_page,
            "page_size": self.page_size,
            "complete": self.complete,
            "refs": [{"id": r.id, "name": r.name, "page": r.page} for r in self.refs.values()],
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> _WalkProgress:
        """Return the progress saved as ``doc`` by :meth:`to_json`."""
        refs = {r["id"]: ImageRef(r["id"], r["name"], r["page"]) for r in doc["refs"]}
        return cls(refs, doc["last_page"], doc["page_size"], doc["complete"])


def _parse_browse_thumbs(html: str) -> tuple[list[tuple[int, str]], str | None, bool]:
    """Return a browse page's thumbnails, its "Next page" link and whether it has the export form.

    Thumbnails are ``(image id, title)``. A thumbnail without an ``/image/<id>/`` link is
    skipped; one without a title gets ``""``.
    """
    soup = BeautifulSoup(html, "html.parser")
    thumbs = []
    for wrapper in soup.find_all("span", class_="thumb_wrapper"):
        link = wrapper.find("a")
        match = _IMAGE_HREF_PK.search(cast(str, link.get("href", ""))) if link else None
        if match:
            img = wrapper.find("img")
            thumbs.append((int(match.group(1)), cast(str, img.get("title", "")) if img else ""))
    next_anchor = soup.find("a", title="Next page")
    next_href = cast("str | None", next_anchor.get("href")) if next_anchor else None
    has_form = soup.find("form", {"id": "export-annotations-prep-form"}) is not None
    return thumbs, next_href, has_form


def _check_walk_page(
    progress: _WalkProgress, thumbs: list[tuple[int, str]], has_next: bool
) -> None:
    """Raise :class:`WalkIncomplete` if the next page of ``progress`` breaks a walk rule.

    The page must list images, a page with a Next link must list as many as page 1, and
    no image id may repeat one already walked or another on the page.
    """
    page = progress.last_page + 1
    if not thumbs:
        raise WalkIncomplete(f"browse page {page} lists no images")
    if has_next and progress.page_size and len(thumbs) != progress.page_size:
        raise WalkIncomplete(
            f"browse page {page} lists {len(thumbs)} images, page 1 lists {progress.page_size}"
        )
    ids = [image_id for image_id, _ in thumbs]
    if len(set(ids)) != len(ids) or any(image_id in progress.refs for image_id in ids):
        raise WalkIncomplete(f"browse page {page} repeats an image id")


def _refs_by_page(refs: list[ImageRef]) -> dict[int, list[ImageRef]]:
    """Return the images of each browse page, pages ascending, images in walk order."""
    pages: dict[int, list[ImageRef]] = {}
    for ref in refs:
        pages.setdefault(ref.page, []).append(ref)
    return {page: pages[page] for page in sorted(pages)}


def _image_list_frame(refs: list[ImageRef]) -> pd.DataFrame:
    """Return the ``image_list.csv`` frame of the walk; ``Image URL`` stays empty."""
    return pd.DataFrame(
        {
            "Name": [r.name for r in refs],
            "Image Page": [r.view_path for r in refs],
            "Image URL": pd.NA,
        }
    )


def _partial_key(s3_prefix: str, source_id: int, name: str) -> str:
    """Return the key of the file ``name`` in a source's ``_partial/``."""
    return f"{s3_io.partial_prefix(s3_prefix, source_id)}{name}"


def _image_failure_codes(failures: ImageFailures) -> list[str]:
    """Return the error codes of a source's image failures."""
    codes = []
    if failures.missing_url > 0:
        codes.append("images_missing_url")
    if failures.upload_failed > 0:
        codes.append("images_upload_failed")
    if failures.skipped > 0:
        codes.append("images_outage")
    return codes


def _next_image_streak(streak: int, outcome: str) -> int:
    """Return the failed-image streak after ``outcome``: reset by ok, unchanged by skipped."""
    if outcome == "ok":
        return 0
    return streak if outcome == "skipped" else streak + 1


def _upload_csv(s3_client: Any, bucket: str, key: str, df: pd.DataFrame) -> None:
    """Serialize ``df`` as CSV without the index and upload it to ``key`` in ``bucket``."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3_io.upload_csv_body(s3_client, bucket=bucket, key=key, csv_text=buf.getvalue())


def export_prep_timeout_seconds(total_images: int | None) -> int:
    """Read timeout for CoralNet ``export_prep`` POST (server builds the export)."""
    n = total_images or 0
    if n <= 0:
        return _EXPORT_PREP_TIMEOUT_MIN
    return min(_EXPORT_PREP_TIMEOUT_MAX, max(_EXPORT_PREP_TIMEOUT_MIN, 120 + n // 20))


def export_serve_timeout_seconds(total_images: int | None) -> int:
    """Read timeout for CoralNet ``export/serve`` GET (download generated CSV)."""
    n = total_images or 0
    if n <= 0:
        return _EXPORT_SERVE_TIMEOUT_MIN
    return min(_EXPORT_SERVE_TIMEOUT_MAX, max(_EXPORT_SERVE_TIMEOUT_MIN, 60 + n // 50))


class CoralNetDownloader:
    """Download CoralNet source artifacts using ``requests`` and upload to S3."""

    CORALNET_URL = "https://coralnet.ucsd.edu"
    LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"

    def __init__(
        self,
        username: str,
        password: str,
        *,
        max_workers: int | None = None,
        export_workers: int | None = None,
        requests_per_second: float | None = None,
    ) -> None:
        """Create the client and its HTTP and S3 sessions; no network call is made.

        Args:
            username: CoralNet account name.
            password: CoralNet account password.
            max_workers: Concurrent image download workers; ``None`` reads the environment
                or falls back to the CPU-based default.
            export_workers: Concurrent page exports of a paginated export; ``None`` reads the
                environment or falls back to 3.
            requests_per_second: Rate limit shared by every CoralNet session; ``None`` reads
                the environment, where unset means no limit.

        Raises:
            ValueError: If a worker count is not an integer of at least 1, or the rate is
                not a positive number.
        """
        self.username = username
        self.password = password
        self.max_workers = (
            validate_workers(max_workers) if max_workers is not None else get_scraper_workers()
        )
        self.export_workers = (
            validate_workers(export_workers) if export_workers is not None else get_export_workers()
        )
        self.requests_per_second = (
            validate_requests_per_second(requests_per_second)
            if requests_per_second is not None
            else get_requests_per_second()
        )
        # Pool sized for concurrent image URL resolution
        self.sessions = SessionFactory(
            pool_size=self.max_workers + 4, requests_per_second=self.requests_per_second
        )
        self.session = self.sessions.coralnet()
        self.media = self.sessions.media()
        self.logged_in = False
        self._login_lock = threading.Lock()
        self._thread_local = threading.local()
        self._export_thread_local = threading.local()
        self.s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=self.max_workers + 4))

    def transport_stats(self) -> dict[str, Any]:
        """Return the request accounting of every session of this client."""
        return self.sessions.stats()

    def _source_main_url(self, source_id: int) -> str:
        """Return the URL of a source's overview page."""
        return f"{self.CORALNET_URL}/source/{source_id}/"

    def _get_source_overview_html(self, source_id: int) -> str:
        """Fetch the overview page of a source and return its HTML."""
        url = self._source_main_url(source_id)
        response = self.session.get(url, timeout=_PAGE_GET_TIMEOUT)
        response.raise_for_status()
        return response.text

    def _perform_login(self, session: requests.Session) -> None:
        """Execute the CoralNet login flow on ``session`` (independent cookie jar).

        Pure side-effect helper: pops a fresh login CSRF, POSTs credentials, and
        leaves ``session`` with a logged-in ``sessionid`` cookie, mutating no other
        state. Callers compose this into :meth:`login` (the main session) and
        :meth:`_new_logged_in_session` (per-worker independent sessions used by
        parallel export).

        Raises:
            RuntimeError: If the login page has no CSRF token or the login fails.
        """
        response = session.get(self.LOGIN_URL, timeout=_PAGE_GET_TIMEOUT)
        response.raise_for_status()
        csrf_token = extract_csrf_token(response.text)
        if not csrf_token:
            raise RuntimeError("Could not find CSRF token")
        for attempt in _retrying(_LOGIN_ATTEMPTS):
            with attempt:
                login_response = session.post(
                    self.LOGIN_URL,
                    data={
                        "username": self.username,
                        "password": self.password,
                        "csrfmiddlewaretoken": csrf_token,
                    },
                    headers={"Referer": self.LOGIN_URL},
                    timeout=_PAGE_GET_TIMEOUT,
                    allow_redirects=True,
                )
                login_response.raise_for_status()
        if "Sign out" not in login_response.text:
            raise RuntimeError("Login failed - the result page shows no signed-in user")

    def _relogin(self, session: requests.Session) -> None:
        """Log ``session`` in again after CoralNet treated it as logged out."""
        with self._login_lock:
            logger.warning("CoralNet session expired; logging in again")
            session.cookies.clear()
            self._perform_login(session)

    def login(self) -> bool:
        """Log the main session in to CoralNet.

        Returns:
            True on success; False when the login fails, after logging the error.
        """
        try:
            self._perform_login(self.session)
            self.logged_in = True
            logger.info("CoralNet login successful for %s", self.username)
            return True
        except Exception:
            logger.exception("CoralNet login failed for %s", self.username)
            return False

    _META_KEYS = ("x", "y", "nimages", "date", "traintime", "pk")
    _META_COLS = ("Classifier nbr", "Accuracy", "Trained on", "Date", "Traintime", "Global id")

    def download_metadata(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
        *,
        overview_html: str | None = None,
    ) -> tuple[bool, int]:
        """Upload the source's classifier history as ``metadata.csv``.

        Args:
            source_id: CoralNet source id.
            bucket_name: Destination S3 bucket.
            s3_prefix: Destination key prefix.
            overview_html: Already-fetched overview page; fetched when ``None``.

        Returns:
            ``(ok, total_images)`` where ``total_images`` is the website image count, or 0
            when the step fails.
        """
        try:
            html = (
                overview_html
                if overview_html is not None
                else self._get_source_overview_html(source_id)
            )
            total_images = parse_total_images_from_source_html(html)
            if total_images:
                logger.info("Total images (website): %s", total_images)

            classifier_data = classifier_plot_data_from_source_html(html)
            if not classifier_data:
                logger.warning("No classifier metadata found for source %s", source_id)
                return True, total_images

            meta_df = pd.DataFrame(
                [[pt.get(k) for k in self._META_KEYS] for pt in classifier_data],
                columns=list(self._META_COLS),
            )
            key = s3_io.object_key_metadata_csv(s3_prefix, source_id)
            _upload_csv(self.s3, bucket_name, key, meta_df)
            logger.info("Metadata saved s3://%s/%s", bucket_name, key)
            return True, total_images
        except Exception:
            logger.exception("download_metadata failed source %s", source_id)
            return False, 0

    def download_labelset(
        self, source_id: int, bucket_name: str, s3_prefix: str = "coralnet-public-images"
    ) -> bool:
        """Upload the source's labelset as ``labelset.csv``.

        Args:
            source_id: CoralNet source id.
            bucket_name: Destination S3 bucket.
            s3_prefix: Destination key prefix.

        Returns:
            True on success or when the source has no labels; False on failure.
        """
        try:
            url = f"{self.CORALNET_URL}/source/{source_id}/labelset/"
            response = self.session.get(url, timeout=_PAGE_GET_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "label-table"})
            if table is None:
                raise RuntimeError("Unable to find the label table")
            rows = table.find_all("tr")
            if not rows or len(rows) <= 1:
                logger.warning("No labelset rows for source %s", source_id)
                return True
            label_ids, names, short_codes = [], [], []
            for row in rows[1:]:
                cells = row.find_all("td")
                if not cells:
                    continue
                link = cells[0].find("a")
                if link and link.get("href"):
                    label_ids.append(cast(str, link["href"]).split("/")[-2])
                    names.append(link.get_text().strip())
                    short_codes.append(cells[1].get_text().strip() if len(cells) > 1 else "")
            if not label_ids:
                logger.warning("No labels found in labelset for source %s", source_id)
                return True
            labelset_df = pd.DataFrame(
                {"Label ID": label_ids, "Name": names, "Short Code": short_codes}
            )
            key = s3_io.object_key_labelset_csv(s3_prefix, source_id)
            _upload_csv(self.s3, bucket_name, key, labelset_df)
            logger.info("Labelset saved s3://%s/%s", bucket_name, key)
            return True
        except Exception:
            logger.exception("download_labelset failed source %s", source_id)
            return False

    def download_annotations(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
        *,
        refs: list[ImageRef],
        total_images: int,
    ) -> bool:
        """Export the source's annotations and upload them as ``annotations.csv``.

        Does not check for an existing ``annotations.csv`` on S3. Sources above
        :data:`_PAGINATED_EXPORT_THRESHOLD` images use :meth:`download_annotations_paginated`.

        Args:
            source_id: CoralNet source id.
            bucket_name: Destination S3 bucket.
            s3_prefix: Destination key prefix.
            refs: The source's images from :meth:`walk_source`; the paginated export
                exports them page by page.
            total_images: Website image count; it picks the export path and sizes timeouts.

        Returns:
            True on success; False on failure, after logging the error. The export must
            pass :func:`decode_export_csv` and name only images of ``refs``; an export that
            does not, after the attempts of :meth:`_export_validated`, uploads nothing.
        """
        try:
            prep_timeout = export_prep_timeout_seconds(total_images)
            serve_timeout = export_serve_timeout_seconds(total_images)
            if total_images > 1000:
                logger.info(
                    "Source %s: %s images; export_prep timeout=%ss serve timeout=%ss",
                    source_id,
                    total_images,
                    prep_timeout,
                    serve_timeout,
                )

            if total_images > _PAGINATED_EXPORT_THRESHOLD:
                logger.info(
                    "Source %s: %s images (>%s threshold), using paginated export",
                    source_id,
                    total_images,
                    _PAGINATED_EXPORT_THRESHOLD,
                )
                return self.download_annotations_paginated(
                    source_id, bucket_name, s3_prefix, refs=refs
                )

            df_annotations = self._export_validated(
                self.session,
                source_id,
                self._browse_page_url(source_id, 1),
                {"image_select_type": "all", **_EXPORT_OPTIONS},
                _export_names(refs),
                prep_timeout=prep_timeout,
                serve_timeout=serve_timeout,
            )
            key = s3_io.object_key_annotation_csv(s3_prefix, source_id)
            _upload_csv(self.s3, bucket_name, key, df_annotations)
            logger.info("Annotations saved s3://%s/%s", bucket_name, key)
            return True
        except Exception:
            logger.exception("download_annotations failed source %s", source_id)
            return False

    def download_annotations_paginated(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
        *,
        refs: list[ImageRef],
        page_workers: int | None = None,
    ) -> bool:
        """Export annotations one browse page at a time for large sources.

        Each page covers ~20 images and returns in 2-10 seconds, avoiding the
        multi-hour blocking wait of the bulk ``image_select_type=all`` export.
        Pages are exported across ``page_workers`` threads, each sleeping
        :data:`_PAGE_DELAY_SECONDS` after a round trip. Each finished page is saved as
        ``_partial/page-<n>.csv``; a page with a saved chunk is not exported again.
        ``annotations.csv`` is assembled from the chunks in page order once all exist.

        Args:
            source_id: CoralNet source id.
            bucket_name: Destination S3 bucket.
            s3_prefix: Destination key prefix.
            refs: The source's images from :meth:`walk_source`, with their browse pages.
            page_workers: Concurrent page round trips; ``None`` uses ``self.export_workers``.

        Returns:
            True on success; False on failure, after logging the error. Chunks of pages
            that finished stay on S3.
        """
        try:
            pages = _refs_by_page(refs)
            saved = self._saved_chunk_pages(source_id, bucket_name, s3_prefix)
            todo = {page: page_refs for page, page_refs in pages.items() if page not in saved}
            logger.info(
                "source %s: paginated export of %d pages, %d already saved",
                source_id,
                len(pages),
                len(pages) - len(todo),
            )
            if todo:
                workers = page_workers if page_workers is not None else self.export_workers
                self._export_pages_parallel(
                    source_id, bucket_name, s3_prefix, todo, max(1, min(workers, len(todo)))
                )
            combined = self._assemble_chunks(source_id, bucket_name, s3_prefix, list(pages))
            annotations_key = s3_io.object_key_annotation_csv(s3_prefix, source_id)
            _upload_csv(self.s3, bucket_name, annotations_key, combined)
            logger.info(
                "source %s: %d annotation rows saved to s3://%s/%s",
                source_id,
                len(combined),
                bucket_name,
                annotations_key,
            )
            return True
        except Exception:
            logger.exception("download_annotations_paginated failed source %s", source_id)
            return False

    def _saved_chunk_pages(self, source_id: int, bucket_name: str, s3_prefix: str) -> set[int]:
        """Return the page numbers whose export chunk is saved in the source's ``_partial/``."""
        keys = s3_io.object_sizes(self.s3, bucket_name, s3_io.partial_prefix(s3_prefix, source_id))
        return {int(m.group(1)) for key in keys if (m := _CHUNK_PAGE.search(key))}

    def _assemble_chunks(
        self, source_id: int, bucket_name: str, s3_prefix: str, pages: list[int]
    ) -> pd.DataFrame:
        """Return the saved export chunks of ``pages`` concatenated in that order.

        Raises:
            FileNotFoundError: If a page has no saved chunk.
        """
        chunks = []
        for page in pages:
            key = _partial_key(s3_prefix, source_id, f"page-{page}.csv")
            chunk = s3_io.read_csv_s3(self.s3, bucket_name, key)
            if chunk is None:
                raise FileNotFoundError(f"s3://{bucket_name}/{key}")
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def _export_pages_parallel(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        pages: dict[int, list[ImageRef]],
        workers: int,
    ) -> None:
        """Export and save each page of ``pages`` (page number to its images) across workers.

        Raises:
            Exception: The first page failure, once every started page has finished; pages
                still queued never start.
        """
        failed = threading.Event()

        def run(page: int, page_refs: list[ImageRef]) -> int | None:
            if failed.is_set():
                return None
            try:
                return self._export_and_save_page(
                    source_id, bucket_name, s3_prefix, page, page_refs, stop=failed
                )
            except BaseException:
                failed.set()
                raise

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_page = {
                executor.submit(run, page, page_refs): page for page, page_refs in pages.items()
            }
            completed = 0
            for fut in concurrent.futures.as_completed(future_to_page):
                try:
                    annotations = fut.result()
                except Exception:
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
                if annotations is None:
                    continue
                completed += 1
                page = future_to_page[fut]
                logger.info(
                    "source %s page=%d images=%d annotations=%d completed=%d/%d",
                    source_id,
                    page,
                    len(pages[page]),
                    annotations,
                    completed,
                    len(pages),
                )

    def _export_and_save_page(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        page_num: int,
        page_refs: list[ImageRef],
        stop: threading.Event | None = None,
    ) -> int:
        """Export one page and save it as ``_partial/page-<n>.csv``; return its row count.

        ``stop`` ends the page's export retries once set (see :meth:`_export_validated`).
        """
        chunk = self._export_one_page(source_id, page_num, page_refs, stop=stop)
        key = _partial_key(s3_prefix, source_id, f"page-{page_num}.csv")
        _upload_csv(self.s3, bucket_name, key, chunk)
        return len(chunk)

    def _browse_page_url(self, source_id: int, page_num: int) -> str:
        """Return the URL of a source's browse page ``page_num``, counted from 1."""
        base = f"{self.CORALNET_URL}/source/{source_id}/browse/images/"
        return base if page_num == 1 else f"{base}?page={page_num}"

    def _export_one_page(
        self,
        source_id: int,
        page_num: int,
        page_refs: list[ImageRef],
        stop: threading.Event | None = None,
    ) -> pd.DataFrame:
        """Export the annotations of ``page_refs``, the images on browse page ``page_num``.

        Uses a worker-local, independently-logged-in session
        (:meth:`_get_export_session`) so concurrent workers don't trample each other's
        server-side Django session state.

        Sends ``image_select_type=selected`` plus ``image_id_list`` (underscore-
        separated PKs) so CoralNet's ``image_search_kwargs_to_queryset`` filters to just
        this page's images, rather than exporting the entire source.
        ``image_form_type=search`` is the hidden, ``required=True`` discriminator on
        CoralNet's ``ImageSearchForm`` -- omitting it makes the form invalid and the
        serve response returns HTML instead of CSV. CoralNet caps ``image_id_list`` at
        100 entries; browse pages have 20.

        ``stop`` ends the export's retries once set (see :meth:`_export_validated`).
        """
        form = {
            "image_form_type": "search",
            "image_select_type": "selected",
            "image_id_list": "_".join(str(r.id) for r in page_refs),
            **_EXPORT_OPTIONS,
        }
        chunk_df = self._export_validated(
            self._get_export_session(),
            source_id,
            self._browse_page_url(source_id, page_num),
            form,
            _export_names(page_refs),
            prep_timeout=_PAGE_EXPORT_TIMEOUT,
            serve_timeout=_PAGE_EXPORT_TIMEOUT,
            stop=stop,
        )
        time.sleep(_PAGE_DELAY_SECONDS)
        return chunk_df

    def _export_validated(
        self,
        session: requests.Session,
        source_id: int,
        page_url: str,
        form: dict[str, Any],
        allowed_names: set[str],
        *,
        prep_timeout: int,
        serve_timeout: int,
        stop: threading.Event | None = None,
    ) -> pd.DataFrame:
        """Return an export whose rows name only ``allowed_names``, retrying the round trip.

        Each further attempt allows ``export_prep`` :data:`_EXPORT_PREP_TIMEOUT_STEP` more
        seconds. A session CoralNet treats as logged out logs in again before the next attempt.
        When the previous attempt's ``export_prep`` POST timed out reading its response, the
        next attempt first sleeps :data:`_EXPORT_PREP_TIMEOUT_RETRY_WAIT_S` (on top of the retry
        wait), so CoralNet can finish the abandoned export before the next POST.

        Args:
            session: Logged-in session the export runs on.
            source_id: CoralNet source id.
            page_url: Browse page whose export form supplies the CSRF token.
            form: Export form fields besides the CSRF token.
            allowed_names: Image names the export may contain.
            prep_timeout: Read timeout of the first ``export_prep`` POST.
            serve_timeout: Read timeout of the ``export/serve`` GET.
            stop: When set before an attempt after the first, the previous attempt's error
                is raised without another attempt.

        Raises:
            Exception: The last attempt's error once no attempt is left, it is not
                worth repeating, or ``stop`` is set.
        """
        expired = False
        last_error: BaseException | None = None
        for attempt in _retrying(_EXPORT_ATTEMPTS):
            # Raised outside `with attempt`, so stamina does not retry it.
            if last_error is not None and stop is not None and stop.is_set():
                raise last_error
            if isinstance(last_error, _ExportPrepTimeout):
                time.sleep(_EXPORT_PREP_TIMEOUT_RETRY_WAIT_S)
            with attempt:
                try:
                    if expired:
                        self._relogin(session)
                        expired = False
                    timeout = min(
                        _EXPORT_PREP_TIMEOUT_MAX,
                        prep_timeout + (attempt.num - 1) * _EXPORT_PREP_TIMEOUT_STEP,
                    )
                    try:
                        df = self._export_round_trip(
                            session, source_id, page_url, form, timeout, serve_timeout
                        )
                    except SessionExpired:
                        expired = True
                        raise
                    _check_export_names(df, allowed_names)
                except Exception as e:
                    last_error = e
                    raise
        return df

    def _export_round_trip(
        self,
        session: requests.Session,
        source_id: int,
        page_url: str,
        form: dict[str, Any],
        prep_timeout: int,
        serve_timeout: int,
    ) -> pd.DataFrame:
        """Fetch the export form's CSRF token, run ``export_prep``, and decode the served CSV.

        Raises:
            SessionExpired: If the browse page has no export form, ``export_prep`` asks for
                a sign-in, or the serve lands on the login page.
            ContentError: If ``export_prep`` or the served CSV is not a valid answer.
            requests.RequestException: On a transport error or an error status; a
                ``export_prep`` read timeout is the subclass :class:`_ExportPrepTimeout`.
        """
        browse = session.get(page_url, timeout=_PAGE_EXPORT_TIMEOUT)
        browse.raise_for_status()
        csrf = parse_export_annotations_prep_form(browse.text)
        if not csrf:
            raise SessionExpired(f"browse page {page_url} has no export form")
        try:
            prep = self._post_export_prep(
                session, source_id, {"csrfmiddlewaretoken": csrf, **form}, page_url, prep_timeout
            )
        except requests.ReadTimeout as e:
            raise _ExportPrepTimeout(*e.args, request=e.request, response=e.response) from e
        prep.raise_for_status()
        timestamp = _export_timestamp(prep)
        serve = session.get(
            f"{self.CORALNET_URL}/source/{source_id}/export/serve/"
            f"?session_data_timestamp={timestamp}",
            timeout=serve_timeout,
        )
        serve.raise_for_status()
        return decode_export_csv(serve)

    def _post_export_prep(
        self,
        session: requests.Session,
        source_id: int,
        form: dict[str, Any],
        referer: str,
        timeout: int,
    ) -> requests.Response:
        """POST ``export_prep`` and return its response, logging each minute it is still running.

        Raises:
            requests.RequestException: If the POST fails to complete.
        """
        url = f"{self.CORALNET_URL}/source/{source_id}/annotation/export_prep/"
        outcome: Queue = Queue()

        def post() -> None:
            try:
                outcome.put(
                    session.post(
                        url,
                        headers={"Referer": referer},
                        data=form,
                        timeout=timeout,
                        allow_redirects=True,
                    )
                )
            except Exception as e:  # noqa: BLE001 - re-raised on the calling thread
                outcome.put(e)

        thread = threading.Thread(target=post, daemon=True)
        started = last_log = time.monotonic()
        thread.start()
        while thread.is_alive():
            thread.join(timeout=1.0)
            now = time.monotonic()
            if now - last_log >= _EXPORT_PREP_HEARTBEAT_S:
                logger.info(
                    "export_prep still waiting for source %s: elapsed=%.0fs timeout=%ss",
                    source_id,
                    now - started,
                    timeout,
                )
                last_log = now
        result = outcome.get_nowait()
        if isinstance(result, Exception):
            raise result
        return cast(requests.Response, result)

    def walk_source(
        self, source_id: int, bucket_name: str, s3_prefix: str, *, total_images: int
    ) -> tuple[list[ImageRef], int]:
        """Walk every browse page of a source and return its images, checked against its count.

        Progress is saved to ``_partial/walk.json`` every
        :data:`_WALK_CHECKPOINT_EVERY_PAGES` pages and when the walk ends. A saved walk
        that ended is returned without any request; an unfinished one continues at
        ``?page=<last page + 1>``.

        A walk is retried up to :data:`_WALK_ATTEMPTS` times. A page that breaks a walk rule
        (see :func:`_check_walk_page`) resumes from the last checkpoint; a page without the
        export form logs the session in again first. When the walk ends, the overview is
        read again: a changed count discards the source's ``_partial/`` and walks against the
        new count; a walk that found another number of images starts again at page 1.

        Args:
            source_id: CoralNet source id.
            bucket_name: Bucket holding the source's ``_partial/``.
            s3_prefix: Key prefix of the scrape.
            total_images: Image count of the overview when the source's download began.

        Returns:
            The images in walk order, and the overview count they match.

        Raises:
            Exception: The last attempt's error once no attempt is left or it is not
                worth repeating, e.g. :class:`WalkIncomplete` or ``requests.HTTPError``.
        """
        expected = total_images
        expired = False
        for attempt in _retrying(_WALK_ATTEMPTS):
            with attempt:
                if expired:
                    self._relogin(self.session)
                    expired = False
                try:
                    refs = self._walk_once(source_id, bucket_name, s3_prefix, expected)
                except SessionExpired:
                    expired = True
                    raise
                except _TotalChanged as e:
                    logger.warning("source %s: %s", source_id, e)
                    expected = e.total
                    raise
        return refs, expected

    def _walk_once(
        self, source_id: int, bucket_name: str, s3_prefix: str, expected: int
    ) -> list[ImageRef]:
        """Finish the source's walk from its saved progress and return its images.

        Raises:
            SessionExpired: If a browse page has no export form.
            WalkIncomplete: If a page breaks a walk rule, the overview count is no longer
                ``expected`` (``_partial/`` is discarded), or the walk found another number
                of images (the walk's progress is discarded).
        """
        key = _partial_key(s3_prefix, source_id, _WALK_FILE)
        saved = s3_io.read_json_s3(self.s3, bucket_name, key)
        progress = _WalkProgress.from_json(saved) if saved is not None else _WalkProgress()
        if progress.complete:
            return list(progress.refs.values())
        if progress.last_page:
            logger.info("source %s: walk resumes after page %d", source_id, progress.last_page)
        self._walk_pages(source_id, bucket_name, key, progress)
        self._check_walk_total(source_id, bucket_name, s3_prefix, len(progress.refs), expected)
        progress.complete = True
        s3_io.write_json_s3(self.s3, bucket_name, key, progress.to_json())
        logger.info("source %s: walk found %d images", source_id, len(progress.refs))
        return list(progress.refs.values())

    def _walk_pages(
        self, source_id: int, bucket_name: str, key: str, progress: _WalkProgress
    ) -> None:
        """Walk the browse pages after ``progress``, checkpointing it to ``key``.

        Raises:
            SessionExpired: If a browse page has no export form.
            WalkIncomplete: If a page breaks a walk rule.
        """
        url: str | None = self._browse_page_url(source_id, progress.last_page + 1)
        while url is not None:
            response = self.session.get(url, timeout=_PAGE_EXPORT_TIMEOUT)
            response.raise_for_status()
            thumbs, next_rel, has_form = _parse_browse_thumbs(response.text)
            if not has_form:
                raise SessionExpired(
                    f"browse page {progress.last_page + 1} of source {source_id} has no export form"
                )
            _check_walk_page(progress, thumbs, has_next=next_rel is not None)
            progress.add_page(thumbs)
            url = urllib.parse.urljoin(url, next_rel) if next_rel else None
            if url is not None and progress.last_page % _WALK_CHECKPOINT_EVERY_PAGES == 0:
                s3_io.write_json_s3(self.s3, bucket_name, key, progress.to_json())
            if progress.last_page % _BROWSE_PROGRESS_EVERY == 0:
                logger.info("source %s: walked %d pages", source_id, progress.last_page)

    def _check_walk_total(
        self, source_id: int, bucket_name: str, s3_prefix: str, found: int, expected: int
    ) -> None:
        """Read the overview again and check the ``found`` images against it.

        Raises:
            ContentError: If the overview lists no image count.
            _TotalChanged: If the count is no longer ``expected``; ``_partial/`` is discarded
                and a fresh state for the new count is saved.
            _WalkCountMismatch: If ``found`` differs from the count; ``walk.json`` is discarded.
        """
        total = self.fetch_total_images(source_id)
        if total is None:
            raise ContentError(f"source {source_id}: overview lists no image count")
        if total != expected:
            self._reset_partial(source_id, bucket_name, s3_prefix, total)
            raise _TotalChanged(total)
        if found != expected:
            key = _partial_key(s3_prefix, source_id, _WALK_FILE)
            self.s3.delete_object(Bucket=bucket_name, Key=key)
            raise _WalkCountMismatch(found, expected)

    def _get_thread_session(self) -> requests.Session:
        """Per-thread session with cookies copied from the main session.

        Safe for read-only requests (e.g. image URL resolution) where threads only
        consume server state. NOT safe for ``export_prep``-style endpoints that write to
        the Django server-side session keyed on ``sessionid`` -- use
        :meth:`_get_export_session` for those.
        """
        s = getattr(self._thread_local, "session", None)
        if s is None:
            s = self.sessions.coralnet()
            s.cookies.update(self.session.cookies)
            self._thread_local.session = s
        return s

    def _new_logged_in_session(self) -> requests.Session:
        """Create a fresh ``requests.Session`` and log it in independently.

        Each call produces an isolated cookie jar (own ``sessionid``), so the returned
        session corresponds to its own server-side Django session. Used by
        :meth:`_get_export_session` to give each phase-2 worker its own session and
        avoid the ``export_prep`` / ``serve`` race that happens when multiple threads
        share one Django sessionid.
        """
        session = self.sessions.coralnet()
        self._perform_login(session)
        return session

    def _get_export_session(self) -> requests.Session:
        """Return this worker thread's dedicated, independently-logged-in session.

        Lazy: the first call from a thread does one login round-trip (~1s),
        then caches the session for all subsequent ``_export_one_page`` calls
        from the same worker. Because CoralNet's ``/annotation/export_prep/``
        writes to ``request.session[f"export_{timestamp}"]`` and
        ``/export/serve/`` reads it back, sharing a sessionid across threads
        causes one thread's prep to evict another's data and the serve to
        return an HTML error page (which then fails ``pd.read_csv``).
        """
        s = getattr(self._export_thread_local, "session", None)
        if s is None:
            s = self._new_logged_in_session()
            self._export_thread_local.session = s
        return s

    def _resolve_one_image_url(self, rel: str) -> str:
        """Return the original-image URL found on the image page at the relative link ``rel``.

        Raises:
            ValueError: If the page has no original image, e.g. a private source.
        """
        session = self._get_thread_session()
        url = urllib.parse.urljoin(self.CORALNET_URL + "/", rel)
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
        with self.media.get(url, stream=True, timeout=_IMAGE_TIMEOUT) as response:
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
            logger.warning("Image %s failed (%s): %s", ref.id, stage, e)
            return stage
        return "ok"

    def download_images(
        self,
        refs: list[ImageRef],
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
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

    def _load_state(
        self, source_id: int, bucket_name: str, s3_prefix: str, total_images: int
    ) -> dict[str, Any] | None:
        """Return the source's ``_partial/state.json``.

        Returns:
            The state; None when there is none, or it was recorded for another overview
            total.
        """
        state = s3_io.read_json_s3(
            self.s3, bucket_name, _partial_key(s3_prefix, source_id, _STATE_FILE)
        )
        if state is None:
            return None
        if state["total_images"] != total_images:
            logger.warning(
                "Source %s: CoralNet lists %d images, %d when the partial download began; "
                "starting over",
                source_id,
                total_images,
                state["total_images"],
            )
            return None
        return cast("dict[str, Any]", state)

    def _reset_partial(
        self, source_id: int, bucket_name: str, s3_prefix: str, total_images: int
    ) -> dict[str, Any]:
        """Delete the source's ``_partial/``, then save and return a fresh state.

        Partial work without a usable state is discarded: a changed source cannot be stitched.
        """
        s3_io.delete_prefix(self.s3, bucket_name, s3_io.partial_prefix(s3_prefix, source_id))
        fresh: dict[str, Any] = {"total_images": total_images, "stages": []}
        self._save_state(source_id, bucket_name, s3_prefix, fresh)
        return fresh

    def _save_state(
        self, source_id: int, bucket_name: str, s3_prefix: str, state: dict[str, Any]
    ) -> None:
        """Stamp ``state`` with the current UTC time and write it to ``_partial/state.json``."""
        state["updated_at"] = datetime.now(UTC).isoformat()
        key = _partial_key(s3_prefix, source_id, _STATE_FILE)
        s3_io.write_json_s3(self.s3, bucket_name, key, state)

    def _finish_stage(
        self, source_id: int, bucket_name: str, s3_prefix: str, state: dict[str, Any], stage: str
    ) -> None:
        """Record ``stage`` as done in the source's state."""
        state["stages"].append(stage)
        self._save_state(source_id, bucket_name, s3_prefix, state)

    def _mark_complete(
        self, source_id: int, bucket_name: str, s3_prefix: str, verification: Verification
    ) -> None:
        """Write the source's ``_complete.json``, then delete its ``_partial/``."""
        marker = {
            "source_id": source_id,
            "scraper_version": _SCRAPER_VERSION,
            "completed_at": datetime.now(UTC).isoformat(),
            **verification.stats,
            "warnings": verification.warnings,
        }
        key = s3_io.object_key_complete_marker(s3_prefix, source_id)
        s3_io.write_json_s3(self.s3, bucket_name, key, marker)
        s3_io.delete_prefix(self.s3, bucket_name, s3_io.partial_prefix(s3_prefix, source_id))
        logger.info("Source %s complete: s3://%s/%s", source_id, bucket_name, key)

    def _prune_orphans(self, source_id: int, bucket_name: str, s3_prefix: str) -> int:
        """Delete the source's image objects that its image list lacks; return how many."""
        keys = orphan_image_keys(self.s3, bucket_name, s3_prefix, source_id)
        s3_io.delete_keys(self.s3, bucket_name, keys)
        if keys:
            logger.info("Source %s: pruned %d orphan image objects", source_id, len(keys))
        return len(keys)

    def fetch_total_images(self, source_id: int) -> int | None:
        """Return the image count on the source's overview page, or None when it lists none."""
        return parse_image_status_counts(self._get_source_overview_html(source_id)).total_images

    def verify_existing(self, source_id: int, bucket_name: str, s3_prefix: str) -> list[str]:
        """Verify a source already on S3 against CoralNet's live image count.

        A source that passes gets its completion marker; nothing is downloaded.

        Args:
            source_id: CoralNet source id.
            bucket_name: Bucket holding the source.
            s3_prefix: Key prefix of the scrape.

        Returns:
            The verification failure codes; empty when the source passed.
        """
        total = self.fetch_total_images(source_id)
        if total is None:
            return ["overview_total_missing"]
        verification = verify_source(
            self.s3, bucket_name, s3_prefix, source_id, expected_total=total
        )
        if verification.ok:
            self._mark_complete(source_id, bucket_name, s3_prefix, verification)
        return verification.failures

    def download_source(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
        *,
        prune: bool = False,
    ) -> DownloadResult:
        """Download one source's artifacts to S3, resuming the stages a previous run finished.

        Logs in when needed, then runs metadata, labelset, the browse walk (and
        ``image_list.csv``), the annotation export and the images, recording each finished
        stage in ``_partial/state.json``. Then the source is verified; only when every step
        succeeded and verification passes is ``_complete.json`` written and ``_partial/``
        deleted. Sources with no confirmed images or no images write nothing.

        Args:
            source_id: CoralNet source id.
            bucket_name: Destination S3 bucket.
            s3_prefix: Destination key prefix.
            prune: Delete image objects the verified image list lacks, before the marker.

        Returns:
            The outcome, with failure codes in ``errors`` and per-kind image counts in
            ``image_failures``. A failed walk or annotations export ends the download, as does
            a walk whose image count differs from the overview's
            (``walk_count_mismatch:<found>!=<total>``; the next run walks again). A failed
            verification adds ``verification_failed`` and discards ``_partial/``.
        """
        sid = source_id
        logger.info("Downloading CoralNet source %s", sid)
        if not self.logged_in and not self.login():
            return DownloadResult(False, errors=["login failed"])
        try:
            overview_html = self._get_source_overview_html(sid)
        except Exception as e:  # noqa: BLE001
            response = e.response if isinstance(e, requests.HTTPError) else None
            status = response.status_code if response is not None else None
            if status in _SOURCE_INACCESSIBLE_STATUSES:
                return DownloadResult(False, errors=[f"cannot access source {sid}: HTTP {status}"])
            return DownloadResult(
                False,
                errors=[f"overview fetch failed: {type(e).__name__}: {e}"],
            )
        accessible, reason = parse_source_access(overview_html)
        if not accessible:
            logger.error("Source %s not accessible: %s", sid, reason)
            return DownloadResult(False, errors=[f"cannot access source {sid}: {reason}"])

        counts = parse_image_status_counts(overview_html)
        if counts.confirmed == 0:
            logger.info("Source %s has zero confirmed images on website; skipping", sid)
            return DownloadResult(ok=True, skipped_no_confirmed=True)
        if counts.total_images is None:
            logger.error("Source %s: overview lists no image count", sid)
            return DownloadResult(False, errors=["overview_total_missing"])
        if counts.total_images == 0:
            logger.info("Source %s reports zero images; skipping", sid)
            return DownloadResult(ok=True, skipped_no_images=True)
        return self._download_stages(
            sid, bucket_name, s3_prefix, overview_html, counts.total_images, prune=prune
        )

    def _download_details(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        overview_html: str,
        state: dict[str, Any],
    ) -> list[str]:
        """Run the metadata and labelset stages not yet done; return their error codes."""
        errs = []
        if "metadata" not in state["stages"]:
            ok, _ = self.download_metadata(
                source_id, bucket_name, s3_prefix, overview_html=overview_html
            )
            if ok:
                self._finish_stage(source_id, bucket_name, s3_prefix, state, "metadata")
            else:
                errs.append("metadata_download_failed")
        if "labelset" not in state["stages"]:
            if self.download_labelset(source_id, bucket_name, s3_prefix):
                self._finish_stage(source_id, bucket_name, s3_prefix, state, "labelset")
            else:
                errs.append("labelset_failed")
        return errs

    def _walk_and_list(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        state: dict[str, Any],
    ) -> list[ImageRef]:
        """Return the walk's images, uploading ``image_list.csv`` unless the walk stage is done.

        A walk that ended on a changed overview count starts ``state`` afresh with that count,
        as the walk already discarded the rest of ``_partial/``.

        Raises:
            _WalkCountMismatch: If the last walk attempt found other than the overview's count
                of images; ``_partial/walk.json`` is deleted so the next run walks again.
        """
        refs, total = self.walk_source(
            source_id, bucket_name, s3_prefix, total_images=state["total_images"]
        )
        if total != state["total_images"]:
            state.update(total_images=total, stages=[])
            self._save_state(source_id, bucket_name, s3_prefix, state)
        if "walk" not in state["stages"]:
            list_key = s3_io.object_key_image_list_csv(s3_prefix, source_id)
            _upload_csv(self.s3, bucket_name, list_key, _image_list_frame(refs))
            logger.info("Image list CSV s3://%s/%s", bucket_name, list_key)
            self._finish_stage(source_id, bucket_name, s3_prefix, state, "walk")
        return refs

    def _download_stages(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        overview_html: str,
        total_images: int,
        *,
        prune: bool,
    ) -> DownloadResult:
        """Run the stages of a source with images, then verify it and write its marker.

        A walk whose last attempt finds other than the overview's count of images ends the
        download with ``walk_count_mismatch:<found>!=<total>`` before the export; the walk is
        not recorded.
        """
        sid = source_id
        state = self._load_state(sid, bucket_name, s3_prefix, total_images) or self._reset_partial(
            sid, bucket_name, s3_prefix, total_images
        )
        errs = self._download_details(sid, bucket_name, s3_prefix, overview_html, state)
        try:
            refs = self._walk_and_list(sid, bucket_name, s3_prefix, state)
        except _WalkCountMismatch as e:
            logger.error("Source %s: %s", sid, e)
            return DownloadResult(False, errors=[*errs, str(e)])
        except Exception:
            logger.exception("Source %s: browse walk failed", sid)
            return DownloadResult(False, errors=[*errs, "image_list_failed"])
        total_images = state["total_images"]
        if "export" not in state["stages"]:
            if not self.download_annotations(
                sid, bucket_name, s3_prefix, refs=refs, total_images=total_images
            ):
                return DownloadResult(False, errors=[*errs, "annotations_failed"])
            self._finish_stage(sid, bucket_name, s3_prefix, state, "export")
        image_failures = ImageFailures()
        if "images" not in state["stages"]:
            image_failures = self._download_missing_images(refs, sid, bucket_name, s3_prefix)
            errs += _image_failure_codes(image_failures)
            if image_failures == ImageFailures():
                self._finish_stage(sid, bucket_name, s3_prefix, state, "images")
        if errs:
            return DownloadResult(ok=False, errors=errs, image_failures=image_failures)
        errs += self._verify_and_mark(sid, bucket_name, s3_prefix, total_images, prune)
        return DownloadResult(ok=not errs, errors=errs, image_failures=image_failures)

    def _verify_and_mark(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        total_images: int,
        prune: bool,
    ) -> list[str]:
        """Verify the source and write its marker when it passes.

        Returns:
            ``["verification_failed"]`` after discarding ``_partial/`` when verification
            fails, else an empty list.
        """
        verification = verify_source(
            self.s3, bucket_name, s3_prefix, source_id, expected_total=total_images
        )
        if verification.ok and prune and self._prune_orphans(source_id, bucket_name, s3_prefix):
            verification = verify_source(
                self.s3, bucket_name, s3_prefix, source_id, expected_total=total_images
            )
        if not verification.ok:
            logger.error(
                "Source %s failed verification: %s", source_id, ", ".join(verification.failures)
            )
            s3_io.delete_prefix(self.s3, bucket_name, s3_io.partial_prefix(s3_prefix, source_id))
            return ["verification_failed"]
        self._mark_complete(source_id, bucket_name, s3_prefix, verification)
        return []
