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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from queue import Queue
from typing import Any, cast

import boto3
import pandas as pd
import requests
from botocore.config import Config as BotoConfig
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

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
from coralnet_scraper.verify import Verification, orphan_image_keys, verify_source

logger = logging.getLogger(__name__)

_SOURCE_INACCESSIBLE_STATUSES = frozenset({403, 404})  # missing or forbidden source overview
_PAGE_GET_TIMEOUT = 90  # simple page GETs; CoralNet can be slow under load
_EXPORT_PREP_TIMEOUT_MIN = 300
_EXPORT_PREP_TIMEOUT_MAX = 7200  # 2 hours
_EXPORT_SERVE_TIMEOUT_MIN = 120
_EXPORT_SERVE_TIMEOUT_MAX = 1800
_EXPORT_PREP_RETRIES = 5
_EXPORT_PREP_RETRY_SLEEP_S = 60  # base sleep between export_prep retries; scaled by attempt
_PAGINATED_EXPORT_THRESHOLD = 5000  # sources above this use per-page export
_PAGE_EXPORT_TIMEOUT = 180  # CoralNet session setup can take >60s even for 20 images
_PAGE_DELAY_SECONDS = 2.0  # politeness delay between page export requests
_WALK_CHECKPOINT_EVERY_PAGES = 50  # save the walk's progress to _partial/walk.json every N pages
_BROWSE_PROGRESS_EVERY = 10  # log walk progress every N pages
_IMAGE_BATCH_SIZE = 1000  # images whose URLs are resolved, then uploaded, per batch
_STATE_FILE = "state.json"
_WALK_FILE = "walk.json"
_SCRAPER_VERSION = importlib.metadata.version("coralnet-scraper")

_IMAGE_HREF_PK = re.compile(r"/image/(\d+)(?:/|$)")
_CHUNK_PAGE = re.compile(r"/page-(\d+)\.csv$")


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


class _WalkCountMismatch(Exception):
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


def _parse_browse_thumbs(html: str) -> tuple[list[tuple[int, str]], str | None]:
    """Return a browse page's thumbnails as ``(image id, title)`` and its "Next page" link.

    A thumbnail without an ``/image/<id>/`` link is skipped; one without a title gets ``""``.
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
    return thumbs, cast("str | None", next_anchor.get("href")) if next_anchor else None


def _image_ids_by_page(refs: list[ImageRef]) -> dict[int, tuple[int, ...]]:
    """Return the image ids of each browse page, pages ascending, ids in walk order."""
    pages: dict[int, list[int]] = {}
    for ref in refs:
        pages.setdefault(ref.page, []).append(ref.id)
    return {page: tuple(pages[page]) for page in sorted(pages)}


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
    return codes


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


def half_open_batches(n_rows: int, batch_size: int) -> list[tuple[int, int]]:
    """Return disjoint [start, end) index ranges covering ``range(n_rows)``."""
    if n_rows < 0 or batch_size < 1:
        raise ValueError("n_rows must be >= 0 and batch_size >= 1")
    return [(s, min(s + batch_size, n_rows)) for s in range(0, n_rows, batch_size)]


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
        if "Sign out" not in login_response.text and login_response.url == self.LOGIN_URL:
            raise RuntimeError("Login failed - invalid credentials or other error")

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
            True on success; False on failure, after logging the error. The exported body
            must parse as a CSV with a ``Name`` column; an empty body, an HTML error page, or
            any other export without one fails and uploads nothing.
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

            browse_url = self._browse_page_url(source_id, 1)
            response = self.session.get(browse_url, timeout=_PAGE_GET_TIMEOUT)
            response.raise_for_status()
            csrf = parse_export_annotations_prep_form(response.text)
            if not csrf:
                raise RuntimeError("Could not find export annotations form / CSRF")
            form_data = {
                "csrfmiddlewaretoken": csrf,
                "browse_action": "export_annotations",
                "image_select_type": "all",
                "label_format": "both",
                "optional_columns": [
                    "annotator_info",
                    "metadata_date_aux",
                    "metadata_other",
                ],
            }
            export_request_url = f"{self.CORALNET_URL}/source/{source_id}/annotation/export_prep/"
            export_response = None
            for attempt in range(1, _EXPORT_PREP_RETRIES + 1):
                attempt_timeout = min(_EXPORT_PREP_TIMEOUT_MAX, prep_timeout + (attempt - 1) * 300)
                try:
                    # Background thread for the long-running POST; the main thread emits
                    # periodic heartbeat logs so we can see progress in the run output.
                    result_queue: Queue = Queue()
                    exception_queue: Queue = Queue()

                    def do_export_prep(
                        attempt_timeout: int = attempt_timeout,
                        result_queue: Queue = result_queue,
                        exception_queue: Queue = exception_queue,
                    ) -> None:
                        try:
                            resp = self.session.post(
                                export_request_url,
                                headers={"Referer": browse_url},
                                data=form_data,
                                timeout=attempt_timeout,
                                allow_redirects=True,
                            )
                            result_queue.put(resp)
                        except Exception as e:
                            exception_queue.put(e)

                    thread = threading.Thread(target=do_export_prep, daemon=True)
                    t_start = time.monotonic()
                    thread.start()

                    heartbeat_interval = 60
                    last_heartbeat = t_start

                    while thread.is_alive():
                        thread.join(timeout=1.0)
                        elapsed = time.monotonic() - t_start

                        if elapsed - last_heartbeat >= heartbeat_interval:
                            logger.info(
                                "export_prep still waiting for source %s: elapsed=%.0fs attempt=%s/%s timeout=%ss",
                                source_id,
                                elapsed,
                                attempt,
                                _EXPORT_PREP_RETRIES,
                                attempt_timeout,
                            )
                            last_heartbeat = elapsed

                    # Check results
                    if not exception_queue.empty():
                        raise exception_queue.get()
                    if not result_queue.empty():
                        export_response = result_queue.get()
                        export_response.raise_for_status()
                        break
                    raise RuntimeError("export_prep thread completed without result")

                except requests.exceptions.Timeout:
                    if attempt >= _EXPORT_PREP_RETRIES:
                        raise
                    logger.warning(
                        "export_prep timed out for source %s (attempt %s/%s, timeout=%ss); retrying",
                        source_id,
                        attempt,
                        _EXPORT_PREP_RETRIES,
                        attempt_timeout,
                    )
                    time.sleep(_EXPORT_PREP_RETRY_SLEEP_S * attempt)
            if export_response is None:
                raise RuntimeError("export_prep produced no response")

            export_timestamp = export_response.json()["session_data_timestamp"]
            serve_url = (
                f"{self.CORALNET_URL}/source/{source_id}/export/serve/"
                f"?session_data_timestamp={export_timestamp}"
            )
            dl = self.session.get(serve_url, timeout=serve_timeout)
            dl.raise_for_status()
            df_annotations = pd.read_csv(io.StringIO(dl.text))
            if "Name" not in df_annotations.columns:
                raise RuntimeError(
                    f"Source {source_id}: export/serve response is not an annotations CSV "
                    "(no 'Name' column)"
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
            pages = _image_ids_by_page(refs)
            saved = self._saved_chunk_pages(source_id, bucket_name, s3_prefix)
            todo = {page: ids for page, ids in pages.items() if page not in saved}
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
        pages: dict[int, tuple[int, ...]],
        workers: int,
    ) -> None:
        """Export and save each page of ``pages`` (page number to image ids) across workers.

        Raises:
            Exception: The first page failure, once every started page has finished; pages
                still queued never start.
        """
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_page = {
                executor.submit(
                    self._export_and_save_page, source_id, bucket_name, s3_prefix, page, ids
                ): page
                for page, ids in pages.items()
            }
            for completed, fut in enumerate(concurrent.futures.as_completed(future_to_page), 1):
                page = future_to_page[fut]
                try:
                    rows = fut.result()
                except Exception:
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
                logger.info(
                    "source %s page=%d images=%d annotations=%d completed=%d/%d",
                    source_id,
                    page,
                    len(pages[page]),
                    rows,
                    completed,
                    len(pages),
                )

    def _export_and_save_page(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        page_num: int,
        image_ids: tuple[int, ...],
    ) -> int:
        """Export one page and save it as ``_partial/page-<n>.csv``; return its row count."""
        chunk = self._export_one_page(source_id, page_num, image_ids)
        key = _partial_key(s3_prefix, source_id, f"page-{page_num}.csv")
        _upload_csv(self.s3, bucket_name, key, chunk)
        return len(chunk)

    def _browse_page_url(self, source_id: int, page_num: int) -> str:
        """Return the URL of a source's browse page ``page_num``, counted from 1."""
        base = f"{self.CORALNET_URL}/source/{source_id}/browse/images/"
        return base if page_num == 1 else f"{base}?page={page_num}"

    def _export_one_page(
        self, source_id: int, page_num: int, image_ids: tuple[int, ...]
    ) -> pd.DataFrame:
        """Export the annotations of ``image_ids``, the images on browse page ``page_num``.

        Uses a worker-local, independently-logged-in session
        (:meth:`_get_export_session`) so concurrent workers don't trample each other's
        server-side Django session state. The worker GETs the browse page on its own
        session to capture a CSRF token bound to that session before POSTing.

        Sends ``image_select_type=selected`` plus ``image_id_list`` (underscore-
        separated PKs) so CoralNet's ``image_search_kwargs_to_queryset`` filters to just
        this page's images, rather than exporting the entire source.
        ``image_form_type=search`` is the hidden, ``required=True`` discriminator on
        CoralNet's ``ImageSearchForm`` -- omitting it makes the form invalid and the
        serve response returns HTML instead of CSV. CoralNet caps ``image_id_list`` at
        100 entries; browse pages have 20.
        """
        session = self._get_export_session()
        page_url = self._browse_page_url(source_id, page_num)
        browse_response = session.get(page_url, timeout=_PAGE_EXPORT_TIMEOUT)
        browse_response.raise_for_status()
        csrf = parse_export_annotations_prep_form(browse_response.text)
        if not csrf:
            raise RuntimeError(
                f"Worker session: no CSRF on browse page {page_num} of source {source_id}"
            )
        form_data = {
            "csrfmiddlewaretoken": csrf,
            "browse_action": "export_annotations",
            "image_form_type": "search",
            "image_select_type": "selected",
            "image_id_list": "_".join(str(pk) for pk in image_ids),
            "label_format": "both",
            "optional_columns": [
                "annotator_info",
                "metadata_date_aux",
                "metadata_other",
            ],
        }
        export_response = session.post(
            f"{self.CORALNET_URL}/source/{source_id}/annotation/export_prep/",
            headers={"Referer": page_url},
            data=form_data,
            timeout=_PAGE_EXPORT_TIMEOUT,
            allow_redirects=True,
        )
        export_response.raise_for_status()
        export_timestamp = export_response.json()["session_data_timestamp"]
        serve_response = session.get(
            f"{self.CORALNET_URL}/source/{source_id}/export/serve/"
            f"?session_data_timestamp={export_timestamp}",
            timeout=_PAGE_EXPORT_TIMEOUT,
        )
        serve_response.raise_for_status()
        chunk_df = pd.read_csv(io.StringIO(serve_response.text))
        time.sleep(_PAGE_DELAY_SECONDS)
        return chunk_df

    def walk_source(self, source_id: int, bucket_name: str, s3_prefix: str) -> list[ImageRef]:
        """Walk every browse page of a source and return its images, keyed by image id.

        Progress is saved to ``_partial/walk.json`` every
        :data:`_WALK_CHECKPOINT_EVERY_PAGES` pages and when the walk ends. A saved walk
        that ended is returned without any request; an unfinished one continues at
        ``?page=<last page + 1>``.

        Args:
            source_id: CoralNet source id.
            bucket_name: Bucket holding the source's ``_partial/``.
            s3_prefix: Key prefix of the scrape.

        Returns:
            The images in walk order; an id seen on two pages is listed once, on the first.

        Raises:
            requests.HTTPError: If a browse page returns an error status.
            RuntimeError: If a browse page lists no images.
        """
        key = _partial_key(s3_prefix, source_id, _WALK_FILE)
        saved = s3_io.read_json_s3(self.s3, bucket_name, key)
        progress = _WalkProgress.from_json(saved) if saved is not None else _WalkProgress()
        if progress.complete:
            return list(progress.refs.values())
        if progress.last_page:
            logger.info("source %s: walk resumes after page %d", source_id, progress.last_page)
        url: str | None = self._browse_page_url(source_id, progress.last_page + 1)
        while url is not None:
            response = self.session.get(url, timeout=_PAGE_EXPORT_TIMEOUT)
            response.raise_for_status()
            thumbs, next_rel = _parse_browse_thumbs(response.text)
            if not thumbs:
                raise RuntimeError(
                    f"No image IDs found on page {progress.last_page + 1} of source "
                    f"{source_id}; browse page DOM may have changed"
                )
            progress.add_page(thumbs)
            url = urllib.parse.urljoin(url, next_rel) if next_rel else None
            if url is not None and progress.last_page % _WALK_CHECKPOINT_EVERY_PAGES == 0:
                s3_io.write_json_s3(self.s3, bucket_name, key, progress.to_json())
            if progress.last_page % _BROWSE_PROGRESS_EVERY == 0:
                logger.info("source %s: walked %d pages", source_id, progress.last_page)
        progress.complete = True
        s3_io.write_json_s3(self.s3, bucket_name, key, progress.to_json())
        logger.info("source %s: walk found %d images", source_id, len(progress.refs))
        return list(progress.refs.values())

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

    def _resolve_one_image_url(self, rel: str) -> str | None:
        """Return the original-image URL found on the image page at the relative link ``rel``.

        Raises:
            ValueError: If the page has no original-image container, e.g. a private source.
        """
        session = self._get_thread_session()
        url = urllib.parse.urljoin(self.CORALNET_URL + "/", rel)
        r = session.get(url, timeout=_PAGE_GET_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        originals = soup.select("div#original_image_container > img")
        if not originals:
            raise ValueError(
                f"CoralNet image {url}: couldn't find original image container (private source?)."
            )
        return cast("str | None", originals[0].attrs.get("src"))

    def get_image_urls(self, image_page_urls: list[str]) -> list[str | None]:
        """Resolve image-page links to original-image URLs, in input order.

        A failed lookup always yields None for that entry, with one worker or several.
        """
        n = len(image_page_urls)
        workers = min(self.max_workers, n) if n > 0 else 1
        results: list[str | None] = [None] * n
        errors = 0
        if workers <= 1:
            for i, rel in enumerate(image_page_urls):
                try:
                    results[i] = self._resolve_one_image_url(rel)
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    logger.warning("Failed to resolve image URL %s: %s", rel, e)
            if errors:
                logger.warning("Image URL resolution: %d/%d failed", errors, n)
            return results

        logger.info("Resolving %d image URLs with %d workers", n, workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self._resolve_one_image_url, rel): i
                for i, rel in enumerate(image_page_urls)
            }
            for done in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=n,
                desc="Fetching image URLs",
                unit="image",
            ):
                idx = future_to_idx[done]
                try:
                    results[idx] = done.result()
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    logger.warning("Failed to resolve image URL %s: %s", image_page_urls[idx], e)
        if errors:
            logger.warning("Image URL resolution: %d/%d failed", errors, n)
        return results

    def download_image_to_s3(
        self, url: str, bucket_name: str, s3_key: str, timeout: int = 30
    ) -> tuple[str, bool]:
        """Stream the image at ``url`` to ``s3_key`` unless a non-empty object is already there.

        Returns:
            ``(s3_key, ok)``; ``ok`` is False when the download or upload failed.
        """
        try:
            if s3_io.object_size(self.s3, bucket_name, s3_key):
                return s3_key, True
            response = self.media.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            self.s3.upload_fileobj(
                response.raw, bucket_name, s3_key, ExtraArgs={"ContentType": "image/jpeg"}
            )
            return s3_key, True
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to download image to S3 %s: %s", url, e)
            return s3_key, False

    def download_images(
        self,
        images: list[tuple[ImageRef, str | None]],
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
    ) -> ImageFailures:
        """Upload every image of ``images`` that has a URL to S3 in parallel.

        Args:
            images: Each image with its original-image URL, or None when it has none.
            source_id: CoralNet source id.
            bucket_name: Destination S3 bucket.
            s3_prefix: Destination key prefix.

        Returns:
            Failure counts: images with no URL as ``missing_url``; uploads that reported
            failure or whose worker raised as ``upload_failed``.
        """
        chunk = [(ref, url) for ref, url in images if url is not None]
        missing_url = len(images) - len(chunk)
        if not chunk:
            logger.warning("No valid Image URL rows for source %s", source_id)
            return ImageFailures(missing_url=missing_url)
        successful = 0
        futures: list[Any] = []
        workers = self.max_workers
        logger.info(
            "Image upload for source %s: %d URLs, %d workers", source_id, len(chunk), workers
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ref, url in chunk:
                sk = s3_io.object_key_image(s3_prefix, source_id, str(ref.id))
                futures.append(executor.submit(self.download_image_to_s3, url, bucket_name, sk))
            total = len(futures)
            for i, fu in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    _key, ok = fu.result()
                    if ok:
                        successful += 1
                except Exception as e:  # noqa: BLE001
                    logger.error("image worker error: %s", e)
                if i % 50 == 0 or i == total:
                    logger.info(
                        "Image upload progress source %s: %s/%s",
                        source_id,
                        i,
                        total,
                    )
        logger.info(
            "Uploaded %s/%s images for source %s",
            successful,
            len(futures),
            source_id,
        )
        return ImageFailures(missing_url=missing_url, upload_failed=len(futures) - successful)

    def _download_missing_images(
        self, refs: list[ImageRef], source_id: int, bucket_name: str, s3_prefix: str
    ) -> ImageFailures:
        """Upload every image of ``refs`` that has no non-empty object on S3 yet.

        Image URLs are resolved, then uploaded, in batches of :data:`_IMAGE_BATCH_SIZE`.

        Returns:
            The failure counts summed over every batch.
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
        failures = ImageFailures()
        for start, end in half_open_batches(len(todo), _IMAGE_BATCH_SIZE):
            batch_refs = todo[start:end]
            urls = self.get_image_urls([r.view_path for r in batch_refs])
            batch = self.download_images(
                list(zip(batch_refs, urls, strict=True)), source_id, bucket_name, s3_prefix
            )
            failures = ImageFailures(
                missing_url=failures.missing_url + batch.missing_url,
                upload_failed=failures.upload_failed + batch.upload_failed,
            )
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
        total_images: int,
    ) -> list[ImageRef]:
        """Return the walk's images, uploading ``image_list.csv`` unless the walk stage is done.

        Raises:
            _WalkCountMismatch: If the walk found other than ``total_images`` images, after
                deleting ``_partial/walk.json`` so the next run walks again.
        """
        refs = self.walk_source(source_id, bucket_name, s3_prefix)
        if len(refs) != total_images:
            walk_key = _partial_key(s3_prefix, source_id, _WALK_FILE)
            self.s3.delete_object(Bucket=bucket_name, Key=walk_key)
            raise _WalkCountMismatch(len(refs), total_images)
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

        A walk that finds other than ``total_images`` images ends the download with
        ``walk_count_mismatch:<found>!=<total>`` before the export; the walk is not recorded.
        """
        sid = source_id
        state = self._load_state(sid, bucket_name, s3_prefix, total_images) or self._reset_partial(
            sid, bucket_name, s3_prefix, total_images
        )
        errs = self._download_details(sid, bucket_name, s3_prefix, overview_html, state)
        try:
            refs = self._walk_and_list(sid, bucket_name, s3_prefix, state, total_images)
        except _WalkCountMismatch as e:
            logger.error("Source %s: %s", sid, e)
            return DownloadResult(False, errors=[*errs, str(e)])
        except Exception:
            logger.exception("Source %s: browse walk failed", sid)
            return DownloadResult(False, errors=[*errs, "image_list_failed"])
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
