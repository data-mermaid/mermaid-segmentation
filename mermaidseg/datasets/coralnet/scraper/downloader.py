"""CoralNet session client: login, probe, and S3-backed downloads."""

from __future__ import annotations

import concurrent
import io
import logging
import random
import re
import threading
import time
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, TypeVar

import boto3
import pandas as pd
import requests
from botocore.config import Config as BotoConfig
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from mermaidseg.datasets.coralnet.etl.config import get_scraper_workers
from mermaidseg.datasets.coralnet.etl.io import read_csv_s3
from mermaidseg.datasets.coralnet.scraper import s3_io
from mermaidseg.datasets.coralnet.scraper.models import DownloadResult, SourceProbe
from mermaidseg.datasets.coralnet.scraper.parsers import (
    classifier_plot_data_from_source_html,
    extract_csrf_token,
    parse_export_annotations_prep_form,
    parse_image_status_counts,
    parse_source_access,
    parse_total_images_from_source_html,
)

logger = logging.getLogger(__name__)

_PAGE_GET_TIMEOUT = 90  # simple page GETs; CoralNet can be slow under load
_EXPORT_PREP_TIMEOUT_MIN = 300
_EXPORT_PREP_TIMEOUT_MAX = 7200  # 2 hours max (was 3600)
_EXPORT_SERVE_TIMEOUT_MIN = 120
_EXPORT_SERVE_TIMEOUT_MAX = 1800
_EXPORT_PREP_RETRIES = 5  # increased from 3
_EXPORT_PREP_RETRY_SLEEP_S = 60  # doubled from 30s for longer waits between retries
_PAGINATED_EXPORT_THRESHOLD = 5000  # sources above this use per-page export
_PAGE_EXPORT_TIMEOUT = 180  # CoralNet session setup can take >60s even for 20 images
_PAGE_RETRIES = 3
_PAGE_RETRY_BACKOFF_S = 30  # sleep between retries, doubles each attempt
_PAGE_DELAY_SECONDS = 2.0  # politeness delay between page export requests
_PAGE_PARALLEL_WORKERS = 3  # concurrent export_prep/serve requests in paginated export
_CHECKPOINT_EVERY_PAGES = 50  # upload partial annotations.csv every N completed pages
_BROWSE_PROGRESS_EVERY = 10  # log phase-1 walk progress every N pages


_T = TypeVar("_T")

_IMAGE_HREF_PK = re.compile(r"/image/(\d+)(?:/|$)")


def _extract_page_image_ids(soup: BeautifulSoup) -> tuple[int, ...]:
    """Return the CoralNet image PKs (in DOM order) for the browse page in ``soup``.

    Image IDs are taken from each thumb_wrapper's ``<a href="/image/NNN/view/">``.
    Duplicates are removed while preserving first-seen order so callers can safely join
    with ``"_".join`` for CoralNet's ``image_id_list`` POST field.
    """
    seen: dict[int, None] = {}
    for wrapper in soup.find_all("span", class_="thumb_wrapper"):
        link = wrapper.find("a")
        if not link:
            continue
        match = _IMAGE_HREF_PK.search(link.get("href", ""))
        if match:
            seen.setdefault(int(match.group(1)), None)
    return tuple(seen)


@dataclass(frozen=True)
class _PageWorkItem:
    """One browse-page worth of export work discovered during phase 1.

    ``image_ids`` is the underscore-formatted CoralNet image PKs for the page (e.g.
    ``(5713071, 5713072, ...)``). The phase-2 POST sends these as
    ``image_id_list=N1_N2_...`` along with ``image_select_type=selected`` so CoralNet
    exports only those images rather than the entire source.
    """

    page_num: int
    page_url: str
    csrf: str
    images_on_page: int
    image_ids: tuple[int, ...]


def _merge_with_existing(new_df: pd.DataFrame, existing_df: pd.DataFrame | None) -> pd.DataFrame:
    """Concat ``existing_df`` (if any) with ``new_df`` and dedup on (Name, Row,
    Column)."""
    if existing_df is None:
        return new_df
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    return combined.drop_duplicates(subset=["Name", "Row", "Column"], keep="last")


def _upload_csv(s3_client: Any, bucket: str, key: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3_io.upload_csv_body(s3_client, bucket=bucket, key=key, csv_text=buf.getvalue())


# Transient network failures worth retrying on a long (multi-hour) scrape.
# ``requests.Timeout`` is the base of both ConnectTimeout and ReadTimeout;
# ChunkedEncodingError is a mid-body connection drop. 5xx responses are handled
# separately in ``_is_transient`` (4xx won't fix itself on retry).
_TRANSIENT_ERRORS = (
    requests.Timeout,
    requests.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)


def _is_transient(exc: requests.RequestException) -> bool:
    """True for errors worth retrying: timeouts, dropped connections, and 5xx."""
    if isinstance(exc, requests.HTTPError):
        resp = exc.response
        return resp is not None and resp.status_code >= 500
    return isinstance(exc, _TRANSIENT_ERRORS)


def _with_retry(
    operation: Callable[[], _T],
    *,
    source_id: int,
    page_num: int,
    kind: str,
) -> _T:
    """Retry ``operation`` on transient HTTP errors with exponential backoff.

    Non-transient errors (e.g. a 4xx) propagate immediately. Transient errors raise
    after :data:`_PAGE_RETRIES` attempts.
    """
    for attempt in range(1, _PAGE_RETRIES + 1):
        try:
            return operation()
        except requests.RequestException as e:
            if not _is_transient(e):
                raise
            if attempt >= _PAGE_RETRIES:
                logger.error(
                    "source %s page=%s %s: %s after %s attempts; giving up",
                    source_id,
                    page_num,
                    kind,
                    type(e).__name__,
                    _PAGE_RETRIES,
                )
                raise
            # Jitter the exponential backoff so concurrent shards retrying the
            # same failing site desync instead of stampeding in lockstep.
            base = _PAGE_RETRY_BACKOFF_S * (2 ** (attempt - 1))
            backoff = random.uniform(0.5 * base, 1.5 * base)
            logger.warning(
                "source %s page=%s %s: %s (attempt %s/%s); retrying in %ss",
                source_id,
                page_num,
                kind,
                type(e).__name__,
                attempt,
                _PAGE_RETRIES,
                backoff,
            )
            time.sleep(backoff)
    raise AssertionError("unreachable")  # pragma: no cover


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

    def __init__(self, username: str, password: str, *, max_workers: int | None = None):
        self.username = username
        self.password = password
        self.max_workers = max_workers if max_workers is not None else get_scraper_workers()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        # Size connection pool for concurrent image URL resolution
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=self.max_workers + 4
        )
        self.session.mount("https://", adapter)
        self.logged_in = False
        self.s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=self.max_workers + 4))

    def _source_main_url(self, source_id: int) -> str:
        return f"{self.CORALNET_URL}/source/{source_id}/"

    def _get_source_overview_html(self, source_id: int) -> str:
        url = self._source_main_url(source_id)
        response = self.session.get(url, timeout=_PAGE_GET_TIMEOUT)
        response.raise_for_status()
        return response.text

    def probe_source(self, source_id: int) -> SourceProbe:
        """HTTP GET overview page; extract accessibility + total images (no S3
        writes)."""
        url = self._source_main_url(source_id)
        try:
            html = self._get_source_overview_html(source_id)
        except Exception as e:  # noqa: BLE001
            return SourceProbe(
                source_id=source_id,
                source_url=url,
                accessible=False,
                error=f"{type(e).__name__}: {e}",
                total_images_website=0,
            )
        accessible, err = parse_source_access(html)
        if not accessible:
            return SourceProbe(
                source_id=source_id,
                source_url=url,
                accessible=False,
                error=err,
                total_images_website=0,
            )
        counts = parse_image_status_counts(html)
        tw = counts.total_images if counts.total_images is not None else 0
        return SourceProbe(
            source_id=source_id,
            source_url=url,
            accessible=True,
            error=None,
            total_images_website=tw,
            n_confirmed_website=counts.confirmed,
            n_unconfirmed_website=counts.unconfirmed,
            n_unclassified_website=counts.unclassified,
        )

    def _perform_login(self, session: requests.Session) -> bool:
        """Execute the CoralNet login flow on ``session`` (independent cookie jar).

        Pure side-effect helper: pops a fresh login CSRF, POSTs credentials, and
        leaves ``session`` with a logged-in ``sessionid`` cookie. Returns True
        on success without mutating any other state. Callers compose this into
        :meth:`login` (the main session) and :meth:`_new_logged_in_session`
        (per-worker independent sessions used by parallel export).
        """

        def _get_login_page() -> requests.Response:
            r = session.get(self.LOGIN_URL, timeout=_PAGE_GET_TIMEOUT)
            r.raise_for_status()
            return r

        # Retry the login round-trip on transient errors: CoralNet has brief
        # availability dips, and a single login timeout otherwise kills the whole
        # job before any source is scraped. Invalid-credential failures raise a
        # RuntimeError (below), which _with_retry does not retry.
        response = _with_retry(_get_login_page, source_id=0, page_num=0, kind="login")
        csrf_token = extract_csrf_token(response.text)
        if not csrf_token:
            raise RuntimeError("Could not find CSRF token")
        login_response = _with_retry(
            lambda: session.post(
                self.LOGIN_URL,
                data={
                    "username": self.username,
                    "password": self.password,
                    "csrfmiddlewaretoken": csrf_token,
                },
                headers={"Referer": self.LOGIN_URL},
                timeout=_PAGE_GET_TIMEOUT,
                allow_redirects=True,
            ),
            source_id=0,
            page_num=0,
            kind="login",
        )
        if "Sign out" not in login_response.text and login_response.url == self.LOGIN_URL:
            raise RuntimeError("Login failed - invalid credentials or other error")
        return True

    def login(self) -> bool:
        try:
            self._perform_login(self.session)
            self.logged_in = True
            logger.info("CoralNet login successful for %s", self.username)
            return True
        except Exception:
            logger.exception("CoralNet login failed for %s", self.username)
            return False

    def check_permissions(self, source_id: int) -> bool:
        try:
            html = self._get_source_overview_html(source_id)
            accessible, err = parse_source_access(html)
            if not accessible:
                logger.error("Permission/overview failed source %s: %s", source_id, err)
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.exception("Permission check failed for source %s: %s", source_id, e)
            return False

    _META_KEYS = ("x", "y", "nimages", "traintime", "date", "pk")
    _META_COLS = ("Classifier nbr", "Accuracy", "Trained on", "Date", "Traintime", "Global id")

    def download_metadata(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
        *,
        overview_html: str | None = None,
    ) -> tuple[bool, int]:
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
            csv_buffer = io.StringIO()
            meta_df.to_csv(csv_buffer, index=False)
            key = f"{s3_prefix}/s{source_id}/metadata.csv"
            s3_io.upload_csv_body(
                self.s3, bucket=bucket_name, key=key, csv_text=csv_buffer.getvalue()
            )
            logger.info("Metadata saved s3://%s/%s", bucket_name, key)
            return True, total_images
        except Exception:
            logger.exception("download_metadata failed source %s", source_id)
            return False, 0

    def download_labelset(
        self, source_id: int, bucket_name: str, s3_prefix: str = "coralnet-public-images"
    ) -> bool:
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
                    label_ids.append(link["href"].split("/")[-2])
                    names.append(link.get_text().strip())
                    short_codes.append(cells[1].get_text().strip() if len(cells) > 1 else "")
            if not label_ids:
                logger.warning("No labels found in labelset for source %s", source_id)
                return True
            labelset_df = pd.DataFrame(
                {"Label ID": label_ids, "Name": names, "Short Code": short_codes}
            )
            buf = io.StringIO()
            labelset_df.to_csv(buf, index=False)
            key = f"{s3_prefix}/s{source_id}/labelset.csv"
            s3_io.upload_csv_body(self.s3, bucket=bucket_name, key=key, csv_text=buf.getvalue())
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
        total_images_hint: int | None = None,
    ) -> bool:
        """Export and download annotations CSV; upload to S3.

        Note: Does not check for existing annotations.csv on S3. If called during remediation,
        the remediation classifier already determined fresh annotations are needed.
        """
        try:
            total_images = total_images_hint
            if total_images is None or total_images < 0:
                total_images = parse_total_images_from_source_html(
                    self._get_source_overview_html(source_id)
                )
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

            if (total_images or 0) > _PAGINATED_EXPORT_THRESHOLD:
                logger.info(
                    "Source %s: %s images (>%s threshold), using paginated export",
                    source_id,
                    total_images,
                    _PAGINATED_EXPORT_THRESHOLD,
                )
                return self.download_annotations_paginated(
                    source_id, bucket_name, s3_prefix, total_images_website=total_images
                )

            browse_url = f"{self.CORALNET_URL}/source/{source_id}/browse/images/"
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
                    ):
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
            buf = io.StringIO()
            df_annotations.to_csv(buf, index=False)
            key = f"{s3_prefix}/s{source_id}/annotations.csv"
            s3_io.upload_csv_body(self.s3, bucket=bucket_name, key=key, csv_text=buf.getvalue())
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
        total_images_website: int | None = None,
        page_workers: int | None = None,
        checkpoint_every_pages: int = _CHECKPOINT_EVERY_PAGES,
    ) -> bool:
        """Export annotations one browse-page at a time for large sources.

        Each page covers ~20 images and returns in 2-10 seconds, avoiding the
        multi-hour blocking wait of the bulk ``image_select_type=all`` export.

        Runs in two phases. Phase 1 walks the browse pages sequentially to
        discover each page's URL and CSRF token (the "Next page" link is only
        known after parsing the previous page). Phase 2 issues the
        ``export_prep`` POST and ``export/serve`` GET for each discovered page
        across ``page_workers`` threads, applying a per-worker politeness sleep
        of :data:`_PAGE_DELAY_SECONDS` between round-trips.

        Loads any existing ``annotations.csv`` from S3 before starting so a
        partial run can resume: new annotations are merged in and the combined
        result is deduplicated on ``(Name, Row, Column)`` before upload. A
        partial CSV is also uploaded every ``checkpoint_every_pages`` completed
        pages so a crash mid-export does not lose all in-memory progress.

        Args:
            page_workers: Concurrent ``export_prep``/``serve`` workers in phase
                2. Defaults to :data:`_PAGE_PARALLEL_WORKERS`. Pass ``1`` for
                deterministic ordering (used by tests).
            checkpoint_every_pages: Upload merged partial CSV after this many
                completed pages. Set very high to disable mid-run uploads.
        """
        try:
            annotations_key = s3_io.object_key_annotation_csv(s3_prefix, source_id)
            existing_df, is_complete = self._load_existing_annotations(
                bucket_name, annotations_key, source_id, total_images_website
            )
            if is_complete:
                return True

            work_items = self._walk_browse_pages(source_id)
            if not work_items:
                logger.warning("source %s: paginated export found no annotation pages", source_id)
                return False

            workers = page_workers if page_workers is not None else _PAGE_PARALLEL_WORKERS
            workers = max(1, min(workers, len(work_items)))

            chunks_by_page = self._export_pages_parallel(
                source_id=source_id,
                work_items=work_items,
                workers=workers,
                existing_df=existing_df,
                bucket_name=bucket_name,
                annotations_key=annotations_key,
                checkpoint_every_pages=checkpoint_every_pages,
            )

            ordered_chunks = [chunks_by_page[item.page_num] for item in work_items]
            combined = _merge_with_existing(
                pd.concat(ordered_chunks, ignore_index=True), existing_df
            )
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

    def _load_existing_annotations(
        self,
        bucket_name: str,
        annotations_key: str,
        source_id: int,
        total_images_website: int | None,
    ) -> tuple[pd.DataFrame | None, bool]:
        """Load an existing ``annotations.csv`` from S3 if present.

        Returns ``(df, is_complete)``. ``df`` is ``None`` when the object does not
        exist; ``is_complete`` is ``True`` only when the existing CSV's unique-image
        count already meets ``total_images_website``, signalling the caller can skip re-
        export. Non-404 ``ClientError``s propagate.
        """
        existing_df = read_csv_s3(self.s3, bucket_name, annotations_key)
        if existing_df is None:
            return None, False
        n_existing_images = existing_df["Name"].nunique()
        logger.info(
            "source %s: found existing annotations.csv with %d rows (%d unique images); will merge",
            source_id,
            len(existing_df),
            n_existing_images,
        )
        if total_images_website and n_existing_images >= total_images_website:
            logger.info(
                "source %s: existing annotations already cover all %d images, skipping re-export",
                source_id,
                total_images_website,
            )
            return existing_df, True
        return existing_df, False

    def _walk_browse_pages(self, source_id: int) -> list[_PageWorkItem]:
        """Phase 1: sequentially walk browse pages, collecting CSRF + URL per page.

        Has to be sequential because each page's "Next page" link and CSRF token are
        only known after parsing the previous page. Logs progress every
        :data:`_BROWSE_PROGRESS_EVERY` pages so long walks (50+ pages) are observable
        rather than appearing hung.
        """
        base_url = f"{self.CORALNET_URL}/source/{source_id}/browse/images/"
        logger.info("source %s: phase 1 starting browse-page walk", source_id)
        work_items: list[_PageWorkItem] = []
        current_url: str | None = base_url
        page_num = 0
        while current_url is not None:
            page_num += 1
            page_response = _with_retry(
                lambda url=current_url: self._raise_for_status(
                    self.session.get(url, timeout=_PAGE_EXPORT_TIMEOUT)
                ),
                source_id=source_id,
                page_num=page_num,
                kind="browse",
            )
            soup = BeautifulSoup(page_response.text, "html.parser")
            form = soup.find("form", {"id": "export-annotations-prep-form"})
            csrf_input = form.find("input", {"name": "csrfmiddlewaretoken"}) if form else None
            if not csrf_input:
                raise RuntimeError(
                    f"Could not find export form / CSRF on page {page_num} of source {source_id}"
                )
            next_anchor = soup.find("a", title="Next page")
            next_rel: str | None = next_anchor.get("href") if next_anchor else None
            image_ids = _extract_page_image_ids(soup)
            if not image_ids:
                raise RuntimeError(
                    f"No image IDs found on page {page_num} of source {source_id}; "
                    "browse page DOM may have changed"
                )
            work_items.append(
                _PageWorkItem(
                    page_num=page_num,
                    page_url=current_url,
                    csrf=csrf_input["value"],
                    images_on_page=len(image_ids),
                    image_ids=image_ids,
                )
            )
            current_url = urllib.parse.urljoin(current_url, next_rel) if next_rel else None
            if page_num % _BROWSE_PROGRESS_EVERY == 0:
                logger.info(
                    "source %s: phase 1 walked %d pages so far%s",
                    source_id,
                    page_num,
                    "" if current_url else " (done)",
                )
        logger.info(
            "source %s: phase 1 complete: %d browse pages discovered", source_id, len(work_items)
        )
        return work_items

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> requests.Response:
        resp.raise_for_status()
        return resp

    def _export_one_page(self, source_id: int, item: _PageWorkItem) -> pd.DataFrame:
        """Phase 2 worker: export_prep POST + serve GET for one page, with retries.

        Uses a worker-local, independently-logged-in session
        (:meth:`_get_export_session`) so concurrent workers don't trample each other's
        server-side Django session state.

        The phase-1 CSRF token on ``item`` was captured from the *main* session and is
        therefore not valid for the worker's separate session; instead the worker GETs
        the browse page once on its own session to capture a fresh CSRF before POSTing.

        Sends ``image_select_type=selected`` plus ``image_id_list`` (underscore-
        separated PKs) so CoralNet's ``image_search_kwargs_to_queryset`` filters to just
        this page's images, rather than exporting the entire source.
        ``image_form_type=search`` is the hidden, ``required=True`` discriminator on
        CoralNet's ``ImageSearchForm`` -- omitting it makes the form invalid and the
        serve response returns HTML instead of CSV. CoralNet caps ``image_id_list`` at
        100 entries; browse pages have 20.
        """
        session = self._get_export_session()
        export_url = f"{self.CORALNET_URL}/source/{source_id}/annotation/export_prep/"

        def do_round_trip() -> pd.DataFrame:
            browse_response = session.get(item.page_url, timeout=_PAGE_EXPORT_TIMEOUT)
            browse_response.raise_for_status()
            soup = BeautifulSoup(browse_response.text, "html.parser")
            form = soup.find("form", {"id": "export-annotations-prep-form"})
            csrf_input = form.find("input", {"name": "csrfmiddlewaretoken"}) if form else None
            if not csrf_input:
                raise RuntimeError(
                    f"Worker session: no CSRF on browse page {item.page_num} of source {source_id}"
                )
            form_data = {
                "csrfmiddlewaretoken": csrf_input["value"],
                "browse_action": "export_annotations",
                "image_form_type": "search",
                "image_select_type": "selected",
                "image_id_list": "_".join(str(pk) for pk in item.image_ids),
                "label_format": "both",
                "optional_columns": [
                    "annotator_info",
                    "metadata_date_aux",
                    "metadata_other",
                ],
            }
            export_response = session.post(
                export_url,
                headers={"Referer": item.page_url},
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
            return pd.read_csv(io.StringIO(serve_response.text))

        chunk_df = _with_retry(
            do_round_trip, source_id=source_id, page_num=item.page_num, kind="export"
        )
        time.sleep(_PAGE_DELAY_SECONDS)
        return chunk_df

    def _export_pages_parallel(
        self,
        *,
        source_id: int,
        work_items: list[_PageWorkItem],
        workers: int,
        existing_df: pd.DataFrame | None,
        bucket_name: str,
        annotations_key: str,
        checkpoint_every_pages: int,
    ) -> dict[int, pd.DataFrame]:
        """Phase 2: parallel export_prep + serve, with periodic S3 checkpoints.

        Checkpoint partials are concatenated incrementally (O(N) per checkpoint, not
        O(N²)) and use completion order rather than page order; the final upload in
        :meth:`download_annotations_paginated` rewrites the object in page order.
        """
        chunks_by_page: dict[int, pd.DataFrame] = {}
        partial_df: pd.DataFrame | None = None
        pending_chunks: list[pd.DataFrame] = []
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(self._export_one_page, source_id, item): item for item in work_items
            }
            for fut in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[fut]
                chunk_df = fut.result()
                chunks_by_page[item.page_num] = chunk_df
                pending_chunks.append(chunk_df)
                completed += 1
                logger.info(
                    "source %s page=%d images=%d annotations=%d completed=%d/%d",
                    source_id,
                    item.page_num,
                    item.images_on_page,
                    len(chunk_df),
                    completed,
                    len(work_items),
                )
                if checkpoint_every_pages > 0 and completed % checkpoint_every_pages == 0:
                    new_batch = pd.concat(pending_chunks, ignore_index=True)
                    partial_df = (
                        new_batch
                        if partial_df is None
                        else pd.concat([partial_df, new_batch], ignore_index=True)
                    )
                    pending_chunks = []
                    _upload_csv(
                        self.s3,
                        bucket_name,
                        annotations_key,
                        _merge_with_existing(partial_df, existing_df),
                    )
                    logger.info(
                        "source %s: checkpoint at %d/%d pages (%d new rows)",
                        source_id,
                        completed,
                        len(work_items),
                        len(partial_df),
                    )
        return chunks_by_page

    def get_images_on_page(self, browse_url: str) -> tuple[dict[str, str], str | None]:
        images: dict[str, str] = {}
        response = self.session.get(browse_url, timeout=_PAGE_GET_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for wrapper in soup.find_all("span", class_="thumb_wrapper"):
            link = wrapper.find("a")
            img = wrapper.find("img")
            if link and img:
                name = img.get("title", "")
                href = link.get("href", "")
                if name and href:
                    images[name] = href
        next_page_element = soup.find("a", title="Next page")
        next_rel = next_page_element.get("href") if next_page_element else None
        return images, next_rel

    def get_images(
        self,
        source_id: int,
        *,
        annotation_status: str | None = None,
        total_images_hint: int | None = None,
        browse_workers: int = 1,
    ) -> tuple[pd.DataFrame | None, bool]:
        """Walk CoralNet browse pages and return a DataFrame of all images.

        When ``total_images_hint`` is provided and ``browse_workers > 1``, pages
        2..N are fetched in parallel via ``?page=N`` URLs rather than sequentially
        following next-link cursors. Page 1 is always fetched first to establish
        the actual page size; subsequent pages are constructed as
        ``base_url?page=N`` (or ``base_url&page=N`` when annotation_status is set).

        Falls back to sequential cursor-following when ``total_images_hint`` is
        absent or ``browse_workers == 1``.
        """
        base_url = f"{self.CORALNET_URL}/source/{source_id}/browse/images"
        if annotation_status:
            base_url = f"{base_url}/?annotation_status={annotation_status}"

        def fetch(url: str, page_num: int) -> tuple[dict[str, str], str | None]:
            return _with_retry(
                lambda: self.get_images_on_page(url),
                source_id=source_id,
                page_num=page_num,
                kind="browse",
            )

        p_bar = tqdm(desc="Fetching images", unit="page")
        try:
            page1_imgs, next_rel = fetch(base_url, 1)
            p_bar.update(1)

            # Ordered list of per-page dicts; page 1 is always first.
            ordered_pages: list[dict[str, str]] = [page1_imgs]

            can_parallel = browse_workers > 1 and total_images_hint is not None
            page1_size = len(page1_imgs)

            if can_parallel and page1_size > 0 and total_images_hint > page1_size:
                # Ceil-divide without importing math.
                n_pages = -(-total_images_hint // page1_size)
                sep = "&" if "?" in base_url else "?"

                workers = min(browse_workers, n_pages - 1)
                page_results: dict[int, dict[str, str]] = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {
                        ex.submit(fetch, f"{base_url}{sep}page={n}", n): n
                        for n in range(2, n_pages + 1)
                    }
                    for fut in concurrent.futures.as_completed(futs):
                        n = futs[fut]
                        try:
                            page_results[n], _ = fut.result()
                        except requests.RequestException:
                            # One dead page must not abort the whole source; leave it
                            # out and let the caller's coverage guard reject a short list.
                            logger.warning(
                                "source %s page=%s failed after retries; skipping", source_id, n
                            )
                        p_bar.update(1)
                missing = (n_pages - 1) - len(page_results)
                if missing:
                    logger.warning(
                        "source %s: %d/%d browse pages failed; result is partial",
                        source_id,
                        missing,
                        n_pages - 1,
                    )
                for n in range(2, n_pages + 1):
                    ordered_pages.append(page_results.get(n, {}))
            else:
                current_url: str | None = (
                    urllib.parse.urljoin(base_url, next_rel) if next_rel else None
                )
                seq_page = 2
                while current_url is not None:
                    imgs, next_rel = fetch(current_url, seq_page)
                    ordered_pages.append(imgs)
                    p_bar.update(1)
                    seq_page += 1
                    current_url = urllib.parse.urljoin(current_url, next_rel) if next_rel else None

            # Merge in page order; first-seen wins for any duplicate name.
            merged: dict[str, str] = {}
            for page_imgs in ordered_pages:
                for name, href in page_imgs.items():
                    merged.setdefault(name, href)

            df = pd.DataFrame(list(merged.items()), columns=["Name", "Image Page"])
            return df, True
        except Exception:
            logger.exception("get_images failed source %s", source_id)
            return None, False
        finally:
            p_bar.close()

    _thread_local = threading.local()
    _export_thread_local = threading.local()

    def _get_thread_session(self) -> requests.Session:
        """Per-thread session with cookies copied from the main session.

        Safe for read-only requests (e.g. image URL resolution) where threads only
        consume server state. NOT safe for ``export_prep``-style endpoints that write to
        the Django server-side session keyed on ``sessionid`` -- use
        :meth:`_get_export_session` for those.
        """
        s = getattr(self._thread_local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(dict(self.session.headers))
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
        session = requests.Session()
        session.headers.update(dict(self.session.headers))
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
        return originals[0].attrs.get("src")

    def get_image_urls(self, image_page_urls: list[str]) -> list[str | None]:
        n = len(image_page_urls)
        workers = min(self.max_workers, n) if n > 0 else 1
        if workers <= 1:
            return [self._resolve_one_image_url(rel) for rel in image_page_urls]

        logger.info("Resolving %d image URLs with %d workers", n, workers)
        results: list[str | None] = [None] * n
        errors = 0
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

    @staticmethod
    def download_image(url: str, path: str, timeout: int = 30) -> tuple[str, bool]:
        dest = Path(path)
        if dest.exists():
            return path, True
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if dest.exists() and dest.stat().st_size > 0:
                return path, True
            return path, False
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to download local file %s: %s", url, e)
            return path, False

    def download_image_to_s3(
        self, url: str, bucket_name: str, s3_key: str, timeout: int = 30
    ) -> tuple[str, bool]:
        try:
            try:
                self.s3.head_object(Bucket=bucket_name, Key=s3_key)
                return s3_key, True
            except self.s3.exceptions.ClientError:
                pass
            response = requests.get(url, timeout=timeout, stream=True)
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
        images_df: pd.DataFrame,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
    ) -> None:
        chunk = images_df[images_df["Image URL"].notna()]
        if chunk.empty:
            logger.warning("No valid Image URL rows for source %s", source_id)
            return
        successful = 0
        futures: list[Any] = []
        workers = self.max_workers
        logger.info(
            "Image upload for source %s: %d URLs, %d workers", source_id, len(chunk), workers
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _, row in chunk.iterrows():
                name = row["Image Page"].replace("/image/", "").replace("/view/", "")
                url = row["Image URL"]
                clean_name = f"{name}.jpg"
                sk = f"{s3_prefix}/s{source_id}/images/{clean_name}"
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

    @staticmethod
    def _local_image_path(dest_dir: Path, image_page: str) -> Path:
        name = image_page.replace("/image/", "").replace("/view/", "")
        return dest_dir / f"{name}.jpg"

    def download_images_local(
        self,
        images_df: pd.DataFrame,
        dest_dir: str | Path,
    ) -> tuple[int, int]:
        """Write JPEGs under ``dest_dir``; returns ``(successful, total)``."""
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        chunk = images_df[images_df["Image URL"].notna()]
        if chunk.empty:
            logger.warning("No valid Image URL rows for local download")
            return 0, 0
        successful = 0
        futures: list[Any] = []
        workers = self.max_workers
        logger.info("Local image download: %d URLs, %d workers", len(chunk), workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _, row in chunk.iterrows():
                url = row["Image URL"]
                path = self._local_image_path(out, row["Image Page"])
                futures.append(executor.submit(self.download_image, url, str(path)))
            total = len(futures)
            for i, fu in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    _path, ok = fu.result()
                    if ok:
                        successful += 1
                except Exception as e:  # noqa: BLE001
                    logger.error("local image worker error: %s", e)
                if i % 50 == 0 or i == total:
                    logger.info("Local image download progress: %s/%s", i, total)
        logger.info("Downloaded %s/%s images to %s", successful, total, out)
        return successful, total

    def download_source_images_local(
        self,
        source_id: int,
        dest_dir: str | Path,
        *,
        confirmed_only: bool = True,
        save_image_list_csv: bool = True,
    ) -> tuple[int, int, Path]:
        """Browse CoralNet, resolve original JPEG URLs, and save under ``dest_dir``."""
        if not self.logged_in and not self.check_permissions(source_id):  # noqa: SIM102
            if not self.login() or not self.check_permissions(source_id):
                raise RuntimeError(f"cannot access CoralNet source {source_id}")
        dest = Path(dest_dir)
        images_dir = dest / "images"
        status = "confirmed" if confirmed_only else None
        images_df, ok = self.get_images(source_id, annotation_status=status)
        if not ok or images_df is None or images_df.empty:
            raise RuntimeError(f"no browse images for source {source_id}")

        images_df = images_df.reset_index(drop=True).copy()
        images_df["Image URL"] = self.get_image_urls(images_df["Image Page"].tolist())
        if save_image_list_csv:
            list_path = dest / "image_list.csv"
            list_path.parent.mkdir(parents=True, exist_ok=True)
            images_df.to_csv(list_path, index=False)
            logger.info("Wrote %s", list_path)

        n_ok, n_total = self.download_images_local(images_df, images_dir)
        return n_ok, n_total, images_dir

    def download_source(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
        *,
        download_metadata: bool = True,
        download_labelset: bool = True,
        download_annotations: bool = True,
        download_images: bool = True,
        clean_prefix_before: bool = False,
    ) -> DownloadResult:
        errs: list[str] = []
        sid = source_id
        logger.info("Downloading CoralNet source %s", sid)
        if not self.logged_in and not self.login():
            return DownloadResult(False, skipped_empty=False, errors=["login failed"])
        if clean_prefix_before:
            n_del = s3_io.delete_source_prefix(self.s3, bucket_name, s3_prefix, sid)
            logger.info("Removed %s S3 keys under s%s/", n_del, sid)
        if not self.check_permissions(sid):
            return DownloadResult(
                False, skipped_empty=False, errors=[f"cannot access source {sid}"]
            )
        try:
            overview_html = self._get_source_overview_html(sid)
        except Exception as e:  # noqa: BLE001
            return DownloadResult(
                False,
                skipped_empty=False,
                errors=[f"overview fetch failed: {type(e).__name__}: {e}"],
            )

        counts = parse_image_status_counts(overview_html)
        if counts.confirmed is not None and counts.confirmed == 0:
            logger.info(
                "Source %s has zero confirmed images on website; skipping download",
                sid,
            )
            return DownloadResult(ok=True, skipped_no_confirmed=True, errors=errs)

        total_images_hint = counts.total_images if counts.total_images is not None else -1
        if total_images_hint == 0:
            logger.info("Source %s reports zero images; skipping remainder", sid)
            return DownloadResult(ok=True, skipped_empty=True, errors=errs)

        if download_metadata:
            metadata_ok, total_images_hint = self.download_metadata(
                sid,
                bucket_name=bucket_name,
                s3_prefix=s3_prefix,
                overview_html=overview_html,
            )
            if not metadata_ok:
                errs.append("metadata_download_failed")

        if download_labelset and not self.download_labelset(
            sid, bucket_name=bucket_name, s3_prefix=s3_prefix
        ):
            errs.append("labelset_failed")

        if download_annotations and not self.download_annotations(
            sid,
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            total_images_hint=total_images_hint if total_images_hint > 0 else None,
        ):
            errs.append("annotations_failed")
            return DownloadResult(ok=False, skipped_empty=False, errors=errs)

        if download_images:
            images_df, images_success = self.get_images(sid)
            batch_size = 1000
            if images_success and images_df is not None and len(images_df) > 0:
                images_df = images_df.reset_index(drop=True).copy()
                images_df["Image URL"] = pd.NA
                buf = io.StringIO()
                images_df.to_csv(buf, index=False)
                list_key = f"{s3_prefix}/s{sid}/image_list.csv"
                s3_io.upload_csv_body(
                    self.s3,
                    bucket=bucket_name,
                    key=list_key,
                    csv_text=buf.getvalue(),
                )
                logger.info("Image list CSV s3://%s/%s", bucket_name, list_key)

                url_col_idx = images_df.columns.get_loc("Image URL")
                for start, end in half_open_batches(len(images_df), batch_size):
                    urls = images_df.iloc[start:end]["Image Page"].tolist()
                    image_urls = self.get_image_urls(urls)
                    images_df.iloc[start:end, url_col_idx] = image_urls

                    subset = images_df.iloc[start:end].copy()
                    self.download_images(subset, sid, bucket_name=bucket_name, s3_prefix=s3_prefix)
            else:
                errs.append("image_list_failed")
                logger.warning("No browse images table for source %s", sid)

        return DownloadResult(ok=not errs, skipped_empty=False, errors=errs)

    def cleanup(self) -> None:
        if self.session:
            self.session.close()
        self.logged_in = False
