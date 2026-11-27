"""The annotation export: bulk for small sources, page by page for large ones."""

from __future__ import annotations

import concurrent
import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, cast

import pandas as pd
import requests

from coralnet_scraper import s3_io
from coralnet_scraper.config import DEFAULT_PREFIX
from coralnet_scraper.errors import ContentError, SessionExpired, _ExportPrepTimeout, _retrying
from coralnet_scraper.parsers import parse_export_annotations_prep_form
from coralnet_scraper.session import _PAGE_EXPORT_TIMEOUT, CoralNetSession, browse_page_url
from coralnet_scraper.verify import strip_status
from coralnet_scraper.walker import ImageRef

logger = logging.getLogger(__name__)

_EXPORT_PREP_TIMEOUT_MIN = 300
_EXPORT_PREP_TIMEOUT_MAX = 7200  # 2 hours
_EXPORT_SERVE_TIMEOUT_MIN = 120
_EXPORT_SERVE_TIMEOUT_MAX = 1800
_EXPORT_PREP_ROWS_PER_S = 250  # export_prep budget; s57 ran about 440 rows/s in a live run
_DEFAULT_POINTS_PER_IMAGE = 100  # when the overview gives no point generation method
_EXPORT_PREP_TIMEOUT_STEP = 300  # export_prep read timeout added per further attempt
_EXPORT_PREP_TIMEOUT_RETRY_WAIT_S = (
    60  # pause after an export_prep read timeout, before the next POST
)
_EXPORT_PREP_HEARTBEAT_S = 60  # log while a long export_prep POST is still running
_PAGINATED_EXPORT_THRESHOLD = 5000  # sources above this use per-page export
_PAGE_DELAY_SECONDS = 2.0  # politeness delay between page export requests
_EXPORT_ATTEMPTS = 4
_EXPORT_OPTIONS: dict[str, Any] = {
    "browse_action": "export_annotations",
    "label_format": "both",
    "optional_columns": ["annotator_info", "metadata_date_aux", "metadata_other"],
}

_CHUNK_PAGE = re.compile(r"/page-(\d+)\.csv$")


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
    _check_export_columns(df)
    return df


def _check_export_columns(df: pd.DataFrame) -> None:
    """Raise :class:`ContentError` if an export frame has a bad column or a row without a name."""
    if any("<" in str(c) or str(c).startswith("Unnamed") for c in df.columns):
        raise ContentError("export CSV has an HTML or unnamed column")
    if "Name" not in df.columns:
        raise ContentError("export CSV has no Name column")
    if df["Name"].isna().any():
        raise ContentError("export CSV has rows without a Name")


def _check_export_has_rows(df: pd.DataFrame) -> None:
    """Raise :class:`ContentError` if an export that must have rows is header-only."""
    if df.empty:
        raise ContentError("export has no rows although an image is Confirmed")


def _refs_by_page(refs: list[ImageRef]) -> dict[int, list[ImageRef]]:
    """Return the images of each browse page, pages ascending, images in walk order."""
    pages: dict[int, list[ImageRef]] = {}
    for ref in refs:
        pages.setdefault(ref.page, []).append(ref)
    return {page: pages[page] for page in sorted(pages)}


def _needs_rows(refs: list[ImageRef]) -> bool:
    """Return whether an export of ``refs`` must have rows: one of them is Confirmed."""
    return any(r.name.endswith(" - Confirmed") for r in refs)


def _is_usable_chunk(chunk: pd.DataFrame | None, page_refs: list[ImageRef]) -> bool:
    """Return whether a saved page chunk can be reused for the page of ``page_refs``."""
    if chunk is None:
        return False
    try:
        _check_export_columns(chunk)
        if _needs_rows(page_refs):
            _check_export_has_rows(chunk)
    except ContentError:
        return False
    return True


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
    return {strip_status(r.name) for r in refs}


def _check_export_names(df: pd.DataFrame, allowed: set[str]) -> None:
    """Raise :class:`ContentError` if the export has a row for an image outside ``allowed``."""
    foreign = set(df["Name"].astype(str)) - allowed
    if foreign:
        raise ContentError(
            f"export has rows for {len(foreign)} images not requested, e.g. {sorted(foreign)[:2]}"
        )


def _upload_csv(s3_client: Any, bucket: str, key: str, df: pd.DataFrame) -> None:
    """Serialize ``df`` as CSV without the index and upload it to ``key`` in ``bucket``."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3_io.upload_csv_body(s3_client, bucket=bucket, key=key, csv_text=buf.getvalue())


def export_prep_timeout_seconds(
    total_images: int | None, points_per_image: int | None = None
) -> int:
    """Read timeout for CoralNet ``export_prep`` POST (server builds the export).

    CoralNet's build time grows with annotation rows, estimated as ``total_images`` times
    ``points_per_image`` (:data:`_DEFAULT_POINTS_PER_IMAGE` when ``None``).
    """
    n = total_images or 0
    if n <= 0:
        return _EXPORT_PREP_TIMEOUT_MIN
    rows = n * (points_per_image or _DEFAULT_POINTS_PER_IMAGE)
    return min(
        _EXPORT_PREP_TIMEOUT_MAX,
        max(_EXPORT_PREP_TIMEOUT_MIN, 120 + rows // _EXPORT_PREP_ROWS_PER_S),
    )


def export_serve_timeout_seconds(total_images: int | None) -> int:
    """Read timeout for CoralNet ``export/serve`` GET (download generated CSV)."""
    n = total_images or 0
    if n <= 0:
        return _EXPORT_SERVE_TIMEOUT_MIN
    return min(_EXPORT_SERVE_TIMEOUT_MAX, max(_EXPORT_SERVE_TIMEOUT_MIN, 60 + n // 50))


class AnnotationExporter:
    """Export a source's annotations to ``annotations.csv`` on S3."""

    def __init__(self, auth: CoralNetSession, s3: Any, *, export_workers: int) -> None:
        """Create the exporter; no network call is made.

        Args:
            auth: The CoralNet session the exports run on.
            s3: S3 client the annotations and page chunks are written to.
            export_workers: Concurrent page exports of a paginated export.
        """
        self.auth = auth
        self.s3 = s3
        self.export_workers = export_workers

    def download_annotations(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = DEFAULT_PREFIX,
        *,
        refs: list[ImageRef],
        total_images: int,
        points_per_image: int | None = None,
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
            points_per_image: The overview's points per image; with ``total_images`` it
                sizes the bulk ``export_prep`` timeout. ``None`` uses a default.

        Returns:
            True on success; False on failure, after logging the error. The export must
            pass :func:`decode_export_csv`, name only images of ``refs`` and have rows when
            one of them is Confirmed; an export that does not, after the attempts of :meth:`_export_validated`, uploads nothing.
        """
        try:
            prep_timeout = export_prep_timeout_seconds(total_images, points_per_image)
            serve_timeout = export_serve_timeout_seconds(total_images)
            if total_images > 1000:
                logger.info(
                    "Source %s: %s images x %s points; export_prep timeout=%ss serve timeout=%ss",
                    source_id,
                    total_images,
                    points_per_image or _DEFAULT_POINTS_PER_IMAGE,
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

            if points_per_image is None:
                logger.info(
                    "Source %s: no points per image on the overview; assuming %s",
                    source_id,
                    _DEFAULT_POINTS_PER_IMAGE,
                )

            df_annotations = self._export_validated(
                self.auth.session,
                source_id,
                browse_page_url(self.auth.CORALNET_URL, source_id, 1),
                {"image_select_type": "all", **_EXPORT_OPTIONS},
                _export_names(refs),
                prep_timeout=prep_timeout,
                serve_timeout=serve_timeout,
                require_rows=_needs_rows(refs),
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
        s3_prefix: str = DEFAULT_PREFIX,
        *,
        refs: list[ImageRef],
        page_workers: int | None = None,
    ) -> bool:
        """Export annotations one browse page at a time for large sources.

        Each page covers ~20 images and returns in 2-10 seconds, avoiding the
        multi-hour blocking wait of the bulk ``image_select_type=all`` export.
        Pages are exported across ``page_workers`` threads, each sleeping
        :data:`_PAGE_DELAY_SECONDS` after a round trip. Each finished page is saved as
        ``_partial/page-<n>.csv``; a page with a usable saved chunk is not exported again,
        and one with an unusable chunk is exported again (see :meth:`_saved_chunk_pages`).
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
            saved = self._saved_chunk_pages(source_id, bucket_name, s3_prefix, pages)
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

    def _saved_chunk_pages(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        pages: dict[int, list[ImageRef]],
    ) -> set[int]:
        """Return the pages of ``pages`` whose saved export chunk in ``_partial/`` is usable.

        A chunk that cannot be read, lacks a ``Name`` column or has no rows for a page with a
        Confirmed image is deleted, so its page is exported again. A chunk of a page outside
        ``pages`` is left alone and not returned.
        """
        keys = s3_io.object_sizes(self.s3, bucket_name, s3_io.partial_prefix(s3_prefix, source_id))
        saved = sorted(
            page
            for key in keys
            if (m := _CHUNK_PAGE.search(key)) and (page := int(m.group(1))) in pages
        )
        chunk_keys = [s3_io.partial_key(s3_prefix, source_id, f"page-{page}.csv") for page in saved]
        with ThreadPoolExecutor(max_workers=self.export_workers) as executor:
            chunks = list(
                executor.map(lambda k: self._read_saved_chunk(bucket_name, k), chunk_keys)
            )
        usable = {
            page
            for page, chunk in zip(saved, chunks, strict=True)
            if _is_usable_chunk(chunk, pages[page])
        }
        unusable = [k for page, k in zip(saved, chunk_keys, strict=True) if page not in usable]
        if unusable:
            logger.warning("source %s: deleting %d unusable saved chunks", source_id, len(unusable))
            s3_io.delete_keys(self.s3, bucket_name, unusable)
        return usable

    def _read_saved_chunk(self, bucket_name: str, key: str) -> pd.DataFrame | None:
        """Return the saved chunk at ``key``, or None when the key is missing or not a CSV."""
        try:
            return s3_io.read_csv_s3(self.s3, bucket_name, key, encoding="utf-8")
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None

    def _assemble_chunks(
        self, source_id: int, bucket_name: str, s3_prefix: str, pages: list[int]
    ) -> pd.DataFrame:
        """Return the saved export chunks of ``pages`` concatenated in that order.

        The chunks are read across ``self.export_workers`` threads.

        Raises:
            FileNotFoundError: If a page has no saved chunk.
        """
        keys = [s3_io.partial_key(s3_prefix, source_id, f"page-{page}.csv") for page in pages]

        def read(key: str) -> pd.DataFrame:
            chunk = s3_io.read_csv_s3(self.s3, bucket_name, key, encoding="utf-8")
            if chunk is None:
                raise FileNotFoundError(f"s3://{bucket_name}/{key}")
            return chunk

        with ThreadPoolExecutor(max_workers=self.export_workers) as executor:
            chunks = list(executor.map(read, keys))
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
        key = s3_io.partial_key(s3_prefix, source_id, f"page-{page_num}.csv")
        _upload_csv(self.s3, bucket_name, key, chunk)
        return len(chunk)

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

        ``stop`` ends the export's retries once set (see :meth:`_export_validated`). A
        header-only export of a page with a Confirmed image is exported again.
        """
        form = {
            "image_form_type": "search",
            "image_select_type": "selected",
            "image_id_list": "_".join(str(r.id) for r in page_refs),
            **_EXPORT_OPTIONS,
        }
        chunk_df = self._export_validated(
            self.auth._get_export_session(),
            source_id,
            browse_page_url(self.auth.CORALNET_URL, source_id, page_num),
            form,
            _export_names(page_refs),
            prep_timeout=_PAGE_EXPORT_TIMEOUT,
            serve_timeout=_PAGE_EXPORT_TIMEOUT,
            stop=stop,
            require_rows=_needs_rows(page_refs),
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
        require_rows: bool = False,
    ) -> pd.DataFrame:
        """Return an export whose rows name only ``allowed_names``, retrying the round trip.

        A header-only export is retried like a foreign name when ``require_rows`` is set.

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
            require_rows: Whether an export without rows is a content failure.

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
                        self.auth._relogin(session)
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
                    if require_rows:
                        _check_export_has_rows(df)
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
            f"{self.auth.CORALNET_URL}/source/{source_id}/export/serve/"
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
        url = f"{self.auth.CORALNET_URL}/source/{source_id}/annotation/export_prep/"
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
