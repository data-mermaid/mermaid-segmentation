"""Download CoralNet sources to S3, resuming the stages a previous run finished."""

from __future__ import annotations

import importlib.metadata
import logging
from datetime import UTC, datetime
from typing import Any, cast

import boto3
import pandas as pd
import requests
from botocore.config import Config as BotoConfig
from bs4 import BeautifulSoup

from coralnet_scraper import s3_io
from coralnet_scraper.config import (
    DEFAULT_PREFIX,
    get_export_workers,
    get_requests_per_second,
    get_scraper_workers,
    validate_requests_per_second,
    validate_workers,
)
from coralnet_scraper.errors import _WalkCountMismatch
from coralnet_scraper.exporter import AnnotationExporter, _upload_csv
from coralnet_scraper.images import ImageFetcher
from coralnet_scraper.models import DownloadResult, ImageFailures
from coralnet_scraper.parsers import (
    classifier_plot_data_from_source_html,
    parse_image_status_counts,
    parse_points_per_image,
    parse_source_access,
    parse_total_images_from_source_html,
)
from coralnet_scraper.session import _PAGE_GET_TIMEOUT, CoralNetSession
from coralnet_scraper.verify import Verification, orphan_image_keys, verify_source
from coralnet_scraper.walker import ImageRef, SourceWalker, _partial_key

logger = logging.getLogger(__name__)

_SOURCE_INACCESSIBLE_STATUS = 404  # a missing source overview; a 403 is a block, not the source
_STATE_FILE = "state.json"
_SCRAPER_VERSION = importlib.metadata.version("coralnet-scraper")


def _image_list_frame(refs: list[ImageRef]) -> pd.DataFrame:
    """Return the ``image_list.csv`` frame of the walk; ``Image URL`` stays empty."""
    return pd.DataFrame(
        {
            "Name": [r.name for r in refs],
            "Image Page": [r.view_path for r in refs],
            "Image URL": pd.NA,
        }
    )


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


class CoralNetDownloader:
    """Download CoralNet source artifacts using ``requests`` and upload to S3.

    The work is split across :attr:`auth` (login and sessions), :attr:`walker` (the browse
    walk), :attr:`exporter` (annotations) and :attr:`images`; this class runs the stages.
    """

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
        self.auth = CoralNetSession(
            username,
            password,
            max_workers=self.max_workers,
            requests_per_second=self.requests_per_second,
        )
        self._s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=self.max_workers + 4))
        self.exporter = AnnotationExporter(self.auth, self._s3, export_workers=self.export_workers)
        self.walker = SourceWalker(self.auth, self._s3, reset_partial=self._reset_partial)
        self.images = ImageFetcher(self.auth, self._s3, max_workers=self.max_workers)

    @property
    def s3(self) -> Any:
        """The S3 client every component writes with."""
        return self._s3

    @s3.setter
    def s3(self, client: Any) -> None:
        """Use ``client`` for this downloader and each of its components."""
        self._s3 = client
        self.exporter.s3 = client
        self.walker.s3 = client
        self.images.s3 = client

    @property
    def session(self) -> requests.Session:
        """The main logged-in CoralNet session."""
        return self.auth.session

    @property
    def logged_in(self) -> bool:
        """Whether the main session has logged in."""
        return self.auth.logged_in

    def login(self) -> bool:
        """Log the main session in to CoralNet; see :meth:`CoralNetSession.login`."""
        return self.auth.login()

    def transport_stats(self) -> dict[str, Any]:
        """Return the request accounting of every session of this client."""
        return self.auth.transport_stats()

    def fetch_total_images(self, source_id: int) -> int | None:
        """Return the image count on the source's overview page, or None when it lists none."""
        return self.auth.fetch_total_images(source_id)

    _META_KEYS = ("x", "y", "nimages", "date", "traintime", "pk")
    _META_COLS = ("Classifier nbr", "Accuracy", "Trained on", "Date", "Traintime", "Global id")

    def download_metadata(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = DEFAULT_PREFIX,
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
                else self.auth._get_source_overview_html(source_id)
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
        self, source_id: int, bucket_name: str, s3_prefix: str = DEFAULT_PREFIX
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
            url = f"{self.auth.CORALNET_URL}/source/{source_id}/labelset/"
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

    def _prune_orphans(
        self, source_id: int, bucket_name: str, s3_prefix: str, *, dry_run: bool = False
    ) -> int:
        """Delete the source's image objects that its image list lacks; return how many.

        With ``dry_run`` it logs each orphan key and deletes nothing.
        """
        keys = orphan_image_keys(self.s3, bucket_name, s3_prefix, source_id)
        if dry_run:
            for key in keys:
                logger.info("Source %s: would prune orphan image object %s", source_id, key)
            return 0
        s3_io.delete_keys(self.s3, bucket_name, keys)
        if keys:
            logger.info("Source %s: pruned %d orphan image objects", source_id, len(keys))
        return len(keys)

    def verify_existing(self, source_id: int, bucket_name: str, s3_prefix: str) -> DownloadResult:
        """Verify a source already on S3 against CoralNet's live image count.

        The overview supplies both the status counts and the total. A source that passes
        gets its completion marker; nothing is downloaded. A source with no confirmed images
        or no images is skipped, and a skip writes no marker.

        Args:
            source_id: CoralNet source id.
            bucket_name: Bucket holding the source.
            s3_prefix: Key prefix of the scrape.

        Returns:
            ``ok`` with a skip flag for a skipped source; ``ok`` for a source that passed;
            otherwise the verification failure codes (or ``overview_total_missing``) in
            ``errors``.
        """
        counts = parse_image_status_counts(self.auth._get_source_overview_html(source_id))
        if counts.confirmed == 0:
            logger.info("Source %s has zero confirmed images on website; skipping", source_id)
            return DownloadResult(ok=True, skipped_no_confirmed=True)
        if counts.total_images is None:
            logger.error("Source %s: overview lists no image count", source_id)
            return DownloadResult(False, errors=["overview_total_missing"])
        if counts.total_images == 0:
            logger.info("Source %s reports zero images; skipping", source_id)
            return DownloadResult(ok=True, skipped_no_images=True)
        verification = verify_source(
            self.s3, bucket_name, s3_prefix, source_id, expected_total=counts.total_images
        )
        if not verification.ok:
            return DownloadResult(False, errors=verification.failures)
        self._mark_complete(source_id, bucket_name, s3_prefix, verification)
        return DownloadResult(True)

    def download_source(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = DEFAULT_PREFIX,
        *,
        prune: bool = False,
        dry_run: bool = False,
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
            dry_run: With ``prune``, log those objects and delete none; verification and the
                marker run as usual.

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
            overview_html = self.auth._get_source_overview_html(sid)
        except Exception as e:  # noqa: BLE001
            response = e.response if isinstance(e, requests.HTTPError) else None
            status = response.status_code if response is not None else None
            if status == 403:
                return DownloadResult(False, errors=["overview_forbidden"])
            if status == _SOURCE_INACCESSIBLE_STATUS:
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
            sid,
            bucket_name,
            s3_prefix,
            overview_html,
            counts.total_images,
            prune=prune,
            dry_run=dry_run,
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
        refs, total = self.walker.walk_source(
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
        dry_run: bool,
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
            if not self.exporter.download_annotations(
                sid,
                bucket_name,
                s3_prefix,
                refs=refs,
                total_images=total_images,
                points_per_image=parse_points_per_image(overview_html),
            ):
                return DownloadResult(False, errors=[*errs, "annotations_failed"])
            self._finish_stage(sid, bucket_name, s3_prefix, state, "export")
        image_failures = ImageFailures()
        if "images" not in state["stages"]:
            image_failures = self.images._download_missing_images(refs, sid, bucket_name, s3_prefix)
            errs += _image_failure_codes(image_failures)
            if image_failures == ImageFailures():
                self._finish_stage(sid, bucket_name, s3_prefix, state, "images")
        if errs:
            return DownloadResult(ok=False, errors=errs, image_failures=image_failures)
        errs += self._verify_and_mark(sid, bucket_name, s3_prefix, total_images, prune, dry_run)
        return DownloadResult(ok=not errs, errors=errs, image_failures=image_failures)

    def _verify_and_mark(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str,
        total_images: int,
        prune: bool,
        dry_run: bool,
    ) -> list[str]:
        """Verify the source and write its marker when it passes.

        With ``prune`` the orphan image objects go first (only logged under ``dry_run``).

        Returns:
            ``["verification_failed"]`` after discarding ``_partial/`` when verification
            fails, else an empty list.
        """
        verification = verify_source(
            self.s3, bucket_name, s3_prefix, source_id, expected_total=total_images
        )
        if (
            verification.ok
            and prune
            and self._prune_orphans(source_id, bucket_name, s3_prefix, dry_run=dry_run)
        ):
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
