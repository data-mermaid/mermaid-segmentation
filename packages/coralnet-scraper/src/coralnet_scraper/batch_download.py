"""Bulk discovery + download orchestration (public CoralNet sources → S3)."""

from __future__ import annotations

import getpass
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from coralnet_scraper.models import DownloadResult
from coralnet_scraper.s3_io import key_exists, object_key_complete_marker

logger = logging.getLogger(__name__)

_SOURCE_LISTING_URL = "https://coralnet.ucsd.edu/source/about/"  # links every public source
_LISTING_TIMEOUT_S = 50.0
_SOURCE_PATH_RE = re.compile(r"/source/(\d+)/")
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive outage failures that stop the run
STOPPED_CIRCUIT_OPEN = "circuit_open"


@dataclass(frozen=True)
class BatchOutcome:
    """Outcome of a batch run.

    Attributes:
        results: A result per attempted source, keyed by source id.
        stopped: Why the run stopped before the last source (``"circuit_open"``), or
            ``None`` when every source was handled.
        skipped_complete: Sources skipped because their completion marker exists.
    """

    results: dict[int, DownloadResult]
    stopped: str | None
    skipped_complete: int = 0


def discover_public_source_ids(session: requests.Session) -> list[int]:
    """Return the sorted ids of the public CoralNet sources linked from the listing page.

    Args:
        session: Session the listing page is fetched with.

    Raises:
        requests.HTTPError: If the listing page returns an error status.
    """
    resp = session.get(_SOURCE_LISTING_URL, timeout=_LISTING_TIMEOUT_S)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    hrefs = [cast(str, a["href"]) for a in soup.find_all("a", href=True)]
    links = sorted(
        {
            urljoin(_SOURCE_LISTING_URL, href)
            for href in hrefs
            if urlparse(urljoin(_SOURCE_LISTING_URL, href)).scheme in ("http", "https")
        }
    )
    # Signed-in pages also link non-source pages under /source/, such as /source/new/.
    matches = (_SOURCE_PATH_RE.fullmatch(urlparse(lk).path) for lk in links)
    return sorted({int(m.group(1)) for m in matches if m})


def load_coralnet_credentials() -> tuple[str, str]:
    """Return the CoralNet ``(username, password)``.

    Reads ``CORALNET_USERNAME`` and ``CORALNET_PASSWORD`` when both are set, otherwise prompts
    on the terminal.
    """
    env_u = os.environ.get("CORALNET_USERNAME") or ""
    env_p = os.environ.get("CORALNET_PASSWORD") or ""
    if env_u.strip() and env_p:
        return env_u.strip(), env_p
    user = input("CoralNet username: ").strip()
    pw = getpass.getpass("CoralNet password: ")
    return user, pw


def should_skip_download(
    *,
    source_id: int,
    bucket: str,
    s3_prefix: str,
    s3_client: Any,
) -> bool:
    """Return whether the batch loop should skip ``source_id``: its ``_complete.json`` exists."""
    if key_exists(s3_client, bucket, object_key_complete_marker(s3_prefix, source_id)):
        logger.info("Skip source %s: complete", source_id)
        return True
    return False


_OUTAGE_CODES = frozenset(
    {
        "login failed",
        "annotations_failed",
        "image_list_failed",
        "images_outage",
        "overview_forbidden",
        "overview_total_missing",
    }
)


def _is_outage_failure(res: DownloadResult, *, raised: bool = False) -> bool:
    """Return whether a failed result signals an outage rather than a problem of one source.

    A source whose download raised, or that failed with no error code, is an outage; otherwise
    the downloader's login, annotations, image-list and overview-fetch errors are.
    """
    if res.ok:
        return False
    return (
        raised
        or not res.errors
        or any(e in _OUTAGE_CODES or e.startswith("overview fetch failed") for e in res.errors)
    )


def _next_failure_streak(streak: int, res: DownloadResult, *, raised: bool = False) -> int:
    """Return the outage streak after ``res``: reset by ok, grown by an outage failure."""
    if res.ok:
        return 0
    return streak + 1 if _is_outage_failure(res, raised=raised) else streak


def run_batch_download(
    source_ids: Iterable[int],
    *,
    downloader: Any,
    bucket: str,
    s3_prefix: str,
    s3_client: Any,
    prune: bool = False,
    dry_run: bool = False,
    delay_seconds: float = 0.0,
    breaker_cooldown_s: float,
) -> BatchOutcome:
    """Download each source that is not skipped.

    Args:
        source_ids: CoralNet source ids to process.
        downloader: Client that provides ``download_source``.
        bucket: Destination S3 bucket.
        s3_prefix: Destination key prefix.
        s3_client: S3 client used for the skip check.
        prune: Let each downloaded source delete image objects its verified image list lacks.
        dry_run: With ``prune``, log the orphan objects instead of deleting them.
        delay_seconds: Pause after each download.
        breaker_cooldown_s: Seconds to pause when the circuit breaker trips before probing the
            next source; 0 stops the run at the trip.

    Returns:
        A result per downloaded source, keyed by source id; sources skipped as complete are
        absent and counted in ``skipped_complete``.
        A source whose skip check or download raises gets a failed result whose error code is
        the exception class name, and the run goes on. After
        :data:`CIRCUIT_BREAKER_THRESHOLD` consecutive outage failures (see
        :func:`_is_outage_failure`) the breaker trips. With ``breaker_cooldown_s`` of 0 the run
        stops with ``stopped="circuit_open"``; otherwise it sleeps once and probes the next
        source: an ok result resets the streak and the run continues (a later trip gets its own
        cooldown), an outage failure stops the run. Skipped sources and other failures neither
        count nor reset the streak.
    """
    results: dict[int, DownloadResult] = {}
    failure_streak = 0
    probing = False
    skipped_complete = 0
    for sid in tqdm(list(source_ids), desc="CoralNet sources"):
        raised = False
        try:
            if should_skip_download(
                source_id=sid,
                bucket=bucket,
                s3_prefix=s3_prefix,
                s3_client=s3_client,
            ):
                skipped_complete += 1
                continue
            res = downloader.download_source(
                sid, bucket_name=bucket, s3_prefix=s3_prefix, prune=prune, dry_run=dry_run
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Source %s failed: %s", sid, e)
            res = DownloadResult(ok=False, errors=[type(e).__name__])
            raised = True
        if res.skipped_no_confirmed:
            logger.info("Skip source %s: zero confirmed images on website", sid)
        results[sid] = res
        previous_streak = failure_streak
        failure_streak = _next_failure_streak(failure_streak, res, raised=raised)
        if res.ok:
            probing = False
        if failure_streak > previous_streak and failure_streak >= CIRCUIT_BREAKER_THRESHOLD:
            if probing or breaker_cooldown_s <= 0:
                logger.error(
                    "Stopping the run: %d consecutive sources failed (last: %s)",
                    failure_streak,
                    sid,
                )
                return BatchOutcome(results, STOPPED_CIRCUIT_OPEN, skipped_complete)
            logger.error(
                "Circuit breaker tripped: %d consecutive sources failed (last: %s); "
                "pausing %.0f s before probing the next source",
                failure_streak,
                sid,
                breaker_cooldown_s,
            )
            time.sleep(breaker_cooldown_s)
            probing = True
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return BatchOutcome(results, None, skipped_complete)
