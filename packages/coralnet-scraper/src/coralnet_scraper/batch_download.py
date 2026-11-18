"""Bulk discovery + download orchestration (public CoralNet sources → S3)."""

from __future__ import annotations

import getpass
import logging
import os
import time
from collections.abc import Iterable
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from coralnet_scraper.models import DownloadResult
from coralnet_scraper.s3_io import annotations_csv_exists

logger = logging.getLogger(__name__)

_SOURCE_LISTING_URL = "https://coralnet.ucsd.edu/source/about/"  # links every public source
_LISTING_TIMEOUT_S = 50.0


def discover_public_source_ids() -> list[int]:
    """Return the sorted ids of the public CoralNet sources linked from the listing page.

    Raises:
        requests.HTTPError: If the listing page returns an error status.
    """
    resp = requests.get(_SOURCE_LISTING_URL, timeout=_LISTING_TIMEOUT_S)
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
    source_links = [lk for lk in links if "/source/" in lk]
    return sorted({int(lk.split("/")[-2]) for lk in source_links})


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
    force: bool,
) -> bool:
    """Return whether the batch loop should skip ``source_id``.

    A source is skipped when its ``annotations.csv`` is already on S3, unless ``force`` is set.
    """
    if force:
        return False
    if annotations_csv_exists(s3_client, bucket, s3_prefix, source_id):
        logger.info("Skip source %s: annotations.csv already on S3", source_id)
        return True
    return False


def run_batch_download(
    source_ids: Iterable[int],
    *,
    downloader: Any,
    bucket: str,
    s3_prefix: str,
    s3_client: Any,
    force: bool = False,
    delay_seconds: float = 0.0,
) -> dict[int, DownloadResult]:
    """Download each source that is not skipped.

    Args:
        source_ids: CoralNet source ids to process.
        downloader: Client that provides ``download_source``.
        bucket: Destination S3 bucket.
        s3_prefix: Destination key prefix.
        s3_client: S3 client used for the skip check.
        force: Download every source, even when its annotations already exist.
        delay_seconds: Pause after each download.

    Returns:
        A result per downloaded source, keyed by source id; skipped sources are absent.
        A source whose skip check or download raises gets a failed result whose error code is
        the exception class name, and the run goes on.
    """
    results: dict[int, DownloadResult] = {}
    for sid in tqdm(list(source_ids), desc="CoralNet sources"):
        try:
            if should_skip_download(
                source_id=sid,
                bucket=bucket,
                s3_prefix=s3_prefix,
                s3_client=s3_client,
                force=force,
            ):
                continue
            res = downloader.download_source(sid, bucket_name=bucket, s3_prefix=s3_prefix)
        except Exception as e:  # noqa: BLE001
            logger.exception("Source %s failed: %s", sid, e)
            res = DownloadResult(ok=False, errors=[type(e).__name__])
        if res.skipped_no_confirmed:
            logger.info("Skip source %s: zero confirmed images on website", sid)
        results[sid] = res
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return results
