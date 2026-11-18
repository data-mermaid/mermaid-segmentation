"""Bulk discovery + download orchestration (public CoralNet sources → S3)."""

from __future__ import annotations

import getpass
import logging
import os
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from mermaidseg.datasets.coralnet.scraper.models import DownloadResult
from mermaidseg.datasets.coralnet.scraper.s3_io import annotations_csv_exists

logger = logging.getLogger(__name__)


def discover_public_source_ids(
    *,
    listing_url: str = "https://coralnet.ucsd.edu/source/about/",
    timeout: float = 50.0,
) -> list[int]:
    resp = requests.get(listing_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    anchors = soup.find_all("a", href=True)
    links = sorted(
        {
            urljoin(listing_url, a["href"])
            for a in anchors
            if urlparse(urljoin(listing_url, a["href"])).scheme in ("http", "https")
        }
    )
    source_links = [lk for lk in links if "/source/" in lk]
    return sorted({int(lk.split("/")[-2]) for lk in source_links})


def load_coralnet_credentials() -> tuple[str, str]:
    """Prefer env CORALNET_USERNAME / CORALNET_PASSWORD."""
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
    legacy_skip_annotations: bool,
    complete_source_ids: set[int] | None,
) -> bool:
    """Return True when the batch loop should not call download for this source."""
    if force:
        return False
    if complete_source_ids is not None and int(source_id) in complete_source_ids:
        return True
    if legacy_skip_annotations and annotations_csv_exists(s3_client, bucket, s3_prefix, source_id):
        logger.info("Skip source %s: annotations.csv already on S3 (legacy skip)", source_id)
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
    legacy_skip_annotations: bool = True,
    complete_source_ids: set[int] | None = None,
    delay_seconds: float = 0.0,
) -> dict[int, DownloadResult]:
    """Download each source when not skipped; returns per-source results."""
    results: dict[int, DownloadResult] = {}
    for sid in tqdm(list(source_ids), desc="CoralNet sources"):
        if should_skip_download(
            source_id=sid,
            bucket=bucket,
            s3_prefix=s3_prefix,
            s3_client=s3_client,
            force=force,
            legacy_skip_annotations=legacy_skip_annotations,
            complete_source_ids=complete_source_ids,
        ):
            continue
        res = downloader.download_source(
            sid, bucket_name=bucket, s3_prefix=s3_prefix, clean_prefix_before=False
        )
        if res.skipped_no_confirmed:
            logger.info("Skip source %s: zero confirmed images on website", sid)
        results[sid] = res
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return results


def complete_source_ids_from_audit_parquet(path: os.PathLike[str] | str) -> set[int]:
    """Load ``source_id`` values that are marked complete in an audit parquet."""
    df = pd.read_parquet(path)
    mask = df["is_complete"].astype(bool)
    return {int(x) for x in df.loc[mask, "source_id"].tolist()}
