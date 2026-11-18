"""Discover public CoralNet sources and download missing ones to S3.

Run: ``uv run --package coralnet-scraper coralnet-scrape``. The destination comes from
``MERMAID_CORALNET_BUCKET`` (required) and ``MERMAID_CORALNET_PREFIX`` (optional).
"""

from __future__ import annotations

import argparse
import logging
import sys

import boto3

from coralnet_scraper.batch_download import (
    discover_public_source_ids,
    load_coralnet_credentials,
    run_batch_download,
)
from coralnet_scraper.config import ENV_SCRAPER_WORKERS, get_bucket, get_prefix
from coralnet_scraper.downloader import CoralNetDownloader

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the scraper command line.

    Args:
        argv: Command-line arguments; ``None`` reads them from ``sys.argv``.

    Returns:
        Process exit code: 0 when every source downloads (or none are attempted), 1 when
        the CoralNet login fails or any source download fails.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Never skip sources (downloads may overwrite artifacts)",
    )
    p.add_argument("--delay-seconds", type=float, default=0.0)
    p.add_argument(
        "--scraper-workers",
        type=int,
        default=None,
        help=(
            "Concurrent image download/upload workers "
            f"(default: auto from CPU count or {ENV_SCRAPER_WORKERS})"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    bucket = get_bucket()
    prefix = get_prefix()
    user, pw = load_coralnet_credentials()
    downloader = CoralNetDownloader(user, pw, max_workers=args.scraper_workers)
    logger.info("Scraper max_workers=%d", downloader.max_workers)
    if not downloader.login():
        return 1
    s3 = boto3.client("s3")
    ids = discover_public_source_ids()
    logger.info("Discovered %d public CoralNet source ids", len(ids))
    results = run_batch_download(
        ids,
        downloader=downloader,
        bucket=bucket,
        s3_prefix=prefix,
        s3_client=s3,
        force=bool(args.force),
        delay_seconds=float(args.delay_seconds),
    )
    failed = sorted(sid for sid, result in results.items() if not result.ok)
    if failed:
        logger.error("Failed to download %d source(s): %s", len(failed), failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
