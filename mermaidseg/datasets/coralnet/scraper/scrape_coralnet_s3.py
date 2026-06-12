"""Discover public CoralNet sources and download missing ones to S3.

Run as a script: ``uv run python -m mermaidseg.datasets.coralnet.scraper.scrape_coralnet_s3``
or ``python mermaidseg/datasets/coralnet/scraper/scrape_coralnet_s3.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import boto3

from mermaidseg.datasets.coralnet.scraper.batch_download import (
    complete_source_ids_from_audit_parquet,
    discover_public_source_ids,
    load_coralnet_credentials,
    run_batch_download,
)
from mermaidseg.datasets.coralnet.scraper.downloader import CoralNetDownloader

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    p.add_argument("--prefix", default="coralnet-public-images")
    p.add_argument(
        "--audit-parquet",
        default=None,
        help="If set, skip sources that are is_complete in this audit parquet",
    )
    p.set_defaults(legacy_skip_annotations=True)
    p.add_argument(
        "--no-legacy-skip-annotations",
        action="store_false",
        dest="legacy_skip_annotations",
        help="Do not skip when annotations.csv exists; still respects audit parquet complete set",
    )
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
        help="Concurrent image download/upload workers (default: auto from CPU count or MERMAID_CORALNET_SCRAPER_WORKERS)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    user, pw = load_coralnet_credentials()
    downloader = CoralNetDownloader(user, pw, max_workers=args.scraper_workers)
    logger.info("Scraper max_workers=%d", downloader.max_workers)
    if not downloader.login():
        return 1
    s3 = boto3.client("s3")
    ids = discover_public_source_ids()
    logger.info("Discovered %d public CoralNet source ids", len(ids))
    complete_ids = None
    if args.audit_parquet:
        complete_ids = complete_source_ids_from_audit_parquet(args.audit_parquet)
        logger.info("Loaded %d complete ids from audit", len(complete_ids))
    run_batch_download(
        ids,
        downloader=downloader,
        bucket=args.bucket,
        s3_prefix=args.prefix,
        s3_client=s3,
        force=bool(args.force),
        legacy_skip_annotations=args.legacy_skip_annotations,
        complete_source_ids=complete_ids,
        delay_seconds=float(args.delay_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
