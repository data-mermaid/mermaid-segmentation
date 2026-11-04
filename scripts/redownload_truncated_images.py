"""Redownload specific CoralNet images and overwrite their S3 originals.

For surgical fixes when a handful of source files are corrupt (e.g. truncated
downloads). Unlike the bulk downloader, this OVERWRITES existing S3 keys, and
validates each download with a full decode before uploading.

Usage (SageMaker, needs CORALNET_USERNAME / CORALNET_PASSWORD):
    uv run python scripts/redownload_truncated_images.py \
        --source-id 800 --image-ids 674993 675056
"""

from __future__ import annotations

import argparse
import io
import logging

import requests

from mermaidseg.datasets.coralnet.preprocessing.inspect import inspect_image
from mermaidseg.datasets.coralnet.scraper.batch_download import (
    load_coralnet_credentials,
)
from mermaidseg.datasets.coralnet.scraper.downloader import CoralNetDownloader

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        required=True,
        help="CoralNet image IDs (the .jpg basenames under s<N>/images/)",
    )
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--prefix", default="coralnet-public-images")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and validate only; do not upload",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    username, password = load_coralnet_credentials()
    downloader = CoralNetDownloader(username, password, max_workers=2)
    if not downloader.login():
        logger.error("CoralNet login failed")
        return 1

    page_urls = [f"/image/{image_id}/view/" for image_id in args.image_ids]
    image_urls = downloader.get_image_urls(page_urls)

    failures = 0
    for image_id, url in zip(args.image_ids, image_urls, strict=True):
        key = f"{args.prefix}/s{args.source_id}/images/{image_id}.jpg"
        if url is None:
            logger.error("%s: could not resolve image URL", image_id)
            failures += 1
            continue

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.content

        inspection = inspect_image(io.BytesIO(data))
        if not inspection.is_valid:
            logger.error(
                "%s: downloaded bytes failed validation (%s: %s) — NOT uploading",
                image_id,
                inspection.issue_type.value,
                inspection.error_message,
            )
            failures += 1
            continue

        try:
            old_size = downloader.s3.head_object(Bucket=args.bucket, Key=key)["ContentLength"]
        except Exception:  # noqa: BLE001 - key may not exist yet
            old_size = None

        if args.dry_run:
            logger.info(
                "[dry-run] %s: valid %s %dx%d, %d bytes (S3 currently %s bytes) -> s3://%s/%s",
                image_id,
                inspection.format,
                inspection.width,
                inspection.height,
                len(data),
                old_size,
                args.bucket,
                key,
            )
            continue

        downloader.s3.put_object(Bucket=args.bucket, Key=key, Body=data, ContentType="image/jpeg")
        logger.info(
            "%s: uploaded %d bytes (was %s) -> s3://%s/%s",
            image_id,
            len(data),
            old_size,
            args.bucket,
            key,
        )

    if failures:
        logger.error("%d/%d images failed", failures, len(args.image_ids))
    else:
        logger.info("All %d images redownloaded and validated", len(args.image_ids))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
