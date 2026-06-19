"""Refresh truncated CoralNet ``image_list.csv`` files on S3.

``image_list.csv`` is the only bridge the ETL has from an annotation ``Name`` to CoralNet's numeric
``image_id`` (how image files are keyed in S3). When that list is truncated, the
annotations x image_list join silently drops every uncovered annotated image regardless of
confirmed status. This happened to ~32 sources in the ``20260526_807b611`` run (e.g. ``s295`` listed
1,064 images while ``annotations.csv`` referenced 79,064).

This script re-walks the CoralNet browse pages for the requested sources via
:meth:`CoralNetDownloader.get_images` (which keeps the per-image ``" - Confirmed/Unconfirmed"``
status suffix in ``Name``) and overwrites ``s<id>/image_list.csv``. The image **files** are intact
in S3 and are NOT re-downloaded. A guard refuses to replace an existing list with a shorter scrape.

Usage::

    export AWS_PROFILE=mermaid-core CORALNET_USERNAME=... CORALNET_PASSWORD=...
    uv run python scripts/refresh_coralnet_image_lists.py --source-ids 295,372,3371 --dry-run
    uv run python scripts/refresh_coralnet_image_lists.py \
        --source-ids-file outputs/coralnet_truncated_imagelist_sources.csv
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mermaidseg.datasets.coralnet.etl import config
from mermaidseg.datasets.coralnet.scraper import s3_io
from mermaidseg.datasets.coralnet.scraper.batch_download import load_coralnet_credentials
from mermaidseg.datasets.coralnet.scraper.downloader import CoralNetDownloader

logger = logging.getLogger("refresh_coralnet_image_lists")


def _image_list_key(prefix: str, source_id: int) -> str:
    return f"{prefix}/s{source_id}/image_list.csv"


def count_s3_images(client: Any, bucket: str, prefix: str, source_id: int) -> int:
    """Number of image files under ``s<id>/images/`` (single paginator pass)."""
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/s{source_id}/images/"):
        total += page.get("KeyCount", 0)
    return total


def existing_image_list_rows(client: Any, bucket: str, prefix: str, source_id: int) -> int | None:
    """Row count (excluding header) of the current ``image_list.csv``; ``None`` if
    absent/unreadable."""
    try:
        obj = client.get_object(Bucket=bucket, Key=_image_list_key(prefix, source_id))
        body = obj["Body"].read()
    except Exception:  # noqa: BLE001 - NoSuchKey or any read error -> treat as absent
        return None
    text = body.decode("utf-8", "replace") if isinstance(body, bytes | bytearray) else str(body)
    rows = [ln for ln in text.splitlines() if ln.strip()]
    return max(0, len(rows) - 1)  # minus header


def refresh_one(
    downloader: Any,
    client: Any,
    bucket: str,
    prefix: str,
    source_id: int,
    *,
    min_coverage: float = 0.9,
    browse_workers: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-scrape and (unless guarded/dry-run) overwrite one source's ``image_list.csv``.

    Guards against shipping another truncated list: the new scrape must list at least as many images
    as the existing file AND at least ``min_coverage`` of the actual S3 image files.

    ``browse_workers`` is forwarded to :meth:`CoralNetDownloader.get_images` to enable
    parallel page fetching. Use ``n_s3`` as the ``total_images_hint`` so the fan-out
    covers the full source even when the existing image_list.csv is truncated.
    """
    existing = existing_image_list_rows(client, bucket, prefix, source_id)
    n_s3 = count_s3_images(client, bucket, prefix, source_id)
    result: dict[str, Any] = {
        "source_id": source_id,
        "existing_rows": existing,
        "s3_images": n_s3,
        "new_rows": 0,
        "uploaded": False,
        "skipped_reason": None,
    }

    df, ok = downloader.get_images(
        source_id,
        total_images_hint=n_s3 or None,
        browse_workers=browse_workers,
    )
    if not ok or df is None or len(df) == 0:
        result["skipped_reason"] = "scrape_failed_or_empty"
        return result
    new_n = len(df)
    result["new_rows"] = new_n

    if existing is not None and new_n < existing:
        result["skipped_reason"] = f"guard: new({new_n}) < existing({existing})"
        return result
    if n_s3 and new_n < min_coverage * n_s3:
        result["skipped_reason"] = f"guard: new({new_n}) < {min_coverage:.0%} of s3_images({n_s3})"
        return result
    if dry_run:
        result["skipped_reason"] = "dry_run"
        return result

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3_io.upload_csv_body(
        client, bucket=bucket, key=_image_list_key(prefix, source_id), csv_text=buf.getvalue()
    )
    result["uploaded"] = True
    return result


def parse_source_ids(args: argparse.Namespace) -> list[int]:
    if args.source_ids:
        return [int(x) for x in args.source_ids.split(",") if x.strip()]
    path = Path(args.source_ids_file)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "source_id" not in df.columns:
        raise SystemExit(f"{path} has no 'source_id' column")
    return [int(x) for x in df["source_id"].tolist()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="refresh-coralnet-image-lists", description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-ids", default=None, help="Comma-separated CoralNet source ids")
    src.add_argument(
        "--source-ids-file", default=None, help="CSV/Parquet with a 'source_id' column"
    )
    p.add_argument("--bucket", default=config.get_bucket())
    p.add_argument("--prefix", default=config.get_prefix())
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="New list must list >= this fraction of S3 image files to be accepted",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Connection-pool size for the downloader (sources processed sequentially)",
    )
    p.add_argument(
        "--browse-workers",
        type=int,
        default=10,
        help="Parallel browse-page workers per source (uses ?page=N fan-out; default 10)",
    )
    p.add_argument("--dry-run", action="store_true", help="Report new vs old counts, do not upload")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    source_ids = parse_source_ids(args)
    logger.info("Refreshing image_list.csv for %d source(s)", len(source_ids))

    username, password = load_coralnet_credentials()
    downloader = CoralNetDownloader(username, password, max_workers=args.max_workers)
    if not downloader.login():
        logger.error("CoralNet login failed")
        return 1
    client = downloader.s3

    rows: list[dict[str, Any]] = []
    for sid in source_ids:
        try:
            r = refresh_one(
                downloader,
                client,
                args.bucket,
                args.prefix,
                sid,
                min_coverage=args.min_coverage,
                browse_workers=args.browse_workers,
                dry_run=args.dry_run,
            )
        except Exception as e:  # noqa: BLE001 - never abort the batch on one source
            r = {"source_id": sid, "skipped_reason": f"error: {type(e).__name__}: {e}"}
            logger.exception("source %s failed", sid)
        rows.append(r)
        logger.info(
            "s%s: existing=%s s3=%s new=%s uploaded=%s%s",
            sid,
            r.get("existing_rows"),
            r.get("s3_images"),
            r.get("new_rows"),
            r.get("uploaded"),
            f" ({r['skipped_reason']})" if r.get("skipped_reason") else "",
        )

    summary = pd.DataFrame(rows)
    uploaded = int(summary.get("uploaded", pd.Series(dtype=bool)).fillna(False).sum())
    logger.info("Done: %d uploaded, %d skipped/failed", uploaded, len(rows) - uploaded)
    if not args.dry_run:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
