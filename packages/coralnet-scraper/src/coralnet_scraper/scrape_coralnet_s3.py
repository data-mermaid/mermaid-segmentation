"""Discover public CoralNet sources and download missing ones to S3.

Run: ``uv run --package coralnet-scraper coralnet-scrape``. The destination comes from
``MERMAID_CORALNET_BUCKET`` (required) and ``MERMAID_CORALNET_PREFIX`` (optional).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

import boto3
import botocore.exceptions

from coralnet_scraper.batch_download import (
    BatchOutcome,
    discover_public_source_ids,
    load_coralnet_credentials,
    run_batch_download,
)
from coralnet_scraper.config import (
    ENV_EXPORT_WORKERS,
    ENV_REQUESTS_PER_SECOND,
    ENV_SCRAPER_WORKERS,
    get_bucket,
    get_prefix,
)
from coralnet_scraper.downloader import CoralNetDownloader
from coralnet_scraper.models import DownloadResult

logger = logging.getLogger(__name__)


_NO_ERROR_CODE = "no_error_code"

_IMAGE_COUNT_ATTR = {
    "images_missing_url": "missing_url",
    "images_upload_failed": "upload_failed",
}


def _summarize_failures(results: dict[int, DownloadResult]) -> list[str]:
    """Return the failure summary lines, or an empty list when no source failed.

    The first line is the number of failed sources. One line follows per error kind, kinds
    sorted, listing the failed source ids sorted. A failed source with no error codes is
    listed under ``no_error_code``. Image kinds read
    ``kind: 202 (14 images), 303 (2 images)``; other kinds read ``kind: [101, 404]``.
    """
    failed = {sid: res for sid, res in results.items() if not res.ok}
    if not failed:
        return []
    by_kind: dict[str, list[int]] = {}
    for sid in sorted(failed):
        for kind in failed[sid].errors or [_NO_ERROR_CODE]:
            by_kind.setdefault(kind, []).append(sid)
    lines = [f"Failed to download {len(failed)} source(s)"]
    for kind in sorted(by_kind):
        attr = _IMAGE_COUNT_ATTR.get(kind)
        if attr is None:
            lines.append(f"{kind}: {by_kind[kind]}")
            continue
        items = [
            f"{sid} ({getattr(failed[sid].image_failures, attr)} images)" for sid in by_kind[kind]
        ]
        lines.append(f"{kind}: {', '.join(items)}")
    return lines


def _source_id_list(text: str) -> list[int]:
    """Parse ``--source-ids``: comma-separated integers, e.g. ``372,295``."""
    try:
        return [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated source ids, got {text!r}"
        ) from None


def _run_report_key(prefix: str, started_at: datetime, run_suffix: str) -> str:
    """Return the S3 key of a run report: ``<prefix>/_runs/<UTC time>-<suffix>.json``."""
    return f"{prefix}/_runs/{started_at.strftime('%Y%m%dT%H%M%SZ')}-{run_suffix}.json"


def _build_run_report(
    *,
    settings: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    transport: dict[str, Any],
    outcome: BatchOutcome,
) -> dict[str, Any]:
    """Return the JSON-ready run report of one batch run."""
    return {
        "settings": settings,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round((finished_at - started_at).total_seconds(), 3),
        "transport": transport,
        "sources": {
            str(sid): {"ok": res.ok, "errors": list(res.errors)}
            for sid, res in outcome.results.items()
        },
        "stopped": outcome.stopped,
    }


def _summary_lines(outcome: BatchOutcome) -> list[str]:
    """Return the end-of-run error lines: the early-stop reason, if any, then the failures."""
    stop = [f"Run stopped early: {outcome.stopped}"] if outcome.stopped else []
    return stop + _summarize_failures(outcome.results)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
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
    p.add_argument(
        "--export-workers",
        type=int,
        default=None,
        help=f"Concurrent page exports of a paginated export (default: 3 or {ENV_EXPORT_WORKERS})",
    )
    p.add_argument(
        "--requests-per-second",
        type=float,
        default=None,
        help=(
            "Rate limit shared by every CoralNet page request "
            f"(default: no limit, or {ENV_REQUESTS_PER_SECOND})"
        ),
    )
    p.add_argument(
        "--source-ids",
        type=_source_id_list,
        default=None,
        help="Comma-separated source ids to process, in order (default: every public source)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the scraper command line.

    Args:
        argv: Command-line arguments; ``None`` reads them from ``sys.argv``.

    Returns:
        Process exit code: 0 when every source downloads (or none are attempted), 1 when
        the CoralNet login fails, any source download fails, or the circuit breaker stops
        the run. A failed or stopped run logs ``Run stopped early: <reason>`` first when it
        stopped, then one error line with the failed-source count and one per error kind
        (``no_error_code`` for a failed source that reports none). Every run that reaches
        the batch loop writes a JSON run report to ``<prefix>/_runs/``.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    started_at = datetime.now(UTC)
    bucket = get_bucket()
    prefix = get_prefix()
    user, pw = load_coralnet_credentials()
    downloader = CoralNetDownloader(
        user,
        pw,
        max_workers=args.scraper_workers,
        export_workers=args.export_workers,
        requests_per_second=args.requests_per_second,
    )
    logger.info(
        "Scraper max_workers=%d export_workers=%d requests_per_second=%s",
        downloader.max_workers,
        downloader.export_workers,
        downloader.requests_per_second,
    )
    if not downloader.login():
        return 1
    s3 = boto3.client("s3")
    discovered = discover_public_source_ids(downloader.session)
    logger.info("Discovered %d public CoralNet source ids", len(discovered))
    ids = args.source_ids if args.source_ids is not None else discovered
    outcome = run_batch_download(
        ids,
        downloader=downloader,
        bucket=bucket,
        s3_prefix=prefix,
        s3_client=s3,
        force=bool(args.force),
        delay_seconds=float(args.delay_seconds),
    )
    transport = downloader.transport_stats()
    report = _build_run_report(
        settings={
            "scraper_workers": downloader.max_workers,
            "export_workers": downloader.export_workers,
            "requests_per_second": downloader.requests_per_second,
            "source_ids": args.source_ids,
        },
        started_at=started_at,
        finished_at=datetime.now(UTC),
        transport=transport,
        outcome=outcome,
    )
    key = _run_report_key(prefix, started_at, uuid.uuid4().hex[:8])
    logger.info("Transport: %s", json.dumps(transport, sort_keys=True))
    summary = _summary_lines(outcome)
    for line in summary:
        logger.error("%s", line)
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(report, indent=2),
            ContentType="application/json",
        )
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError):
        logger.exception("Could not write run report s3://%s/%s", bucket, key)
        return 1
    logger.info("Run report s3://%s/%s", bucket, key)
    return 1 if summary else 0


if __name__ == "__main__":
    raise SystemExit(main())
