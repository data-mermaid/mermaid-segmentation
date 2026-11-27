"""Discover public CoralNet sources and download the incomplete ones to S3.

Run: ``uv run --package coralnet-scraper coralnet-scrape``. The destination comes from
``MERMAID_CORALNET_BUCKET`` (required) and ``MERMAID_CORALNET_PREFIX`` (optional).
``coralnet-scrape verify-existing`` marks sources already on S3 that verify;
``coralnet-scrape invalidate`` deletes completion markers so the next run downloads again.
``--parent-version`` with ``--parent-manifest-sha256`` pins a published ``coralnet-raw`` version:
its images count as present for the default run and ``verify-existing``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import botocore.exceptions

from coralnet_scraper import s3_io
from coralnet_scraper.batch_download import (
    BatchOutcome,
    discover_public_source_ids,
    load_coralnet_credentials,
    run_batch_download,
)
from coralnet_scraper.config import (
    DEFAULT_EXPORT_WORKERS,
    ENV_EXPORT_WORKERS,
    ENV_REQUESTS_PER_SECOND,
    ENV_SCRAPER_WORKERS,
    get_bucket,
    get_prefix,
)
from coralnet_scraper.downloader import CoralNetDownloader
from coralnet_scraper.models import DownloadResult
from coralnet_scraper.pin import RawPin, load_raw_pin

logger = logging.getLogger(__name__)


_MIN_LISTING_RATIO = 0.95  # a listing below this share of the previous one stops the run
DEFAULT_BREAKER_COOLDOWN_S = 600.0  # pause after a circuit-breaker trip before probing
_SOURCE_PREFIX = re.compile(r"/s(\d+)/$")

_IMAGE_COUNT_ATTR = {
    "images_missing_url": "missing_url",
    "images_upload_failed": "upload_failed",
    "images_outage": "skipped",
}


def _summarize_failures(results: dict[int, DownloadResult]) -> list[str]:
    """Return the failure summary lines, or an empty list when no source failed.

    The first line is the number of failed sources. One line follows per error kind, kinds
    sorted, listing the failed source ids sorted. Image kinds read
    ``kind: 202 (14 images), 303 (2 images)``; other kinds read ``kind: [101, 404]``.
    """
    failed = {sid: res for sid, res in results.items() if not res.ok}
    if not failed:
        return []
    by_kind: dict[str, list[int]] = {}
    for sid in sorted(failed):
        for kind in failed[sid].errors:
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
        "skipped": {
            "complete": outcome.skipped_complete,
            "no_confirmed": sorted(
                sid for sid, res in outcome.results.items() if res.skipped_no_confirmed
            ),
            "no_images": sorted(
                sid for sid, res in outcome.results.items() if res.skipped_no_images
            ),
        },
        "stopped": outcome.stopped,
    }


def _summary_lines(outcome: BatchOutcome) -> list[str]:
    """Return the end-of-run error lines: the early-stop reason, if any, then the failures."""
    stop = [f"Run stopped early: {outcome.stopped}"] if outcome.stopped else []
    return stop + _summarize_failures(outcome.results)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delay-seconds", type=float, default=0.0)
    p.add_argument(
        "--breaker-cooldown",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Pause this long when the circuit breaker trips, then probe the next source; "
            f"0 stops at the trip (default: {DEFAULT_BREAKER_COOLDOWN_S:g})"
        ),
    )
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
        help=(
            "Concurrent page exports of a paginated export "
            f"(default: {DEFAULT_EXPORT_WORKERS} or {ENV_EXPORT_WORKERS})"
        ),
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
    p.add_argument(
        "--prune",
        action="store_true",
        help="Delete image objects missing from a verified image list, before its marker",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="With --prune, log the orphan image objects instead of deleting them",
    )
    p.add_argument(
        "--allow-source-drop",
        action="store_true",
        help="Run even when CoralNet lists under 95%% of the sources it listed last run",
    )
    p.add_argument(
        "--parent-version",
        help="Version of the published coralnet-raw dataset whose images count as present",
    )
    p.add_argument(
        "--parent-manifest-sha256",
        help="Manifest digest of --parent-version; required with it",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs") / "segdata-cache",
        help="Local cache of the pinned version's tables (default: outputs/segdata-cache)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    commands = p.add_subparsers(dest="command", metavar="{verify-existing,invalidate}")
    verify = commands.add_parser(
        "verify-existing",
        help="Verify unmarked sources already on S3 and mark those that pass; download nothing",
    )
    verify.add_argument(
        "--source-ids",
        dest="verify_ids",
        type=_source_id_list,
        default=None,
        help="Comma-separated source ids to verify, in order (default: every public source)",
    )
    invalidate = commands.add_parser(
        "invalidate", help="Delete completion markers and partial state; download nothing"
    )
    which = invalidate.add_mutually_exclusive_group(required=True)
    which.add_argument("--source-ids", dest="invalidate_ids", type=_source_id_list)
    which.add_argument("--all", dest="invalidate_all", action="store_true")
    which.add_argument(
        "--changed",
        action="store_true",
        help="Only marked sources whose live image count differs from the marker's",
    )
    invalidate.add_argument(
        "--dry-run",
        dest="invalidate_dry_run",
        action="store_true",
        help="Log the sources that would be invalidated; delete nothing",
    )
    invalidate.add_argument(
        "--yes", action="store_true", help="Confirm --all, which invalidates every source"
    )
    args = p.parse_args(argv)
    if args.breaker_cooldown is not None and not (
        math.isfinite(args.breaker_cooldown) and args.breaker_cooldown >= 0
    ):
        p.error("--breaker-cooldown must be a non-negative number of seconds")
    if args.command is not None and (
        args.prune or args.allow_source_drop or args.dry_run or args.breaker_cooldown is not None
    ):
        p.error(
            "--prune, --dry-run, --allow-source-drop and --breaker-cooldown "
            f"do not apply to {args.command}"
        )
    if (args.parent_version is None) != (args.parent_manifest_sha256 is None):
        p.error("--parent-version and --parent-manifest-sha256 go together")
    if args.command == "invalidate" and args.parent_version is not None:
        p.error("--parent-version and --parent-manifest-sha256 do not apply to invalidate")
    if args.command is None and args.dry_run and not args.prune:
        p.error("--dry-run needs --prune")
    if (
        args.command == "invalidate"
        and args.invalidate_all
        and not (args.yes or args.invalidate_dry_run)
    ):
        p.error("invalidate --all needs --yes (or --dry-run to preview)")
    return args


def _listing_shrank(previous: list[int] | None, current: list[int]) -> bool:
    """Return whether ``current`` lists under :data:`_MIN_LISTING_RATIO` of ``previous``."""
    return previous is not None and len(current) < _MIN_LISTING_RATIO * len(previous)


def _guard_discovery(
    s3: Any, bucket: str, prefix: str, discovered: list[int], *, allow_drop: bool
) -> bool:
    """Compare the listing with the previous run's; save it unless the run must stop.

    Returns:
        False when the listing is empty, or shrank too far and ``allow_drop`` is not set.
    """
    if not discovered:
        logger.error("CoralNet listed no public sources; stopping")
        return False
    key = f"{prefix}/_meta/discovered_source_ids.json"
    previous = s3_io.read_json_s3(s3, bucket, key)
    if _listing_shrank(previous, discovered) and not allow_drop:
        logger.error(
            "CoralNet lists %d sources, %d last run; stopping (--allow-source-drop to run)",
            len(discovered),
            len(previous),
        )
        return False
    dropped = sorted(set(previous or []) - set(discovered))
    if dropped:
        logger.warning("%d sources are no longer listed: %s", len(dropped), dropped)
    s3_io.write_json_s3(s3, bucket, key, sorted(discovered))
    return True


def _stored_source_ids(s3: Any, bucket: str, prefix: str) -> list[int]:
    """Return the ids of the sources that have an ``s<id>/`` folder under ``prefix``."""
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{prefix.strip('/')}/", Delimiter="/"
    )
    folders = [cp["Prefix"] for page in pages for cp in page.get("CommonPrefixes", [])]
    return sorted(int(m.group(1)) for f in folders if (m := _SOURCE_PREFIX.search(f)))


def _changed_source_ids(
    downloader: CoralNetDownloader, s3: Any, bucket: str, prefix: str
) -> tuple[list[int], list[int]]:
    """Return the marked sources whose live image count differs from their marker's.

    Returns:
        ``(changed, unchecked)``: the changed sources, and those whose count could not be read.
    """
    changed, unchecked = [], []
    for sid in _stored_source_ids(s3, bucket, prefix):
        marker = s3_io.read_json_s3(s3, bucket, s3_io.object_key_complete_marker(prefix, sid))
        if marker is None:
            continue
        try:
            live = downloader.fetch_total_images(sid)
        except Exception:
            logger.exception("Source %s: could not read the live image count", sid)
            unchecked.append(sid)
            continue
        if live != marker["total_images"]:
            logger.info("Source %s: %s images, %s when marked", sid, live, marker["total_images"])
            changed.append(sid)
    return changed, unchecked


def _invalidate(
    args: argparse.Namespace, downloader: CoralNetDownloader, s3: Any, bucket: str, prefix: str
) -> int:
    """Delete the completion marker and ``_partial/`` of the sources ``args`` selects.

    With ``--dry-run`` it logs the keys it would delete, and deletes nothing.

    Returns:
        1 when ``--changed`` could not read some source's live count, else 0.
    """
    unchecked: list[int] = []
    if args.invalidate_ids is not None:
        ids = args.invalidate_ids
    elif args.invalidate_all:
        ids = _stored_source_ids(s3, bucket, prefix)
    else:
        ids, unchecked = _changed_source_ids(downloader, s3, bucket, prefix)
    for sid in ids:
        marker_key = s3_io.object_key_complete_marker(prefix, sid)
        partial = s3_io.partial_prefix(prefix, sid)
        if args.invalidate_dry_run:
            logger.info("Source %s: would delete %s and %s", sid, marker_key, partial)
            continue
        s3.delete_object(Bucket=bucket, Key=marker_key)
        s3_io.delete_prefix(s3, bucket, partial)
    verb = "Would invalidate" if args.invalidate_dry_run else "Invalidated"
    logger.info("%s %d sources: %s", verb, len(ids), ids)
    if unchecked:
        logger.error("Could not check %d sources: %s", len(unchecked), unchecked)
    return 1 if unchecked else 0


def _verify_existing(
    ids: list[int], downloader: CoralNetDownloader, s3: Any, bucket: str, prefix: str
) -> int:
    """Verify each unmarked source of ``ids`` already on S3; marked ones are skipped.

    Sources with no confirmed images or no images are logged as skipped and are not failures.

    Returns:
        1 when any source fails verification, else 0. Each failure is logged as an error.
    """
    failed: dict[int, list[str]] = {}
    for sid in ids:
        if s3_io.key_exists(s3, bucket, s3_io.object_key_complete_marker(prefix, sid)):
            continue
        try:
            result = downloader.verify_existing(sid, bucket, prefix)
        except Exception as e:
            logger.exception("Source %s: verification raised", sid)
            failed[sid] = [type(e).__name__]
            continue
        if result.skipped_no_confirmed or result.skipped_no_images:
            reason = "no confirmed images" if result.skipped_no_confirmed else "no images"
            logger.info("Source %s skipped: %s", sid, reason)
        elif not result.ok:
            failed[sid] = result.errors
    for sid in sorted(failed):
        logger.error("Source %s not verified: %s", sid, ", ".join(failed[sid]))
    return 1 if failed else 0


def _load_pin(args: argparse.Namespace, s3: Any) -> RawPin | None:
    """Return the pinned ``coralnet-raw`` version the flags name, or ``None`` without them."""
    if args.parent_version is None:
        return None
    return load_raw_pin(s3, args.parent_version, args.parent_manifest_sha256, args.cache_dir)


def main(argv: list[str] | None = None) -> int:
    """Run the scraper command line.

    Args:
        argv: Command-line arguments; ``None`` reads them from ``sys.argv``.

    Returns:
        Process exit code: 0 when every source downloads (or none are attempted), 1 when
        the CoralNet login fails, the source listing is empty or shrank below 95% of the previous run's
        (unless ``--allow-source-drop``), any source download fails, or the circuit breaker
        stops the run. A failed or stopped run logs ``Run stopped early: <reason>`` first
        when it stopped, then one error line with the failed-source count and one per error
        kind. Every run that
        reaches the batch loop writes a JSON run report to ``<prefix>/_runs/``. The
        ``verify-existing`` and ``invalidate`` subcommands read no credentials, do not log
        in, and return their own codes. With ``--parent-version`` the pinned version's tables are
        synced into ``--cache-dir`` before the downloader is built.
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
    # The subcommands read only public overview and listing pages, so they never log in.
    user, pw = load_coralnet_credentials() if args.command is None else ("", "")
    s3 = boto3.client("s3")
    pin = _load_pin(args, s3)
    downloader = CoralNetDownloader(
        user,
        pw,
        max_workers=args.scraper_workers,
        export_workers=args.export_workers,
        requests_per_second=args.requests_per_second,
        pin=pin,
    )
    logger.info(
        "Scraper max_workers=%d export_workers=%d requests_per_second=%s",
        downloader.max_workers,
        downloader.export_workers,
        downloader.requests_per_second,
    )
    if args.command == "invalidate":
        return _invalidate(args, downloader, s3, bucket, prefix)
    if args.command is None and not downloader.login():
        return 1
    discovered = discover_public_source_ids(downloader.session)
    logger.info("Discovered %d public CoralNet source ids", len(discovered))
    ids = args.source_ids if args.source_ids is not None else discovered
    if args.command == "verify-existing":
        ids = args.verify_ids if args.verify_ids is not None else ids
        return _verify_existing(ids, downloader, s3, bucket, prefix)
    if not _guard_discovery(s3, bucket, prefix, discovered, allow_drop=args.allow_source_drop):
        return 1
    breaker_cooldown_s = (
        DEFAULT_BREAKER_COOLDOWN_S if args.breaker_cooldown is None else args.breaker_cooldown
    )
    outcome = run_batch_download(
        ids,
        downloader=downloader,
        bucket=bucket,
        s3_prefix=prefix,
        s3_client=s3,
        prune=bool(args.prune),
        dry_run=bool(args.dry_run),
        delay_seconds=float(args.delay_seconds),
        breaker_cooldown_s=breaker_cooldown_s,
    )
    transport = downloader.transport_stats()
    report = _build_run_report(
        settings={
            "scraper_workers": downloader.max_workers,
            "export_workers": downloader.export_workers,
            "requests_per_second": downloader.requests_per_second,
            "source_ids": args.source_ids,
            "prune": bool(args.prune),
            "dry_run": bool(args.dry_run),
            "allow_source_drop": bool(args.allow_source_drop),
            "breaker_cooldown_s": breaker_cooldown_s,
            "parent_version": args.parent_version,
            "parent_manifest_sha256": args.parent_manifest_sha256,
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
