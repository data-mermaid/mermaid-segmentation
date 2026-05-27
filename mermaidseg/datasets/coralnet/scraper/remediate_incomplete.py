"""CLI: probe CoralNet website for incomplete sources, classify actions, redownload to S3."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from mermaidseg.datasets.coralnet.etl.config import get_bucket, get_prefix
from mermaidseg.datasets.coralnet.scraper.batch_download import load_coralnet_credentials
from mermaidseg.datasets.coralnet.scraper.classify import classify_from_audit_series
from mermaidseg.datasets.coralnet.scraper.downloader import CoralNetDownloader
from mermaidseg.datasets.coralnet.scraper.models import RemediationAction

logger = logging.getLogger("coralnet_remediate")

_DO_REDOWNLOAD = frozenset(
    {
        RemediationAction.FULL_REDOWNLOAD,
        RemediationAction.REDOWNLOAD_CSV,
        RemediationAction.REDOWNLOAD_IMAGES,
    }
)


def _action_download_kwargs(action: RemediationAction) -> dict:
    if action == RemediationAction.FULL_REDOWNLOAD:
        return {
            "download_metadata": True,
            "download_labelset": True,
            "download_annotations": True,
            "download_images": True,
        }
    if action == RemediationAction.REDOWNLOAD_CSV:
        return {
            "download_metadata": False,
            "download_labelset": False,
            "download_annotations": True,
            "download_images": False,
        }
    if action == RemediationAction.REDOWNLOAD_IMAGES:
        return {
            "download_metadata": False,
            "download_labelset": False,
            "download_annotations": False,
            "download_images": True,
        }
    raise ValueError(f"no download mapping for {action}")


def _read_parquet(path: str | Path) -> pd.DataFrame:
    # pathlib.Path normalizes s3:// to s3:/…; pass the string through to pandas/s3fs.
    return pd.read_parquet(str(path))


def cmd_probe(args: argparse.Namespace) -> int:
    incomplete = _read_parquet(args.incomplete_parquet)
    audit = _read_parquet(args.audit_parquet)
    if "source_id" not in incomplete.columns:
        logger.error("incomplete parquet must have source_id")
        return 2
    ids = incomplete["source_id"].astype(int).tolist()
    audit_by = audit.set_index("source_id", drop=False)
    user, pw = load_coralnet_credentials()
    dl = CoralNetDownloader(user, pw)
    if not dl.login():
        return 1

    rows: list[dict] = []

    def work(sid: int) -> dict:
        probe = dl.probe_source(sid)
        if sid not in audit_by.index:
            action = RemediationAction.MANUAL_REVIEW
            base: dict = {"source_id": sid}
        else:
            ar = audit_by.loc[sid]
            if isinstance(ar, pd.DataFrame):
                ar = ar.iloc[0]
            action = classify_from_audit_series(probe, ar.to_dict())
            base = ar.to_dict()
        gap = None
        if probe.accessible:
            gap = probe.total_images_website - int(base.get("n_images_s3") or 0)
        return {
            **{k: base[k] for k in base if k != "source_id"},
            "source_id": sid,
            "probe_accessible": probe.accessible,
            "probe_error": probe.error,
            "total_images_website": probe.total_images_website,
            "n_confirmed_website": probe.n_confirmed_website,
            "n_unconfirmed_website": probe.n_unconfirmed_website,
            "n_unclassified_website": probe.n_unclassified_website,
            "source_url": probe.source_url,
            "images_gap": gap,
            "action": str(action),
        }

    for sid in ids:
        rows.append(work(sid))
        if args.delay_seconds:
            time.sleep(args.delay_seconds)

    report = pd.DataFrame(rows).sort_values("source_id")
    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(report))
    return 0


def _redownload_queue(report: pd.DataFrame) -> list[pd.Series]:
    rows: list[pd.Series] = []
    for _, row in report.sort_values("source_id").iterrows():
        if RemediationAction(row["action"]) in _DO_REDOWNLOAD:
            rows.append(row)
    return rows


def cmd_redownload(args: argparse.Namespace) -> int:
    report = _read_parquet(args.report)
    queue = _redownload_queue(report)
    total = len(queue)
    if total == 0:
        logger.info("No sources queued for redownload")
        return 0

    logger.info("Redownload queue: %d source(s)", total)
    user, pw = load_coralnet_credentials()
    dl = CoralNetDownloader(user, pw, max_workers=args.scraper_workers)
    logger.info("Scraper max_workers=%d", dl.max_workers)
    if not dl.login():
        return 1
    bucket, prefix = args.bucket, args.prefix
    for i, row in enumerate(queue, start=1):
        action = RemediationAction(row["action"])
        sid = int(row["source_id"])
        clean = bool(args.clean_prefix) and action == RemediationAction.FULL_REDOWNLOAD
        website_images = row.get("total_images_website")
        images_gap = row.get("images_gap")
        detail = ""
        if pd.notna(website_images):
            detail = f" website_images={int(website_images)}"
            if pd.notna(images_gap):
                detail += f" gap={int(images_gap)}"
        logger.info(
            "Redownload %d/%d: source %s action=%s clean=%s%s",
            i,
            total,
            sid,
            action.value,
            clean,
            detail,
        )
        if args.dry_run:
            continue
        kwargs = _action_download_kwargs(action)
        res = dl.download_source(
            sid,
            bucket_name=bucket,
            s3_prefix=prefix,
            clean_prefix_before=clean,
            **kwargs,
        )
        logger.info(
            "Redownload %d/%d: source %s finished ok=%s errors=%s",
            i,
            total,
            sid,
            res.ok,
            res.errors,
        )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_probe(args)
    if rc != 0:
        return rc
    redownload_args = argparse.Namespace(
        report=str(args.report_out),
        bucket=args.bucket,
        prefix=args.prefix,
        clean_prefix=args.clean_prefix,
        dry_run=args.dry_run,
        scraper_workers=args.scraper_workers,
    )
    return cmd_redownload(redownload_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coralnet-remediate", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def _add_verbose(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    p_probe = sub.add_parser(
        "probe", help="Merge audit + website probe; write reconciliation parquet"
    )
    _add_verbose(p_probe)
    p_probe.add_argument(
        "--incomplete-parquet",
        default=None,
        help="local path or s3:// URI (default: incomplete_df.parquet next to --audit-parquet)",
    )
    p_probe.add_argument(
        "--audit-parquet",
        required=True,
        help="local path or s3:// URI",
    )
    p_probe.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="default: reconciliation_report.parquet next to --audit-parquet",
    )
    p_probe.add_argument("--delay-seconds", type=float, default=0.0)
    p_probe.set_defaults(func=cmd_probe)

    p_down = sub.add_parser(
        "redownload", help="Apply redownload for actionable rows in report parquet"
    )
    _add_verbose(p_down)
    p_down.add_argument(
        "--report",
        required=True,
        help="local path or s3:// URI to reconciliation_report.parquet",
    )
    p_down.add_argument("--bucket", default=get_bucket())
    p_down.add_argument("--prefix", default=get_prefix())
    p_down.add_argument(
        "--clean-prefix",
        action="store_true",
        help="For full_redownload only, delete s{N}/ prefix on S3 before download",
    )
    p_down.add_argument("--dry-run", action="store_true")
    p_down.add_argument(
        "--scraper-workers",
        type=int,
        default=None,
        help="Concurrent image download/upload workers (default: auto from CPU count or MERMAID_CORALNET_SCRAPER_WORKERS)",
    )
    p_down.set_defaults(func=cmd_redownload)

    p_all = sub.add_parser(
        "all", help="probe then redownload (respects redownload flags on nested parser)"
    )
    _add_verbose(p_all)
    p_all.add_argument(
        "--incomplete-parquet",
        default=None,
        help="local path or s3:// URI (default: incomplete_df.parquet next to --audit-parquet)",
    )
    p_all.add_argument(
        "--audit-parquet",
        required=True,
        help="local path or s3:// URI",
    )
    p_all.add_argument("--report-out", type=Path, default=None)
    p_all.add_argument("--delay-seconds", type=float, default=0.0)
    p_all.add_argument("--bucket", default=get_bucket())
    p_all.add_argument("--prefix", default=get_prefix())
    p_all.add_argument("--clean-prefix", action="store_true")
    p_all.add_argument("--dry-run", action="store_true")
    p_all.add_argument(
        "--scraper-workers",
        type=int,
        default=None,
        help="Concurrent image download/upload workers (default: auto from CPU count or MERMAID_CORALNET_SCRAPER_WORKERS)",
    )
    p_all.set_defaults(func=cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    if args.command in ("probe", "all"):
        audit_dir = Path(args.audit_parquet).parent
        if getattr(args, "incomplete_parquet", None) is None:
            args.incomplete_parquet = str(audit_dir / "incomplete_df.parquet")
        if getattr(args, "report_out", None) is None:
            args.report_out = audit_dir / "reconciliation_report.parquet"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
