"""CLI entry point for the CoralNet ETL.

Run with ``python -m mermaidseg.datasets.coralnet.etl <subcommand>`` or via the ``coralnet-etl``
console script declared in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from pathlib import Path

import pandas as pd

from .annotations import build_annotations
from .audit import audit_sources
from .config import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_IMAGE_WORKERS,
    DEFAULT_RESIZE_THRESHOLD,
    DEFAULT_WORKERS,
    get_bucket,
    get_output_s3_prefix,
    get_prefix,
)
from .images import build_images
from .io import (
    compute_version_tag,
    make_s3_client,
    upload_to_s3,
    write_parquet_deterministic,
)
from .schemas import (
    ANNOTATIONS_PRIMARY_KEY,
    ANNOTATIONS_SCHEMA,
    AUDIT_PRIMARY_KEY,
    AUDIT_SCHEMA,
    IMAGES_PRIMARY_KEY,
    IMAGES_SCHEMA,
)

logger = logging.getLogger("coralnet_etl")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _output_dir(args: argparse.Namespace) -> Path:
    version = args.version_tag or compute_version_tag()
    base = Path(args.output_dir or "outputs/coralnet-etl")
    out = base / version
    out.mkdir(parents=True, exist_ok=True)
    return out


def _file_name(kind: str, version: str) -> str:
    return f"coralnet_{kind}_{version}.parquet"


def _audit_source_ids_from_args(args: argparse.Namespace) -> list[int] | None:
    has_csv = getattr(args, "source_ids", None)
    has_file = getattr(args, "source_ids_file", None)
    if has_csv and has_file:
        raise SystemExit("Use only one of --source-ids or --source-ids-file.")
    if has_csv:
        return [int(x.strip()) for x in str(has_csv).split(",") if x.strip()]
    if has_file:
        path_str = str(has_file)
        if path_str.startswith("s3://") or Path(path_str).suffix.lower() == ".parquet":
            df = pd.read_parquet(path_str)
        else:
            df = pd.read_csv(path_str)
        if "source_id" not in df.columns:
            raise SystemExit("--source-ids-file must contain a source_id column.")
        return [int(x) for x in df["source_id"].tolist()]
    return None


def _maybe_upload(
    local_path: Path, kind: str, version: str, bucket: str, args: argparse.Namespace
) -> None:
    if not args.upload_to_s3:
        return
    prefix = get_output_s3_prefix().rstrip("/")
    key = f"{prefix}/{version}/{local_path.name}"
    target = f"s3://{bucket}/{key}"
    logger.info("Uploading %s -> %s", local_path, target)
    upload_to_s3(make_s3_client(), local_path, target)


def _cmd_audit(args: argparse.Namespace) -> int:
    version = args.version_tag or compute_version_tag()
    out_dir = _output_dir(args)
    audit_df = audit_sources(
        bucket=args.bucket,
        prefix=args.prefix,
        workers=args.workers,
        limit_sources=args.limit_sources,
        source_ids=_audit_source_ids_from_args(args),
    )
    out_path = out_dir / _file_name("audit", version)
    write_parquet_deterministic(audit_df, out_path, AUDIT_SCHEMA, AUDIT_PRIMARY_KEY)
    logger.info("Wrote audit parquet: %s (%d rows)", out_path, len(audit_df))
    _maybe_upload(out_path, "audit", version, args.bucket, args)
    return 0


def _cmd_build_annotations(args: argparse.Namespace) -> int:
    version = args.version_tag or compute_version_tag()
    out_dir = _output_dir(args)
    audit_df = pd.read_parquet(args.audit)
    annotations_df = build_annotations(
        audit_df, bucket=args.bucket, prefix=args.prefix, workers=args.workers
    )
    out_path = out_dir / _file_name("annotations", version)
    write_parquet_deterministic(
        annotations_df, out_path, ANNOTATIONS_SCHEMA, ANNOTATIONS_PRIMARY_KEY
    )
    logger.info("Wrote annotations parquet: %s (%d rows)", out_path, len(annotations_df))
    _maybe_upload(out_path, "annotations", version, args.bucket, args)
    return 0


def _cmd_build_images(args: argparse.Namespace) -> int:
    version = args.version_tag or compute_version_tag()
    out_dir = _output_dir(args)
    annotations_df = pd.read_parquet(args.annotations)
    checkpoint_path = out_dir / _file_name("images", version)
    images_df = build_images(
        annotations_df,
        bucket=args.bucket,
        prefix=args.prefix,
        workers=args.workers,
        resize_threshold=args.resize_threshold,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        limit=args.limit_images,
    )
    write_parquet_deterministic(images_df, checkpoint_path, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)
    # Best-effort cleanup of part files now that final is consolidated.
    for part in sorted(checkpoint_path.parent.glob(f"{checkpoint_path.stem}.part_*.parquet")):
        with contextlib.suppress(OSError):
            part.unlink()
    logger.info("Wrote images parquet: %s (%d rows)", checkpoint_path, len(images_df))
    _maybe_upload(checkpoint_path, "images", version, args.bucket, args)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    version = args.version_tag or compute_version_tag()
    out_dir = _output_dir(args)

    # Audit
    audit_df = audit_sources(
        bucket=args.bucket,
        prefix=args.prefix,
        workers=args.workers,
        limit_sources=args.limit_sources,
        source_ids=_audit_source_ids_from_args(args),
    )
    audit_path = out_dir / _file_name("audit", version)
    write_parquet_deterministic(audit_df, audit_path, AUDIT_SCHEMA, AUDIT_PRIMARY_KEY)
    logger.info("audit -> %s (%d rows)", audit_path, len(audit_df))
    _maybe_upload(audit_path, "audit", version, args.bucket, args)

    # Annotations
    annotations_df = build_annotations(
        audit_df, bucket=args.bucket, prefix=args.prefix, workers=args.workers
    )
    annotations_path = out_dir / _file_name("annotations", version)
    write_parquet_deterministic(
        annotations_df, annotations_path, ANNOTATIONS_SCHEMA, ANNOTATIONS_PRIMARY_KEY
    )
    logger.info("annotations -> %s (%d rows)", annotations_path, len(annotations_df))
    _maybe_upload(annotations_path, "annotations", version, args.bucket, args)

    # Images
    images_path = out_dir / _file_name("images", version)
    images_df = build_images(
        annotations_df,
        bucket=args.bucket,
        prefix=args.prefix,
        workers=max(args.workers, DEFAULT_IMAGE_WORKERS),
        resize_threshold=args.resize_threshold,
        checkpoint_path=images_path,
        checkpoint_every=args.checkpoint_every,
        limit=args.limit_images,
    )
    write_parquet_deterministic(images_df, images_path, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)
    for part in sorted(images_path.parent.glob(f"{images_path.stem}.part_*.parquet")):
        with contextlib.suppress(OSError):
            part.unlink()
    logger.info("images -> %s (%d rows)", images_path, len(images_df))
    _maybe_upload(images_path, "images", version, args.bucket, args)
    return 0


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bucket", default=get_bucket(), help="S3 bucket (env: MERMAID_CORALNET_BUCKET)"
    )
    p.add_argument(
        "--prefix", default=get_prefix(), help="S3 prefix (env: MERMAID_CORALNET_PREFIX)"
    )
    p.add_argument("--output-dir", default=None, help="Local output directory")
    p.add_argument(
        "--upload-to-s3", action="store_true", help="Upload result(s) to s3 after writing locally"
    )
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent S3 workers")
    p.add_argument(
        "--version-tag", default=None, help="Override version tag (default: YYYYMMDD_<sha>)"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coralnet-etl",
        description="Productionized ETL for CoralNet annotations + image metadata.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Walk S3 and emit a source audit parquet")
    _add_common_flags(p_audit)
    p_audit.add_argument(
        "--limit-sources", type=int, default=None, help="Audit only N sources (dev)"
    )
    p_audit.add_argument(
        "--source-ids",
        default=None,
        help="Comma-separated source ids (audit these only; no S3 walk)",
    )
    p_audit.add_argument(
        "--source-ids-file",
        type=Path,
        default=None,
        help="Parquet or CSV with source_id column (audit those ids only)",
    )
    p_audit.set_defaults(func=_cmd_audit)

    p_anns = sub.add_parser("build-annotations", help="Merge per-source CSVs into one parquet")
    _add_common_flags(p_anns)
    p_anns.add_argument("--audit", type=Path, required=True, help="Path to audit parquet")
    p_anns.set_defaults(func=_cmd_build_annotations)

    p_imgs = sub.add_parser(
        "build-images", help="Scan JPEG headers and emit image metadata parquet"
    )
    _add_common_flags(p_imgs)
    p_imgs.add_argument(
        "--annotations", type=Path, required=True, help="Path to annotations parquet"
    )
    p_imgs.add_argument("--resize-threshold", type=int, default=DEFAULT_RESIZE_THRESHOLD)
    p_imgs.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    p_imgs.add_argument(
        "--limit-images", type=int, default=None, help="Process only N unique images (dev)"
    )
    p_imgs.set_defaults(func=_cmd_build_images)

    p_all = sub.add_parser("all", help="Run audit, build-annotations, and build-images in sequence")
    _add_common_flags(p_all)
    p_all.add_argument("--limit-sources", type=int, default=None, help="Audit only N sources (dev)")
    p_all.add_argument("--source-ids", default=None, help="Comma-separated source ids")
    p_all.add_argument("--source-ids-file", type=Path, default=None, help="Parquet or CSV ids")
    p_all.add_argument(
        "--limit-images", type=int, default=None, help="Process only N unique images (dev)"
    )
    p_all.add_argument("--resize-threshold", type=int, default=DEFAULT_RESIZE_THRESHOLD)
    p_all.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    p_all.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
