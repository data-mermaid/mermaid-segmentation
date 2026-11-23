"""Argparse registration and logging setup for ``segdata build coralnet``, free of heavy imports."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...common.store import add_store_argument

_DEFAULT_OUTPUT_DIR = Path("outputs/coralnet-dataset")


def _stage(name: str) -> Callable[[argparse.Namespace], int]:
    """Return a handler that runs ``build._cmd_<name>``, importing ``build`` on first call."""

    def run(args: argparse.Namespace) -> int:
        from . import build

        return int(getattr(build, f"_cmd_{name}")(args))

    return run


def raw_flag_errors(args: argparse.Namespace) -> list[str]:
    """Problems with the ``raw`` flags; empty when they are right.

    A directory ``--store`` needs ``--source-ids``, ``--store s3`` takes none, and the parent
    pin's two flags come together.
    """
    errors: list[str] = []
    on_s3 = str(args.store) == "s3"
    if not on_s3 and args.source_ids is None:
        errors.append("a directory --store needs --source-ids")
    if on_s3 and args.source_ids is not None:
        errors.append("--store s3 takes no --source-ids")
    if (args.parent_version is None) != (args.parent_manifest_sha256 is None):
        errors.append("--parent-version and --parent-manifest-sha256 go together")
    return errors


def _run_raw(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Check the flags, then run ``raw.run_raw``, importing ``raw`` on first call."""
    errors = raw_flag_errors(args)
    if errors:
        parser.error("; ".join(errors))
    from . import raw

    return raw.run_raw(args)


def _parse_source_ids(raw: str) -> list[int]:
    """Parse ``--source-ids 1,2,3`` into a list of ints.

    Raises:
        argparse.ArgumentTypeError: If no source id is given.
    """
    source_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not source_ids:
        raise argparse.ArgumentTypeError("expected at least one source id")
    return source_ids


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--version-tag", default=None, help="Override version tag (default: YYYYMMDD_<sha>)"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files that already exist under etl-outputs/coralnet/<version>/",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Local output directory; files go under <output-dir>/<version>/",
    )
    p.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload outputs to etl-outputs/coralnet/<version>/ after writing them locally",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent S3 workers (default: MERMAID_CORALNET_ETL_WORKERS, else 16)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def _add_source_ids_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--source-ids",
        type=_parse_source_ids,
        default=None,
        help="Comma-separated source ids (default: every s<id>/ folder under the prefix)",
    )


def add_subcommands(sub: Any) -> None:
    """Register the ``coralnet`` command and its six stage subcommands on ``sub``."""
    coralnet = sub.add_parser(
        "coralnet",
        help="Build the CoralNet annotations, images and training-manifest parquets",
        description="Build the CoralNet annotations, images and training-manifest parquets.",
    )
    stages = coralnet.add_subparsers(dest="command", required=True)

    p_anns = stages.add_parser("build-annotations", help="Merge per-source CSVs into one parquet")
    _add_common_flags(p_anns)
    _add_source_ids_flag(p_anns)
    p_anns.set_defaults(func=_stage("build_annotations"))

    p_imgs = stages.add_parser("build-images", help="Scan JPEG headers into an images parquet")
    _add_common_flags(p_imgs)
    p_imgs.add_argument("--annotations", type=Path, required=True, help="Annotations parquet")
    p_imgs.set_defaults(func=_stage("build_images"))

    p_resize = stages.add_parser("resize", help="Resize and upload images missing on S3")
    _add_common_flags(p_resize)
    p_resize.add_argument("--images", type=Path, required=True, help="Images parquet")
    p_resize.set_defaults(func=_stage("resize"))

    p_manifest = stages.add_parser("build-manifest", help="Build the training manifest parquet")
    _add_common_flags(p_manifest)
    p_manifest.add_argument("--annotations", type=Path, required=True, help="Annotations parquet")
    p_manifest.add_argument("--images", type=Path, required=True, help="Images parquet")
    p_manifest.set_defaults(func=_stage("build_manifest"))

    p_all = stages.add_parser("all", help="Run every stage in order with one version tag")
    _add_common_flags(p_all)
    _add_source_ids_flag(p_all)
    p_all.set_defaults(func=_stage("all"))

    p_raw = stages.add_parser(
        "raw", help="Publish a frozen build run as a coralnet-raw dataset version"
    )
    add_store_argument(p_raw)
    p_raw.add_argument(
        "--build-run",
        default="20260623_nogit",
        help="Build run under etl-outputs/coralnet/ to publish (default: %(default)s)",
    )
    _add_source_ids_flag(p_raw)
    p_raw.add_argument("--parent-version", default=None, help="coralnet-raw version to reuse")
    p_raw.add_argument(
        "--parent-manifest-sha256", default=None, help="Manifest sha256 of --parent-version"
    )
    p_raw.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent S3 workers (default: MERMAID_CORALNET_ETL_WORKERS, else 16)",
    )
    p_raw.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_raw.set_defaults(func=lambda args: _run_raw(p_raw, args))


def configure_logging(verbose: bool) -> None:
    """Log to stdout at INFO, or DEBUG when ``verbose``."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
