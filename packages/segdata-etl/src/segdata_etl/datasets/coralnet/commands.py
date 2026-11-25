"""Argparse registration and logging setup for ``segdata build coralnet``, free of heavy imports."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...common.cli_flags import _pin_flags, half_pin_error

_DEFAULT_OUTPUT_DIR = Path("outputs/coralnet-dataset")


def _stage(name: str) -> Callable[[argparse.Namespace], int]:
    """Return a handler that runs ``build._cmd_<name>``, importing ``build`` on first call."""

    def run(args: argparse.Namespace) -> int:
        from . import build

        return int(getattr(build, f"_cmd_{name}")(args))

    return run


def raw_flag_errors(args: argparse.Namespace) -> list[str]:
    """Problems with the ``raw`` flags; empty when they are right.

    The parent pin's two flags come together.
    """
    pin_error = half_pin_error(args, "parent-")
    return [] if pin_error is None else [pin_error]


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


_LEGACY_SOURCE_IDS_HELP = (
    "Comma-separated source ids (default: every s<id>/ folder under the prefix)"
)
_RAW_SOURCE_IDS_HELP = (
    "Comma-separated source ids to stage (default: every source in the build run)"
)


def _add_source_ids_flag(
    p: argparse.ArgumentParser, help_text: str = _LEGACY_SOURCE_IDS_HELP
) -> None:
    p.add_argument("--source-ids", type=_parse_source_ids, default=None, help=help_text)


def add_subcommands(sub: Any) -> None:
    """Register the ``coralnet`` command and its stage subcommands on ``sub``."""
    coralnet = sub.add_parser(
        "coralnet",
        help="Build the CoralNet annotations and images parquets",
        description="Build the CoralNet annotations and images parquets.",
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

    p_all = stages.add_parser("all", help="Run every stage in order with one version tag")
    _add_common_flags(p_all)
    _add_source_ids_flag(p_all)
    p_all.set_defaults(func=_stage("all"))

    p_raw = stages.add_parser(
        "raw", help="Stage a frozen build run as a coralnet-raw dataset version"
    )
    p_raw.add_argument(
        "--stage-dir", type=Path, required=True, help="Local directory the version is staged in"
    )
    p_raw.add_argument(
        "--build-run",
        default="20260623_nogit",
        help="Build run under etl-outputs/coralnet/ to stage (default: %(default)s)",
    )
    _add_source_ids_flag(p_raw, _RAW_SOURCE_IDS_HELP)
    _pin_flags(p_raw, "parent-", "the coralnet-raw parent to reuse", required=False)
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
