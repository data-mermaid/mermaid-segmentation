"""Command-line entry point: one subcommand per dataset pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import boto3
from botocore.config import Config

from . import benthos_yuval, catlin_seaview, moorea_labeled_corals, pacific_labeled_corals

DEFAULT_BUCKET = "dev-datamermaid-sm-sources"
DEFAULT_WORKERS = 16
WORKDIR_ROOT = Path("outputs") / "external-datasets"

_BOTO_CONFIG = Config(retries={"max_attempts": 10, "mode": "adaptive"})

# Each dataset module exports ``add_arguments(parser, *, workdir)`` and ``run(args, client) -> int``; its
# module name is the subcommand. Adding a dataset is one entry here.
DATASET_MODULES: tuple[ModuleType, ...] = (
    catlin_seaview,
    moorea_labeled_corals,
    pacific_labeled_corals,
    benthos_yuval,
)


def default_workdir(dataset: str) -> Path:
    """Return the default local working directory for ``dataset``."""
    return WORKDIR_ROOT / dataset


def make_s3_client() -> Any:
    """Build a boto3 S3 client with adaptive retries for long-running jobs."""
    return boto3.client("s3", config=_BOTO_CONFIG)


def _shared_parser(dataset: str) -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--bucket", default=DEFAULT_BUCKET)
    parent.add_argument("--prefix", default=f"external_validation_datasets/{dataset}")
    parent.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parent.add_argument("--dry-run", action="store_true")
    parent.add_argument(
        "--no-skip-existing-s3", dest="skip_existing_s3", action="store_false", default=True
    )
    parent.add_argument("--overwrite", action="store_true")
    parent.add_argument("-v", "--verbose", action="store_true")
    return parent


def build_parser(modules: Sequence[ModuleType] = DATASET_MODULES) -> argparse.ArgumentParser:
    """Build the top-level parser with one subcommand per dataset module."""
    parser = argparse.ArgumentParser(prog="external-datasets", description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    for module in modules:
        name = module.__name__.rsplit(".", 1)[-1]
        dataset_parser = sub.add_parser(name, parents=[_shared_parser(name)])
        module.add_arguments(dataset_parser, workdir=default_workdir(name))
        dataset_parser.set_defaults(run=module.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, build the S3 client and run the chosen dataset pipeline."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.run(args, make_s3_client()))
