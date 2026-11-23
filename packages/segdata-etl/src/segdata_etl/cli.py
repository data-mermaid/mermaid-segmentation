"""``segdata`` console script: ``build coralnet <stage>``, ``build <external dataset>`` and ``split``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any

import boto3
from botocore.config import Config

from segdata_spec import DATASET_BUCKET

from .datasets import (
    benthos_yuval,
    catlin_seaview,
    coralscapes_v2,
    mermaid,
    moorea_labeled_corals,
    pacific_labeled_corals,
    ucsd_mosaics,
)
from .datasets.coralnet import commands as coralnet_commands
from .splits import builder as split_builder
from .splits import leakage as split_leakage
from .splits.builder import SplitParams

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

# Uniform-format builds stage a version under ``--stage-dir``; each module has the same contract as
# DATASET_MODULES. Adding one is one entry here.
UNIFORM_MODULES: tuple[ModuleType, ...] = (mermaid, coralscapes_v2, ucsd_mosaics)


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


def _uniform_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--stage-dir", type=Path, required=True, help="Local directory the version is staged in"
    )
    parent.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parent.add_argument("-v", "--verbose", action="store_true")
    return parent


def _pin_flags(parser: argparse.ArgumentParser, prefix: str, what: str) -> None:
    parser.add_argument(f"--{prefix}version", required=True, help=f"Version of {what}")
    parser.add_argument(
        f"--{prefix}manifest-sha256", required=True, help=f"Manifest sha256 of {what}"
    )


def _split_cache_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, required=True, help="Local segdata cache")
    parser.add_argument("--bucket", default=DATASET_BUCKET, help="Bucket holding the versions")
    parser.add_argument("-v", "--verbose", action="store_true")


def _split_build_flags(parser: argparse.ArgumentParser) -> None:
    _pin_flags(parser, "", "the curated coralnet dataset")
    _pin_flags(parser, "raw-", "the coralnet-raw parent")
    _pin_flags(parser, "taxonomy-", "the mermaid-benthic taxonomy")
    _split_cache_flags(parser)
    inputs = {
        "--near-dup-labels": "Near-duplicate pair labels CSV; dup / unsure pairs link sources",
        "--metadata": "Source metadata CSV: source_id, admins, latitude, longitude",
        "--test-sources": "Source ids whose groups go to test, one per line",
        "--exclude-sources": "Source ids left out entirely, one per line",
        "--training-config": "Training config; its training.class_subset is the class space",
    }
    for flag, help_text in inputs.items():
        parser.add_argument(flag, type=Path, required=True, help=help_text)
    parser.add_argument("--out-dir", type=Path, required=True)
    defaults = SplitParams()
    tuning_help = {
        "val_frac": "Share of images assigned to val",
        "test_frac": "Share of images assigned to test",
        "max_group_frac": "Largest group, as a share of the smaller eval split's image target, that may go to val or test",
        "max_id_gap": "Max source ID gap for two sources to count as one project",
        "max_km": "Max distance in km for two sources to count as one project",
        "min_class_points": "Min points a class needs to be counted in the class space",
        "min_eval_points": "Min points per class that val and test should hold",
        "w_coverage": "Annealing weight on class coverage of val and test",
        "w_size": "Annealing weight on split size deviation",
        "iters": "Annealing iterations per restart",
        "restarts": "Annealing restarts; the best run wins",
        "seed": "Random seed for annealing",
    }
    for field in fields(SplitParams):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(getattr(defaults, field.name)),
            default=getattr(defaults, field.name),
            help=tuning_help[field.name] + " (default: %(default)s)",
        )


def _add_split_subcommands(top: Any) -> None:
    split = top.add_parser("split", help="Build or check the CoralNet train/val/test split")
    split_sub = split.add_subparsers(dest="command", required=True)
    build = split_sub.add_parser("build", help="Build a CoralNet split into --out-dir")
    _split_build_flags(build)
    build.set_defaults(func=lambda args: split_builder.run(args, make_s3_client()))
    leakage = split_sub.add_parser("leakage", help="Check a split file for cross-split images")
    leakage.add_argument("--split-file", type=Path, required=True)
    _pin_flags(leakage, "", "the coralnet dataset")
    _split_cache_flags(leakage)
    leakage.add_argument("--out-dir", type=Path, default=None, help="Write the report CSVs here")
    leakage.set_defaults(func=lambda args: split_leakage.run(args, make_s3_client()))


def build_parser(
    modules: Sequence[ModuleType] = DATASET_MODULES,
    uniform_modules: Sequence[ModuleType] = UNIFORM_MODULES,
) -> argparse.ArgumentParser:
    """Build the top-level parser: ``build coralnet <stage>``, one ``build`` entry per dataset, and ``split``."""
    parser = argparse.ArgumentParser(prog="segdata", description=__doc__)
    top = parser.add_subparsers(dest="action", required=True)
    build = top.add_parser("build", help="Build and publish a dataset")
    build_sub = build.add_subparsers(dest="dataset", required=True)
    coralnet_commands.add_subcommands(build_sub)
    for module in modules:
        name = module.__name__.rsplit(".", 1)[-1]
        dataset_parser = build_sub.add_parser(name, parents=[_shared_parser(name)])
        module.add_arguments(dataset_parser, workdir=default_workdir(name))
        dataset_parser.set_defaults(run=module.run)
    for module in uniform_modules:
        name = module.__name__.rsplit(".", 1)[-1]
        uniform_parser = build_sub.add_parser(name, parents=[_uniform_parser()])
        module.add_arguments(uniform_parser, workdir=default_workdir(name))
        uniform_parser.set_defaults(run=module.run)
    _add_split_subcommands(top)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the chosen CoralNet stage or external dataset pipeline.

    Returns:
        The exit code of the chosen pipeline.
    """
    args = build_parser().parse_args(argv)
    if hasattr(args, "func"):
        coralnet_commands.configure_logging(args.verbose)
        return int(args.func(args))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.run(args, make_s3_client()))


if __name__ == "__main__":
    raise SystemExit(main())
