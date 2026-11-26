"""``segdata`` console script: ``build``, ``ingest``, ``taxonomy``, ``split``, ``curate``, ``promote`` and ``features``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any

from segdata_spec import DATASET_BUCKET

from . import features, taxonomy
from . import promote as promote_module
from .common.cli_flags import pin_flags
from .common.release import INGEST_BUCKET, ingest_prefix
from .common.s3_objects import make_s3_client
from .common.stage import DEFAULT_CACHE_DIR
from .datasets import benthos_yuval as benthos_yuval_build
from .datasets import catlin_seaview as catlin_seaview_build
from .datasets import coralscapes_v2, mermaid, ucsd_mosaics
from .datasets import moorea_labeled_corals as moorea_labeled_corals_build
from .datasets import pacific_labeled_corals as pacific_labeled_corals_build
from .datasets.coralnet import commands as coralnet_commands
from .ingest import (
    benthos_yuval,
    catlin_seaview,
    moorea_labeled_corals,
    pacific_labeled_corals,
)
from .splits import builder as split_builder
from .splits import leakage as split_leakage
from .splits.builder import SplitParams

DEFAULT_WORKERS = 16
WORKDIR_ROOT = Path("outputs") / "external-datasets"

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


def _client_for(args: argparse.Namespace) -> Any:
    """Build the S3 client, pooled for ``--workers`` threads when the subcommand has the flag."""
    return make_s3_client(workers=getattr(args, "workers", None))


def _shared_parser(dataset: str) -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--bucket", default=INGEST_BUCKET)
    parent.add_argument("--prefix", default=ingest_prefix(dataset))
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


def _split_cache_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, required=True, help="Local segdata cache")
    parser.add_argument("--bucket", default=DATASET_BUCKET, help="Bucket holding the versions")
    parser.add_argument("-v", "--verbose", action="store_true")


def _split_build_flags(parser: argparse.ArgumentParser) -> None:
    pin_flags(parser, "", "the curated coralnet dataset")
    pin_flags(parser, "raw-", "the coralnet-raw parent")
    pin_flags(parser, "taxonomy-", "the mermaid-benthic taxonomy")
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
    build.set_defaults(func=lambda args: split_builder.run(args, _client_for(args)))
    leakage = split_sub.add_parser("leakage", help="Check a split file for cross-split images")
    leakage.add_argument("--split-file", type=Path, required=True)
    pin_flags(leakage, "", "the coralnet dataset")
    _split_cache_flags(leakage)
    leakage.add_argument("--out-dir", type=Path, default=None, help="Write the report CSVs here")
    leakage.set_defaults(func=lambda args: split_leakage.run(args, _client_for(args)))


# Each external builder module exports ``add_arguments``, ``run(args, client)`` and ``DATASET``.
_EXTERNAL_BUILDS: tuple[tuple[ModuleType, str], ...] = (
    (catlin_seaview_build, "Stage the ingested Catlin Seaview prefix as a dataset version"),
    (
        moorea_labeled_corals_build,
        "Stage the ingested Moorea Labeled Corals prefix as a dataset version",
    ),
    (
        pacific_labeled_corals_build,
        "Stage the ingested Pacific Labeled Corals prefix as a dataset version",
    ),
    (
        benthos_yuval_build,
        "Stage the ingested Benthos Yuval tiles and masks as a dense dataset version",
    ),
)


def _add_external_build(build_sub: Any, module: ModuleType, help_text: str) -> None:
    parser = build_sub.add_parser(module.DATASET, help=help_text, parents=[_uniform_parser()])
    module.add_arguments(parser, workdir=default_workdir(module.DATASET))
    parser.set_defaults(func=lambda args: module.run(args, _client_for(args)))


def _add_external_builds(build_sub: Any) -> None:
    for module, help_text in _EXTERNAL_BUILDS:
        _add_external_build(build_sub, module, help_text)


def _add_taxonomy_subcommand(top: Any) -> None:
    parser = top.add_parser("taxonomy", help="Fetch and stage the MERMAID benthic taxonomy")
    taxonomy.add_arguments(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(
        func=lambda args: taxonomy.run_taxonomy(args, source_client=_client_for(args))
    )


def _run_curate(args: argparse.Namespace) -> int:
    from .datasets.coralnet.curate import versions

    return versions.run(args, source_client=_client_for(args))


def _add_curate_subcommand(top: Any) -> None:
    curate = top.add_parser("curate", help="Stage a curated dataset version from its parents")
    curate_sub = curate.add_subparsers(dest="dataset", required=True)
    parser = curate_sub.add_parser("coralnet", help="Stage a curated coralnet version")
    pin_flags(parser, "raw-", "the coralnet-raw parent")
    pin_flags(parser, "taxonomy-", "the mermaid-benthic taxonomy")
    parser.add_argument(
        "--stage-dir", type=Path, required=True, help="Local directory the version is staged in"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local cache for the synced parent tables and taxonomy; the stage dir reads "
        "staged parents (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=_run_curate)


def _add_promote_subcommand(top: Any) -> None:
    promote = top.add_parser(
        "promote", help="Upload a staged dataset or taxonomy version, write-once"
    )
    promote.add_argument("--stage-dir", type=Path, required=True, help="Local stage directory")
    target = promote.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", help="Dataset name to promote")
    target.add_argument("--taxonomy", help="Taxonomy name to promote")
    promote.add_argument("--version", required=True)
    promote.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    promote.add_argument("-v", "--verbose", action="store_true")
    promote.set_defaults(run=_run_promote)


def _add_features_subcommand(top: Any) -> None:
    parser = top.add_parser("features", help="Attach feature sets to published dataset versions")
    features_sub = parser.add_subparsers(dest="command", required=True)
    attach = features_sub.add_parser(
        "attach", help="Copy existing feature vectors into a published version as a feature set"
    )
    features.add_arguments(attach)
    attach.set_defaults(run=features.run)


def _run_promote(args: argparse.Namespace, client: Any) -> int:
    if args.taxonomy is not None:
        promote_module.promote_taxonomy(
            client,
            stage_dir=args.stage_dir,
            name=args.taxonomy,
            version=args.version,
            workers=args.workers,
        )
        return 0
    promote_module.promote(
        client,
        stage_dir=args.stage_dir,
        dataset=args.dataset,
        version=args.version,
        workers=args.workers,
    )
    return 0


def build_parser(
    modules: Sequence[ModuleType] = DATASET_MODULES,
    uniform_modules: Sequence[ModuleType] = UNIFORM_MODULES,
) -> argparse.ArgumentParser:
    """Build the top-level parser: ``build``, one ``ingest`` entry per dataset, ``taxonomy``, ``split``, ``curate``, ``promote`` and ``features``."""
    parser = argparse.ArgumentParser(prog="segdata", description=__doc__)
    top = parser.add_subparsers(dest="action", required=True)
    build = top.add_parser("build", help="Build and publish a dataset")
    build_sub = build.add_subparsers(dest="dataset", required=True)
    coralnet_commands.add_subcommands(build_sub)
    _add_external_builds(build_sub)
    ingest = top.add_parser("ingest", help="Validate and publish an external dataset")
    ingest_sub = ingest.add_subparsers(dest="dataset", required=True)
    for module in modules:
        name = module.__name__.rsplit(".", 1)[-1]
        dataset_parser = ingest_sub.add_parser(name, parents=[_shared_parser(name)])
        module.add_arguments(dataset_parser, workdir=default_workdir(name))
        dataset_parser.set_defaults(run=module.run)
    for module in uniform_modules:
        name = module.__name__.rsplit(".", 1)[-1]
        uniform_parser = build_sub.add_parser(name, parents=[_uniform_parser()])
        module.add_arguments(uniform_parser, workdir=default_workdir(name))
        uniform_parser.set_defaults(run=module.run)
    _add_split_subcommands(top)
    _add_taxonomy_subcommand(top)
    _add_curate_subcommand(top)
    _add_promote_subcommand(top)
    _add_features_subcommand(top)
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
    return int(args.run(args, _client_for(args)))


if __name__ == "__main__":
    raise SystemExit(main())
