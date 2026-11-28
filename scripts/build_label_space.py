#!/usr/bin/env python3
"""Write the label space of a model: its classes, concept channels and per-source-label resolution.

Reads the tables of every dataset that ``--data-config`` pins, and its taxonomy, from a tables-only
``--cache-dir`` (filled by ``scripts/sync_data.py --tables-only``); no image is read. Writes
``classes.csv``, ``concepts.csv``, ``source_labels.csv`` and ``label_space.meta.yaml`` to ``--out``.

The build refuses, and writes nothing, when a label that trains a class has no concept row, or a
label's concept values disagree with its class, unless ``--acknowledge`` (columns dataset,
native_name, kind, reason; default ``<out>/acknowledged.csv``) lists it with a reason.

    python scripts/build_label_space.py --cache-dir /path/to/cache \
        --training-config configs/training_config_cbm_final.yaml \
        --out configs/label_spaces/cbm_final
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mermaidseg.data.config import DataSection
from mermaidseg.dataset_reconciliation.artifact_meta import repo_relative
from mermaidseg.dataset_reconciliation.label_space import (
    CLASSES_CSV,
    CONCEPTS_CSV,
    META_YAML,
    SOURCE_LABELS_CSV,
    write_label_space,
)
from mermaidseg.dataset_reconciliation.label_space_build import (
    build_label_space,
    summarize_label_space,
)
from mermaidseg.io import REPO_ROOT, load_section, resolve_repo_path
from mermaidseg.run_setup import DEFAULT_SCHEMA_CONFIG
from segdata_spec.git import clean_git_state


def _tracked_inputs(args: argparse.Namespace, acknowledged: Path) -> tuple[Path, ...]:
    """The repo files the build reads or the meta pins.

    These are the configs, concept table, removal list, mask, acknowledged file and each dataset's
    split file. An empty optional key, an entry without a split file and a missing ``acknowledged`` file name
    nothing.
    """
    training = load_section(args.training_config, "training")
    data = DataSection.model_validate(load_section(args.data_config, "data"))
    named = [training["concept_mapping_path"], training["concept_removal"]]
    named.append(training["concept_source_mask"])
    return (
        args.data_config,
        args.training_config,
        args.schema_config,
        *(resolve_repo_path(value) for value in named if value),
        *((acknowledged,) if acknowledged.is_file() else ()),
        *(REPO_ROOT / e.split_file for e in data.datasets.values() if e.split_file is not None),
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and build the label space directory."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs/data_config.yaml")
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--schema-config", type=Path, default=Path(DEFAULT_SCHEMA_CONFIG))
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Cache that scripts/sync_data.py --tables-only (or a full sync) synced --data-config into",
    )
    p.add_argument("--out", type=Path, required=True, help="Label space directory to write")
    p.add_argument("--acknowledge", type=Path, default=None)
    args = p.parse_args(argv)
    acknowledged = args.acknowledge or args.out / "acknowledged.csv"

    git = clean_git_state(
        Path(__file__).resolve().parent, inputs=_tracked_inputs(args, acknowledged)
    )
    space = build_label_space(
        args.data_config,
        args.training_config,
        args.schema_config,
        args.cache_dir,
        acknowledged,
        git.sha,
    )
    write_label_space(args.out, space)

    print("\n".join(summarize_label_space(space)))
    files = [CLASSES_CSV, CONCEPTS_CSV, SOURCE_LABELS_CSV, META_YAML]
    print("wrote " + ", ".join(repo_relative(args.out / name) for name in files))


if __name__ == "__main__":
    main()
