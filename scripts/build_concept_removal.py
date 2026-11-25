#!/usr/bin/env python3
"""Write the concept channels to remove from training for too little support.

Rules (see ``mermaidseg.dataset_reconciliation.concept_removal``), on True annotations:
images >= 100 or >= 50 outside CoralNet; annotations >= 500 with each dense mask counting 50;
any non-CoralNet dataset or >= 3 CoralNet sources. A concept failing any rule is removed.

Counts come from the ``stats/label_counts.parquet`` of every dataset with a train block in
``--data-config``, read from the synced ``--cache-dir`` and restricted to the train block's groups.
A label is its concept label key, lower-cased: the ``labels.parquet`` name, or for MERMAID its
``mermaid_benthic_attribute``; ``unlabeled`` labels and label id 0 are not counted. A points
dataset whose train block names no annotator must have a single annotator. A CoralNet image's source is its group ``s<source id>``. Concept values come from the
frozen concept schema training uses.

    python scripts/build_concept_removal.py --cache-dir /path/to/cache \
        --out configs/concept_removals/min_coverage_trainval.csv
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Collection
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_concept_source_mask import _git_commit, _rel, concept_keys  # noqa: E402

from mermaidseg.data.config import DataSection, entry_split_groups  # noqa: E402
from mermaidseg.data.factory import open_dataset_version  # noqa: E402
from mermaidseg.dataset_reconciliation.concept_removal import (  # noqa: E402
    CORALNET,
    CoverageRule,
    concept_coverage,
    removed_concepts,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema  # noqa: E402
from mermaidseg.dataset_reconciliation.concept_source_mask import (  # noqa: E402
    file_sha1,
    meta_path,
)
from mermaidseg.dataset_reconciliation.concepts import DEFAULT_CLASS_TO_CONCEPTS_CSV  # noqa: E402
from mermaidseg.io import load_section  # noqa: E402
from mermaidseg.run_setup import DEFAULT_SCHEMA_CONFIG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["dataset", "image_id", "label", "n", "source"]
# Labels that mark unannotated points or pixels, never a class.
IGNORE_LABEL_NAMES: frozenset[str] = frozenset({"unlabeled", "unlabeled / unidentified"})


def train_label_counts(
    dataset: str,
    label_counts: pd.DataFrame,
    images: pd.DataFrame,
    labels: pd.DataFrame,
    groups: Collection[str] | None,
) -> pd.DataFrame:
    """Return one dataset's per-image label counts over ``groups``, in ``COLUMNS``.

    Args:
        dataset: Dataset name, written to ``dataset`` (and to ``source`` outside CoralNet).
        label_counts: ``stats/label_counts.parquet`` rows (``image_id``, ``label_id``, ``n``).
        images: ``images.parquet`` rows (``image_id``, ``group``).
        labels: ``labels.parquet`` rows (``label_id``, ``name``).
        groups: Groups to count; ``None`` counts all.

    Raises:
        ValueError: A group is not in ``images``, or a counted label id has no name.
    """
    if groups is not None:
        unknown = sorted(set(groups) - set(images["group"]))
        if unknown:
            raise ValueError(f"{dataset}: groups not in the dataset version: {unknown}")
        images = images[images["group"].isin(list(groups))]
    group_of = images.set_index("image_id")["group"]
    counts = label_counts[
        label_counts["image_id"].isin(group_of.index) & (label_counts["label_id"] != 0)
    ]
    names = counts["label_id"].map(labels.set_index("label_id")["name"])
    unnamed = sorted(set(counts.loc[names.isna(), "label_id"]))
    if unnamed:
        raise ValueError(f"{dataset}: label ids missing from labels.parquet: {unnamed}")
    ignored = names.astype(str).str.strip().str.lower().isin(IGNORE_LABEL_NAMES).to_numpy()
    label = counts["label_id"].map(concept_keys(dataset, labels))
    out = pd.DataFrame(
        {
            "dataset": dataset,
            "image_id": counts["image_id"].astype(str).to_numpy(),
            "label": label.astype(str).str.strip().str.lower().to_numpy(),
            "n": counts["n"].to_numpy(),
        }
    )
    out["source"] = (
        counts["image_id"].map(group_of).astype(str).str.removeprefix("s").to_numpy()
        if dataset == CORALNET
        else dataset
    )
    return out[~ignored].reset_index(drop=True)[COLUMNS]


def require_single_annotator(dataset: str, annotators: Collection[str]) -> None:
    """Raise if ``annotators`` (a points dataset's ``annotator`` column) holds several names.

    ``stats/label_counts.parquet`` counts every annotator's labels and cannot be filtered by
    annotator, so a train block, whether or not it names annotators, would count each position
    once per annotator.
    """
    found = sorted(set(annotators))
    if len(found) > 1:
        raise ValueError(
            f"{dataset}: several annotators {found} in points.parquet; "
            "label counts cannot be filtered by annotator"
        )


def section_train_label_counts(section: DataSection, cache_dir: Path) -> pd.DataFrame:
    """Concatenate :func:`train_label_counts` over every dataset with a train block.

    Raises:
        ValueError: No dataset has a train block, or a points dataset has several annotators.
    """
    parts = []
    for name, entry in section.datasets.items():
        if entry.train is None:
            continue
        version = open_dataset_version(section, name, cache_dir)
        if version.points is not None:
            require_single_annotator(name, version.points.column("annotator").to_pylist())
        parts.append(
            train_label_counts(
                name,
                version.label_counts.to_pandas(),
                version.images.to_pandas(),
                version.labels.to_pandas(),
                entry_split_groups(entry, "train", REPO_ROOT),
            )
        )
    if not parts:
        raise ValueError("no dataset has a train block to count")
    return pd.concat(parts, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and build the concept removal file."""
    rule = CoverageRule()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs/data_config.yaml")
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory scripts/sync_data.py synced --data-config into",
    )
    p.add_argument("--concepts", type=Path, default=Path(DEFAULT_CLASS_TO_CONCEPTS_CSV))
    p.add_argument("--schema-config", type=Path, default=Path(DEFAULT_SCHEMA_CONFIG))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--coverage-out", type=Path, default=None, help="optional CSV with every channel"
    )
    for name, value in rule.as_dict().items():
        p.add_argument(f"--{name.replace('_', '-')}", type=int, default=value)
    args = p.parse_args(argv)
    rule = CoverageRule(**{name: getattr(args, name) for name in rule.as_dict()})

    section = DataSection.model_validate(load_section(args.data_config, "data"))
    counts = section_train_label_counts(section, args.cache_dir)

    schema_sources = set(load_section(args.schema_config, "data")["datasets"])
    schema = ConceptSchema.from_csv(args.concepts, sources=schema_sources)
    coverage = concept_coverage(counts, schema, rule)
    removed = removed_concepts(coverage)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    removed.to_csv(args.out, index=False)
    if args.coverage_out is not None:
        coverage.to_csv(args.coverage_out)

    by_kind = coverage.groupby("kind")["removed"].agg(["size", "sum"])
    counted = sorted(counts["dataset"].unique())
    meta = {
        "rule": rule.as_dict(),
        "rule_description": (
            f"remove if (images < {rule.min_images} and non-CoralNet images < "
            f"{rule.min_noncoralnet_images}) or annotations < {rule.min_annotations} "
            f"(dense mask = {rule.dense_annotations_per_mask}) or (no non-CoralNet dataset and "
            f"CoralNet sources < {rule.min_coralnet_sources})"
        ),
        "data_config": _rel(args.data_config),
        "data_config_sha1": file_sha1(args.data_config),
        "dataset_versions": {name: section.datasets[name].version for name in counted},
        "datasets_counted": counted,
        "concepts_csv": _rel(args.concepts),
        "concepts_csv_sha1": file_sha1(args.concepts),
        "schema_config": _rel(args.schema_config),
        "n_channels": int(len(coverage)),
        "n_removed": int(len(removed)),
        "removed_by_kind": {k: int(v) for k, v in by_kind["sum"].items()},
        "channels_by_kind": {k: int(v) for k, v in by_kind["size"].items()},
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    meta_path(args.out).write_text(yaml.safe_dump(meta, sort_keys=False))

    taxonomic = coverage["kind"] != "binary"
    print(
        f"removed {int(coverage.loc[taxonomic, 'removed'].sum())} of {int(taxonomic.sum())} taxonomic "
        f"and {int(coverage.loc[~taxonomic, 'removed'].sum())} of {int((~taxonomic).sum())} binary "
        "concept channels"
    )
    print(f"wrote {_rel(args.out)} and {_rel(meta_path(args.out))}")


if __name__ == "__main__":
    main()
