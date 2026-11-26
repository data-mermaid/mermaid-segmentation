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
from pathlib import Path

import yaml

from mermaidseg.data.config import DataSection
from mermaidseg.dataset_reconciliation.artifact_meta import (
    concept_sidecar_meta,
    meta_path,
    repo_relative,
)
from mermaidseg.dataset_reconciliation.concept_removal import (
    CoverageRule,
    concept_coverage,
    removed_concepts,
    section_train_label_counts,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import DEFAULT_CLASS_TO_CONCEPTS_CSV
from mermaidseg.io import REPO_ROOT, load_section
from mermaidseg.lineage import git_state
from mermaidseg.run_setup import DEFAULT_SCHEMA_CONFIG


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
    git = git_state()

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
        "dataset_versions": {name: section.datasets[name].version for name in counted},
        "datasets_counted": counted,
        "n_channels": int(len(coverage)),
        "n_removed": int(len(removed)),
        "removed_by_kind": {k: int(v) for k, v in by_kind["sum"].items()},
        "channels_by_kind": {k: int(v) for k, v in by_kind["size"].items()},
        **concept_sidecar_meta(args.data_config, args.concepts, args.schema_config, git),
    }
    meta_path(args.out).write_text(yaml.safe_dump(meta, sort_keys=False))

    taxonomic = coverage["kind"] != "binary"
    print(
        f"removed {int(coverage.loc[taxonomic, 'removed'].sum())} of {int(taxonomic.sum())} taxonomic "
        f"and {int(coverage.loc[~taxonomic, 'removed'].sum())} of {int((~taxonomic).sum())} binary "
        "concept channels"
    )
    print(f"wrote {repo_relative(args.out)} and {repo_relative(meta_path(args.out))}")


if __name__ == "__main__":
    main()
