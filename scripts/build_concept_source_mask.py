#!/usr/bin/env python3
"""Write the CoralNet source x binary concept pairs to mask in training.

A pair is masked when the source has False points for the concept and no True points
(``only_false``). Points come from the ``points.parquet`` of the ``coralnet`` version that
``--data-config`` pins (default: the final-run mixture ``configs/data_config.yaml``), read from the
synced ``--cache-dir``, over the groups of its ``--split`` block and selected by that block's
``annotators`` as training selects them; an image's source is its group ``s<source id>``. Concept values come from the frozen concept schema training uses (``--concepts``
restricted to the datasets of ``--schema-config``). No images are read.

Outputs ``--out`` (one row per masked pair: source_id, concept, n_true, n_false, n_not_given) and a
``.meta.yaml`` sidecar recording the inputs, their sha1s and the counted sources (``source_ids``).
Training refuses a CoralNet source that is not in ``source_ids``, so rebuild the mask whenever the
CoralNet train sources of the run's data config change.

    python scripts/build_concept_source_mask.py --cache-dir /path/to/cache \
        --out configs/concept_source_masks/coralnet_only_false.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mermaidseg.data.config import DataSection, Split
from mermaidseg.dataset_reconciliation.artifact_meta import (
    concept_sidecar_meta,
    meta_path,
    repo_relative,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    MASK_COLUMNS,
    _source_sort_key,
    only_false_pairs,
    section_point_annotations,
    source_concept_counts,
)
from mermaidseg.dataset_reconciliation.concepts import DEFAULT_CLASS_TO_CONCEPTS_CSV
from mermaidseg.io import REPO_ROOT, load_section
from mermaidseg.run_setup import DEFAULT_SCHEMA_CONFIG
from segdata_spec.git import clean_git_state

DATASET = "coralnet"


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and build the concept source mask file."""
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
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "configs/concept_source_masks/coralnet_only_false.csv",
    )
    p.add_argument(
        "--counts-out",
        type=Path,
        default=None,
        help="optional CSV with the counts of every source x binary concept pair (masked or not)",
    )
    args = p.parse_args(argv)
    split: Split = args.split

    git = clean_git_state(
        Path(__file__).resolve().parent, inputs=(args.data_config, args.schema_config)
    )

    section = DataSection.model_validate(load_section(args.data_config, "data"))
    annotations, dataset_version = section_point_annotations(
        section, args.cache_dir, split, DATASET
    )

    schema_sources = set(load_section(args.schema_config, "data")["datasets"])
    schema = ConceptSchema.from_csv(args.concepts, sources=schema_sources)

    counts = source_concept_counts(annotations, schema, dataset_source=DATASET)
    masked = only_false_pairs(counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    masked[MASK_COLUMNS].to_csv(args.out, index=False)
    if args.counts_out is not None:
        counts.to_csv(args.counts_out, index=False)

    n_sources = int(counts["source_id"].nunique())
    n_concepts = int(counts["concept"].nunique())
    meta = {
        "rule": "only_false",
        "rule_description": "source has n_false > 0 and n_true == 0 for the binary concept",
        "dataset": DATASET,
        "dataset_version": dataset_version,
        "split": split,
        **concept_sidecar_meta(args.data_config, args.concepts, args.schema_config, git.sha),
        "n_sources": n_sources,
        "n_binary_concepts": n_concepts,
        "n_pairs": n_sources * n_concepts,
        "n_masked": len(masked),
        "n_points": len(annotations),
        "source_ids": sorted({str(s) for s in counts["source_id"]}, key=_source_sort_key),
    }
    meta_path(args.out).write_text(yaml.safe_dump(meta, sort_keys=False))

    print(
        f"{len(masked)} of {n_sources} sources x {n_concepts} binary concepts "
        f"= {n_sources * n_concepts} pairs are only_false"
    )
    print(f"wrote {repo_relative(args.out)} and {repo_relative(meta_path(args.out))}")


if __name__ == "__main__":
    main()
