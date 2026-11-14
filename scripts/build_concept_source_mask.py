#!/usr/bin/env python3
"""Write the CoralNet source x binary concept pairs to mask in training.

A pair is masked when the source has False points for the concept and no True points
(``only_false``). Points come from the CoralNet train split of ``--data-config``, built exactly as
training builds it (source whitelist, image blacklist, annotation overrides); concept values come
from the frozen concept schema training uses (``--concepts`` restricted to the datasets of
``--schema-config``). No images are read.

Outputs ``--out`` (one row per masked pair: source_id, concept, n_true, n_false, n_not_given) and a
``.meta.yaml`` sidecar recording the inputs and their sha1s.

    python scripts/build_concept_source_mask.py \
        --out configs/concept_source_masks/coralnet_only_false.csv
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
from pathlib import Path

import yaml

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    MASK_COLUMNS,
    file_sha1,
    meta_path,
    only_false_pairs,
    source_concept_counts,
)
from mermaidseg.dataset_reconciliation.concepts import DEFAULT_CLASS_TO_CONCEPTS_CSV
from mermaidseg.datasets import build_datasets, setup_local_cache
from mermaidseg.datasets.factory import configured_source_names
from mermaidseg.io import setup_config
from mermaidseg.run_setup import DEFAULT_SCHEMA_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs/data_config.yaml")
    p.add_argument("--concepts", type=Path, default=Path(DEFAULT_CLASS_TO_CONCEPTS_CSV))
    p.add_argument("--schema-config", type=Path, default=Path(DEFAULT_SCHEMA_CONFIG))
    p.add_argument("--split", default="train")
    p.add_argument(
        "--out", type=Path, default=REPO_ROOT / "configs/concept_source_masks/coralnet_only_false.csv"
    )
    p.add_argument(
        "--counts-out",
        type=Path,
        default=None,
        help="optional CSV with the counts of every source x binary concept pair (masked or not)",
    )
    args = p.parse_args()

    cfg = setup_config({"data": str(args.data_config)})
    setup_local_cache(cfg.data)
    ds = build_datasets(cfg.data, names=["coralnet"], splits=[args.split])[("coralnet", args.split)]

    schema_sources = configured_source_names(setup_config({"data": str(args.schema_config)}).data)
    schema = ConceptSchema.from_csv(args.concepts, sources=schema_sources)

    counts = source_concept_counts(ds.df_annotations, schema, dataset_source=ds.SOURCE_NAME)
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
        "dataset": ds.SOURCE_NAME,
        "split": args.split,
        "data_config": _rel(args.data_config),
        "data_config_sha1": file_sha1(args.data_config),
        "concepts_csv": _rel(args.concepts),
        "concepts_csv_sha1": file_sha1(args.concepts),
        "schema_config": _rel(args.schema_config),
        "n_sources": n_sources,
        "n_binary_concepts": n_concepts,
        "n_pairs": n_sources * n_concepts,
        "n_masked": len(masked),
        "n_points": len(ds.df_annotations),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    meta_path(args.out).write_text(yaml.safe_dump(meta, sort_keys=False))

    print(
        f"{len(masked)} of {n_sources} sources x {n_concepts} binary concepts "
        f"= {n_sources * n_concepts} pairs are only_false"
    )
    print(f"wrote {_rel(args.out)} and {_rel(meta_path(args.out))}")


if __name__ == "__main__":
    main()
