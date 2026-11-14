#!/usr/bin/env python3
"""Write the concept channels to remove from training for too little support.

Rules (see ``mermaidseg.dataset_reconciliation.concept_removal``), on True annotations:
images >= 100 or >= 50 outside CoralNet; annotations >= 500 with each dense mask counting 50;
any non-CoralNet dataset or >= 3 CoralNet sources. A concept failing any rule is removed.

Counts come from ``--label-counts``, the output of ``scripts/train_label_counts.py`` on
``--data-config`` (its train splits). CoralNet images get their source from the CoralNet train
split of the same config. Concept values come from the frozen concept schema training uses.

    python scripts/train_label_counts.py --data-config configs/data_config_trainval.yaml \
        --output dataset_stats/label_counts_trainval.parquet
    python scripts/build_concept_removal.py \
        --data-config configs/data_config_trainval.yaml \
        --label-counts dataset_stats/label_counts_trainval.parquet \
        --out configs/concept_removals/min_coverage_trainval.csv
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_concept_source_mask import REPO_ROOT, _git_commit, _rel  # noqa: E402

from mermaidseg.dataset_reconciliation.concept_removal import (  # noqa: E402
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
from mermaidseg.datasets import build_datasets, setup_local_cache  # noqa: E402
from mermaidseg.datasets.factory import configured_source_names  # noqa: E402
from mermaidseg.io import setup_config  # noqa: E402
from mermaidseg.run_setup import DEFAULT_SCHEMA_CONFIG  # noqa: E402


def coralnet_image_sources(data_config: Path) -> pd.Series:
    """CoralNet train ``image_id`` -> ``source_id`` for ``data_config``."""
    cfg = setup_config({"data": str(data_config)})
    setup_local_cache(cfg.data)
    ds = build_datasets(cfg.data, names=["coralnet"], splits=["train"])[("coralnet", "train")]
    images = ds.df_images[["image_id", "source_id"]].astype(str).drop_duplicates("image_id")
    return images.set_index("image_id")["source_id"]


def main() -> None:
    rule = CoverageRule()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--label-counts", type=Path, required=True)
    p.add_argument("--concepts", type=Path, default=Path(DEFAULT_CLASS_TO_CONCEPTS_CSV))
    p.add_argument("--schema-config", type=Path, default=Path(DEFAULT_SCHEMA_CONFIG))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--coverage-out", type=Path, default=None, help="optional CSV with every channel")
    for name, value in rule.as_dict().items():
        p.add_argument(f"--{name.replace('_', '-')}", type=int, default=value)
    args = p.parse_args()
    rule = CoverageRule(**{name: getattr(args, name) for name in rule.as_dict()})

    counts = pd.read_parquet(args.label_counts)
    counts["image_id"] = counts["image_id"].astype(str)
    img2src = coralnet_image_sources(args.data_config)
    is_cn = counts["dataset"] == "coralnet"
    counts["source"] = counts["dataset"]
    counts.loc[is_cn, "source"] = counts.loc[is_cn, "image_id"].map(img2src)
    missing = int(counts.loc[is_cn, "source"].isna().sum())
    if missing:
        raise SystemExit(
            f"{missing} CoralNet count rows have no source in {args.data_config}; "
            "were the label counts built from the same data config?"
        )

    schema_sources = configured_source_names(setup_config({"data": str(args.schema_config)}).data)
    schema = ConceptSchema.from_csv(args.concepts, sources=schema_sources)
    coverage = concept_coverage(counts, schema, rule)
    removed = removed_concepts(coverage)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    removed.to_csv(args.out, index=False)
    if args.coverage_out is not None:
        coverage.to_csv(args.coverage_out)

    by_kind = coverage.groupby("kind")["removed"].agg(["size", "sum"])
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
        "label_counts": _rel(args.label_counts),
        "label_counts_sha1": file_sha1(args.label_counts),
        "datasets_counted": sorted(counts["dataset"].unique()),
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
