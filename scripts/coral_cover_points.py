#!/usr/bin/env python3
"""Per-image ground-truth hard coral counts for a CoralNet split, for cover estimates.

Each annotated point gets its concept row from the CoralNet class through ``class_to_concepts.csv``
(``build_concept_lookup``, as in ``evaluate.py``). A point is hard coral when its
``order__scleractinia`` entry is positive. Writes one row per image: source, image id, number of
points, hard coral points, and points whose order is not given (unmapped labels included), which a
cover estimate counts as not hard coral.

Usage:
    python scripts/coral_cover_points.py --split test \
        --out eval_out/coral_cover/coralnet_test_image_cover.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset  # noqa: E402
from mermaidseg.datasets.local_cache import setup_local_cache  # noqa: E402
from mermaidseg.evaluation.gt_mapping import build_concept_lookup  # noqa: E402

HARD_CORAL = "order__scleractinia"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--concept-id2name", type=Path,
                   default=_REPO_ROOT / "eval_out/scratch100k_hier/concept_id2name.json")
    p.add_argument("--sources", type=int, nargs="+", default=None,
                   help="CoralNet source ids (default: the split's whitelist_sources)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = yaml.safe_load(args.data_config.read_text())
    data = data.get("data", data)
    setup_local_cache(data)
    split = (data.get("coralnet") or {}).get(args.split) or {}
    base = CoralNetDataset(whitelist_sources=args.sources or split.get("whitelist_sources"),
                           image_blacklist=split.get("image_blacklist"),
                           annotation_overrides=split.get("annotation_overrides"),
                           transform=None, split=args.split)
    ann = base.df_annotations

    id2name = json.loads(args.concept_id2name.read_text())
    concept_names = [id2name[k] for k in sorted(id2name, key=int)]
    channel = concept_names.index(HARD_CORAL)
    labels = ann["source_label_name"].astype(str).str.lower()
    lookup = build_concept_lookup("coralnet", labels.unique().tolist(), concept_names,
                                  _REPO_ROOT / "configs/class_to_concepts.csv")
    value = {name: int(lookup.row(name)[channel]) for name in labels.unique()}  # 2 pos, 1 neg, 0 not given
    code = labels.map(value).to_numpy()

    per_image = (pd.DataFrame({"source_id": ann["source_id"].to_numpy(), "image_id": ann["image_id"].to_numpy(),
                               "hard": code == 2, "order_not_given": code == 0})
                 .groupby(["source_id", "image_id"], sort=True)
                 .agg(n_points=("hard", "size"), n_hard=("hard", "sum"), n_order_not_given=("order_not_given", "sum"))
                 .reset_index())
    per_image["image_id"] = per_image["image_id"].astype(np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_image.to_parquet(args.out, index=False)
    print(f"{len(per_image)} images, {per_image.n_points.sum()} points, "
          f"{per_image.source_id.nunique()} sources -> {args.out}")


if __name__ == "__main__":
    main()
