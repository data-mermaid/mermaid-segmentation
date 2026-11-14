#!/usr/bin/env python3
"""Which CoralNet sources of a split can be used for hard coral cover, under a run's concept CSV.

Per source, from the annotations mapped through ``class_to_concepts.csv`` (hard coral = ``order__scleractinia``
positive):

- ``images``, ``points``, ``points_per_image`` (median), ``cover`` (pooled share of hard coral points) and
  ``live_cover`` (pooled share of live hard coral points: scleractinia positive and ``live`` positive);
- ``live_unknown_frac``: share of the hard coral points whose ``live`` is not given (neither live nor dead, so they
  do not count as live hard coral);
- ``dead_hard_frac``: share of the hard coral points whose ``live`` concept is negative (dead coral that the
  scleractinia mapping counts as hard coral);
- ``coral_no_order_frac``: share of all points whose label looks like coral (``coral``, ``scleractinia``, a genus
  in the CSV with order scleractinia; not soft / dead / rubble / "not coral") but whose order is not given, so they
  count as not hard coral;
- ``top_dead`` / ``top_coral_no_order``: the labels behind those two.

Also writes source name, latitude and longitude from the scraped source metadata.

Usage:
    python scripts/coral_cover_eligibility.py --split test \
        --concept-id2name eval_out/scratch100k_hier/cbm_scratch100k_kl_w0.5_taxv2/concept_id2name.json \
        --class-to-concepts nbs/model_checkpoints/cbm_scratch100k_kl_w0.5_taxv2/class_to_concepts.csv \
        --out eval_out/coral_cover/test_sources_cover_eligibility_taxv2.csv
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
LIVE = "live"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-config", type=Path, default=_REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--concept-id2name", type=Path, required=True)
    p.add_argument("--class-to-concepts", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=_REPO_ROOT / "data/coralnet_dedup/coralnet_source_metadata.csv")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = yaml.safe_load(args.data_config.read_text())
    data = data.get("data", data)
    setup_local_cache(data)
    split = (data.get("coralnet") or {}).get(args.split) or {}
    base = CoralNetDataset(whitelist_sources=split.get("whitelist_sources"),
                           image_blacklist=split.get("image_blacklist"),
                           annotation_overrides=split.get("annotation_overrides"),
                           transform=None, split=args.split)
    ann = base.df_annotations

    id2name = json.loads(args.concept_id2name.read_text())
    concept_names = [id2name[k] for k in sorted(id2name, key=int)]
    hard_ch = concept_names.index(HARD_CORAL)
    live_ch = concept_names.index(LIVE) if LIVE in concept_names else None
    labels = ann["source_label_name"].astype(str).str.lower()
    uniq = labels.unique().tolist()
    lookup = build_concept_lookup("coralnet", uniq, concept_names, args.class_to_concepts)
    hard = {n: int(lookup.row(n)[hard_ch]) for n in uniq}  # 2 pos, 1 neg, 0 not given
    live = {n: int(lookup.row(n)[live_ch]) if live_ch is not None else 0 for n in uniq}

    csv = pd.read_csv(args.class_to_concepts, dtype=str)
    genera = set(csv.loc[csv["order"].str.lower() == "scleractinia", "genus"].dropna().str.lower()) - {"not_given"}
    coralish = {n: ("coral" in n or "scleractin" in n or any(w in genera for w in n.replace("_", " ").split()))
                and not any(w in n for w in ("soft", "dead", "not coral", "rubble")) for n in uniq}

    df = pd.DataFrame({"source_id": ann["source_id"].to_numpy(), "image_id": ann["image_id"].to_numpy(),
                       "label": labels.to_numpy()})
    df["hard"] = df.label.map(hard) == 2
    df["dead_hard"] = df.hard & (df.label.map(live) == 1)
    df["live_hard"] = df.hard & (df.label.map(live) == 2)
    df["live_unknown"] = df.hard & (df.label.map(live) == 0)
    df["coral_no_order"] = (df.label.map(hard) == 0) & df.label.map(coralish)

    rows = []
    for s, d in df.groupby("source_id"):
        per_img = d.groupby("image_id").size()
        rows.append({
            "source_id": int(s), "images": len(per_img), "points": len(d),
            "points_per_image": int(np.median(per_img)), "cover": d.hard.mean(),
            "live_cover": d.live_hard.mean(),
            "dead_hard_frac": d.dead_hard.sum() / max(d.hard.sum(), 1),
            "live_unknown_frac": d.live_unknown.sum() / max(d.hard.sum(), 1),
            "coral_no_order_frac": d.coral_no_order.mean(),
            "top_dead": ", ".join(f"{k} ({v})" for k, v in d[d.dead_hard].label.value_counts().head(3).items()),
            "top_live_unknown": ", ".join(f"{k} ({v})" for k, v in
                                          d[d.live_unknown].label.value_counts().head(3).items()),
            "top_coral_no_order": ", ".join(f"{k} ({v})" for k, v in
                                            d[d.coral_no_order].label.value_counts().head(3).items()),
        })
    out = pd.DataFrame(rows)
    meta = pd.read_csv(args.metadata)[["source_id", "name", "latitude", "longitude"]]
    out = out.merge(meta, on="source_id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} sources -> {args.out}")


if __name__ == "__main__":
    main()
