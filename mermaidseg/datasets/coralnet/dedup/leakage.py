"""Check a data config's CoralNet train / val / test splits for duplicated images.

Reuses the per-image content hashes written by ``hashing --all-sources`` (no
re-hashing), so this runs in seconds. For every (eval split, other split) combination it reports
how many eval images also appear in the other split, per eval source, and lists the source pairs
responsible together with their annotation overlap.

Usage:
    python -m mermaidseg.datasets.coralnet.dedup.leakage --config configs/data_config_dedup.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mermaidseg.datasets.coralnet.dedup.common import (
    DEFAULT_OUTPUTS,
    DEFAULT_SPLIT_CONFIG,
    add_split_labels,
    coralnet_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--hash-mode", default="exact")
    args = parser.parse_args()

    splits = coralnet_splits(args.config)
    hashes = pd.read_parquet(args.outputs / f"image_hashes_{args.hash_mode}_all.parquet")
    pairs = pd.read_csv(args.outputs / f"source_overlap_{args.hash_mode}_all.csv")

    for k, v in splits.items():
        uncached = sorted(v - set(hashes["source_id"]))
        print(f"{k}: {len(v)} sources, {hashes['source_id'].isin(v).sum():,} hashed images; uncached: {uncached}")

    per_source = []
    for e, o in [("val", "train"), ("test", "train"), ("test", "val")]:
        if e not in splits or o not in splits:
            continue
        other = set(hashes.loc[hashes["source_id"].isin(splits[o]), "hash"])
        x = hashes[hashes["source_id"].isin(splits[e])]
        dup = x["hash"].isin(other)
        print(f"\n{e} images also in {o}: {dup.sum():,}/{len(x):,} ({100 * dup.mean():.1f}%)")
        t = x.assign(dup=dup).groupby("source_id")["dup"].agg(n_images="size", n_dup="sum").reset_index()
        t["pct_dup"] = 100 * t["n_dup"] / t["n_images"]
        t.insert(0, "eval_split", e)
        t.insert(1, "other_split", o)
        per_source.append(t)
        print(t[t["n_dup"] > 0].drop(columns=["eval_split", "other_split"]).round(1).to_string(index=False))

    cross = add_split_labels(pairs, splits)
    cross = cross[cross["cross_split"]]

    tag = args.config.stem
    pd.concat(per_source).to_csv(args.outputs / f"split_leakage_{tag}.csv", index=False)
    cross.to_csv(args.outputs / f"split_leakage_pairs_{tag}.csv", index=False)
    print(f"\nWrote {args.outputs}/split_leakage_{tag}.csv and split_leakage_pairs_{tag}.csv")


if __name__ == "__main__":
    main()
