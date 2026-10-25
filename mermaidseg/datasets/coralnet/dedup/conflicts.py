"""Classify label disagreements between CoralNet sources that share images, and pick a source.

Input is the image-pair list from ``hashing --all-sources``. Every shared point (same
image content, same ``(row, col)``) gets its two CoralNet labels compared through
``configs/class_to_concepts.csv`` (``common.compare_labels``): each label is a row of taxonomic ranks (kingdom ... genus) and
attribute flags, each either specified or ``not_given``. A pair of labels is

- ``same_label``: identical CoralNet label id;
- ``equivalent``: different ids but identical concept rows (synonyms, e.g. renamed labels);
- ``a_finer`` / ``b_finer``: no field conflicts and one label specifies a strict superset of the
  other's fields (e.g. "hard coral" vs "acropora"): consistent, one source is more specific;
- ``compatible``: no field conflicts but neither is a superset (different, non-clashing detail);
- ``contradictory``: some field is specified on both sides with different values;
- ``unmapped``: at least one label is missing from the concept CSV.

Per source pair it then recommends which source's annotations to keep for the shared images:

1. ``review`` if contradictions exceed ``--max-contradictory`` of the shared points;
2. otherwise the finer source, if it wins at least ``--finer-majority`` of the a/b-finer points;
3. otherwise (labels equivalent or no clear winner) the newer source (higher CoralNet source id,
   since ids are sequential), which is most likely the curated re-upload.

Pairwise picks need not be transitive inside a cluster of overlapping sources, so sources are
also ranked per cluster (connected component of the overlap graph): +1 for every pair a source
is kept in, -1 for every pair it loses (``review`` / ``either`` / ``merge`` count 0), ties broken by
the newer source. ``dedup`` keeps each duplicated image's copy from its best-ranked source.

Outputs (``--outputs``): ``annotation_conflicts.csv`` (one row per source pair),
``annotation_conflict_labels.csv`` (which label pairs disagree) and ``annotation_source_ranking.csv``.

Usage:
    python -m mermaidseg.datasets.coralnet.dedup.conflicts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mermaidseg.datasets.coralnet.dedup.common import (
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONCEPTS_CSV,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUTS,
    DEFAULT_SPLIT_CONFIG,
    add_split_labels,
    compare_labels,
    coralnet_splits,
    load_concepts,
    load_id2name,
    specificity,
)

CATEGORIES = ("same_label", "equivalent", "a_finer", "b_finer", "compatible", "contradictory", "unmapped")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS,
                   help="Where the hashing outputs are and these outputs go.")
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--concepts-csv", type=Path, default=DEFAULT_CONCEPTS_CSV)
    p.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    p.add_argument("--max-contradictory", type=float, default=0.05)
    p.add_argument("--finer-majority", type=float, default=0.8)
    args = p.parse_args()

    root = Path(args.cache_root) / DEFAULT_BUCKET
    id2name = load_id2name(root)
    concepts = load_concepts(args.concepts_csv)

    pairs = pd.read_csv(args.outputs / "source_overlap_pairs_exact_all.csv", dtype={"image_a": str, "image_b": str})
    pairs["pair_id"] = np.arange(len(pairs))
    involved = pd.concat([
        pairs[["source_a", "image_a"]].set_axis(["source_id", "image_id"], axis=1),
        pairs[["source_b", "image_b"]].set_axis(["source_id", "image_id"], axis=1),
    ]).drop_duplicates()
    ann = pq.read_table(root / DEFAULT_MANIFEST, columns=["source_id", "image_id", "row", "col", "coralnet_id"],
                        filters=[("source_id", "in", sorted(involved["source_id"].unique().tolist()))]).to_pandas()
    ann["image_id"] = ann["image_id"].astype(str)
    ann = ann.merge(involved, on=["source_id", "image_id"]).drop_duplicates(["source_id", "image_id", "row", "col"])
    print(f"{len(pairs):,} image pairs, {len(ann):,} annotations on involved images")

    pts = (
        pairs[["pair_id", "source_a", "image_a", "source_b", "image_b"]]
        .merge(ann.rename(columns={"source_id": "source_a", "image_id": "image_a", "coralnet_id": "label_a"}),
               on=["source_a", "image_a"])
        .merge(ann.rename(columns={"source_id": "source_b", "image_id": "image_b", "coralnet_id": "label_b"}),
               on=["source_b", "image_b", "row", "col"])
    )
    print(f"{len(pts):,} shared points")

    # Classify each distinct (label_a, label_b) combination once.
    feat = {lid: (concepts.loc[n] if n in concepts.index else None) for lid, n in id2name.items()}
    combos = pts[["label_a", "label_b"]].drop_duplicates()
    combos["category"] = [
        "same_label" if a == b else compare_labels(feat.get(a), feat.get(b))
        for a, b in zip(combos["label_a"], combos["label_b"], strict=True)
    ]
    pts = pts.merge(combos, on=["label_a", "label_b"])

    combo_counts = (
        pts[pts["category"] != "same_label"]
        .groupby(["source_a", "source_b", "label_a", "label_b", "category"]).size().rename("n_points").reset_index()
    )
    combo_counts["name_a"] = combo_counts["label_a"].map(id2name)
    combo_counts["name_b"] = combo_counts["label_b"].map(id2name)
    combo_counts = combo_counts.sort_values(["source_a", "source_b", "n_points"], ascending=[True, True, False])
    combo_counts.to_csv(args.outputs / "annotation_conflict_labels.csv", index=False)

    counts = pts.groupby(["source_a", "source_b", "category"]).size().unstack(fill_value=0)  # noqa: PD010
    counts = counts.reindex(columns=list(CATEGORIES), fill_value=0)
    n = counts.sum(axis=1)
    out = counts.add_prefix("n_").reset_index()
    out.insert(2, "n_shared_points", n.to_numpy())
    for c in CATEGORIES:
        out[f"pct_{c}"] = 100 * out[f"n_{c}"] / out["n_shared_points"]

    # Source descriptors: label vocabulary and how specific its labels are.
    src_ann = pq.read_table(root / DEFAULT_MANIFEST, columns=["source_id", "image_id", "coralnet_id"],
                            filters=[("source_id", "in", sorted(set(out["source_a"]) | set(out["source_b"])))]).to_pandas()
    spec_of = {lid: specificity(f) for lid, f in feat.items()}
    src = src_ann.groupby("source_id").agg(
        n_labels=("coralnet_id", "nunique"),
        ann_per_image=("coralnet_id", "size"),
        n_img=("image_id", "nunique"),
        median_image_id=("image_id", lambda s: float(np.median(pd.to_numeric(s, errors="coerce")))),
    )
    src["ann_per_image"] = src["ann_per_image"] / src["n_img"]
    src["mean_tax_ranks"] = src_ann.assign(s=src_ann["coralnet_id"].map(spec_of)).groupby("source_id")["s"].mean()
    for side in ("a", "b"):
        for c in ("n_labels", "ann_per_image", "mean_tax_ranks"):
            out[f"{c}_{side}"] = out[f"source_{side}"].map(src[c])

    def recommend(r: pd.Series) -> tuple[str, str]:
        if r["n_shared_points"] == 0:
            return "merge", "no shared point coordinates; annotations are complementary"
        if r["n_same_label"] == r["n_shared_points"]:
            return "either", "identical labels"
        if r["n_contradictory"] > args.max_contradictory * r["n_shared_points"]:
            return "review", f"{r['pct_contradictory']:.0f}% of shared points contradict"
        finer = r["n_a_finer"] + r["n_b_finer"]
        if finer and r["n_a_finer"] >= args.finer_majority * finer:
            return str(int(r["source_a"])), f"finer on {100 * r['n_a_finer'] / finer:.0f}% of consistent disagreements"
        if finer and r["n_b_finer"] >= args.finer_majority * finer:
            return str(int(r["source_b"])), f"finer on {100 * r['n_b_finer'] / finer:.0f}% of consistent disagreements"
        newer = max(r["source_a"], r["source_b"])
        why = "labels equivalent" if not finer else "no clear finer source"
        return str(int(newer)), f"{why}; newer source"

    rec = out.apply(recommend, axis=1, result_type="expand")
    out["keep"], out["reason"] = rec[0], rec[1]
    out["newer_source"] = out[["source_a", "source_b"]].max(axis=1)

    overlap = pd.read_csv(args.outputs / "source_overlap_exact_all.csv")
    out = overlap[["source_a", "source_b", "split_a", "split_b", "cross_split", "n_images_a", "n_images_b",
                   "n_overlap_images", "pct_image_overlap"]].merge(out, on=["source_a", "source_b"], how="left")
    count_cols = ["n_shared_points", *[f"n_{c}" for c in CATEGORIES]]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    no_pts = out["n_shared_points"] == 0
    out.loc[no_pts, "keep"] = "merge"
    out.loc[no_pts, "reason"] = "no shared point coordinates; annotations are complementary"
    out = add_split_labels(out, coralnet_splits(args.split_config))
    out = out.sort_values("n_shared_points", ascending=False)
    out.to_csv(args.outputs / "annotation_conflicts.csv", index=False)

    # Per-cluster source ranking from the pairwise picks.
    graph = nx.Graph()
    graph.add_edges_from(out[["source_a", "source_b"]].to_numpy())
    cluster_of = {s: i for i, comp in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True))
                  for s in comp}
    score = dict.fromkeys(graph.nodes, 0)
    n_review = dict.fromkeys(graph.nodes, 0)
    for a, b, keep in zip(out["source_a"], out["source_b"], out["keep"], strict=True):
        if keep == str(a):
            score[a] += 1
            score[b] -= 1
        elif keep == str(b):
            score[b] += 1
            score[a] -= 1
        elif keep == "review":
            n_review[a] += 1
            n_review[b] += 1
    splits = coralnet_splits(args.split_config)
    split_of = {s: k for k, v in splits.items() for s in v}
    ranking = pd.DataFrame({"source_id": list(graph.nodes)})
    ranking["cluster"] = ranking["source_id"].map(cluster_of)
    ranking["split"] = ranking["source_id"].map(split_of).fillna("unused")
    ranking["score"] = ranking["source_id"].map(score)
    ranking["n_review_pairs"] = ranking["source_id"].map(n_review)
    for c in ("n_img", "n_labels", "ann_per_image", "mean_tax_ranks"):
        ranking[c] = ranking["source_id"].map(src[c])
    ranking = ranking.sort_values(["cluster", "score", "source_id"], ascending=[True, False, False])
    ranking["rank"] = ranking.groupby("cluster").cumcount() + 1
    ranking.to_csv(args.outputs / "annotation_source_ranking.csv", index=False)

    dis = out[out["n_same_label"] < out["n_shared_points"]]
    print(f"\n{len(out)} overlapping pairs, {len(dis)} with any label disagreement")
    tot = dis[[f"n_{c}" for c in CATEGORIES]].sum()
    print("Shared points in disagreeing pairs by category:")
    print((100 * tot / tot.sum()).round(1).to_string())
    print("\nRecommendation (pairs with disagreements):")
    print(dis["reason"].str.split(";").str[0].str.replace(r"\d+%", "N%", regex=True).value_counts().to_string())
    print(f"\nWrote {args.outputs}/annotation_conflicts.csv and annotation_conflict_labels.csv")


if __name__ == "__main__":
    main()
