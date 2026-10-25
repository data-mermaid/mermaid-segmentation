"""Split the deduplicated CoralNet dataset into train/val/test by source group.

Inputs: the sources of ``--split-config`` (the universe), the deduplication outputs of ``dedup``
(blacklist + merged annotations, in ``--config-dir``), the content hashes of ``hashing --all-sources``
and the source metadata of ``mermaidseg.datasets.coralnet.scraper.source_metadata``.

Sources in ``--exclude-sources`` (low quality) are left out entirely.

1. Counts. Per source, images and points per training class on the deduplicated data
   (blacklisted copies removed, merged annotations applied). Labels are mapped the way training
   maps them: ``configs/coralnet_to_mermaid_mapping_temporary.json``, then rolled up through the
   MERMAID benthic hierarchy into the ``class_subset`` of ``--training-config``
   (``label_roll_up``). Labels that end up outside the subset are not counted.
2. Groups. Sources that belong together are put on the same side: connected components of
   - sources that share an image (the copies are deduplicated, but the sources stay together), and
   - sources of one project: at least one admin in common and either IDs at most ``--max-id-gap``
     apart (in the sorted list of universe sources) or locations at most ``--max-km`` apart.
3. Eligibility. A group may go to val/test only if none of its sources shares an image with
   another source and it has at most ``--max-group-frac`` of an eval split's target size.
   Everything else is train.
4. Assignment. Simulated annealing over the eligible groups minimises, for val and test,
   ``w_size * (size error)^2 + mean_c (share error of class c)^2``, where the size error is the
   relative deviation from ``--val-frac`` / ``--test-frac`` of all images and the share error of a
   class is the relative deviation of the split's share of that class's points from the same
   fraction. Classes with fewer than ``--min-class-points`` points are not in that term.
   A coverage term adds ``w_coverage * (1 - n / m)^2`` for every class with ``n < m`` points in
   val or test, where ``m = min(--min-eval-points, eligible points of the class / 4)``, so every
   class that the eligible groups can provide shows up in both eval splits.

Outputs: ``--out-config`` (``--split-config`` with the CoralNet block replaced: new whitelists plus
``image_blacklist`` / ``annotation_overrides``), ``source_splits.csv`` in ``--config-dir`` and
``class_distribution.csv`` / ``summary.json`` in ``--out-dir``.

Usage:
    python -m mermaidseg.datasets.coralnet.dedup.splits
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from mermaidseg.dataset_reconciliation.concepts import initialize_benthic_hierarchy
from mermaidseg.dataset_reconciliation.registry import roll_up_label
from mermaidseg.datasets.coralnet.dedup.common import (
    CONFIG_DIR,
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_EXCLUDE_SOURCES,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUTS,
    DEFAULT_SPLIT_CONFIG,
    REPO_ROOT,
    coralnet_splits,
    load_id2name,
    read_source_list,
    repo_relative,
)

SPLITS = ("train", "val", "test")


def dedup_annotations(root: Path, sources: list[int], config_dir: Path) -> pd.DataFrame:
    """Manifest points of ``sources`` after the blacklist and the annotation overrides.

    Mirrors ``CoralNetDataset(image_blacklist=..., annotation_overrides=...)``.
    """
    ann = pq.read_table(root / DEFAULT_MANIFEST, columns=["source_id", "image_id", "row", "col", "coralnet_id"],
                        filters=[("source_id", "in", sources)]).to_pandas()
    ann["image_id"] = ann["image_id"].astype(str)
    blacklist = set(pd.read_parquet(config_dir / "image_blacklist.parquet")["image_id"].astype(str))
    overrides = pd.read_parquet(config_dir / "annotation_overrides.parquet")[list(ann.columns)]
    overrides = overrides[overrides["source_id"].isin(sources)].astype(ann.dtypes.to_dict())
    ann = ann[~ann["image_id"].isin(blacklist | set(overrides["image_id"]))]
    return pd.concat([ann, overrides], ignore_index=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def project_edges(meta: pd.DataFrame, sources: list[int], max_id_gap: int, max_km: float) -> list[tuple]:
    """Pairs of sources sharing an admin that are close in ID or in location."""
    rank = {s: i for i, s in enumerate(sources)}
    meta = meta[meta["source_id"].isin(rank) & meta["admins"].notna()].set_index("source_id")
    by_admin = meta["admins"].str.split(";").explode()
    edges = {}
    for _, ids in by_admin.groupby(by_admin):
        ids = sorted(set(ids.index))
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if (a, b) in edges:
                    continue
                near_id = abs(rank[a] - rank[b]) <= max_id_gap
                near_loc = haversine_km(meta.loc[a, "latitude"], meta.loc[a, "longitude"],
                                        meta.loc[b, "latitude"], meta.loc[b, "longitude"]) <= max_km  # NaN -> False
                if near_id or near_loc:
                    edges[a, b] = "id" if near_id else "location"
    return [(a, b, why) for (a, b), why in edges.items()]


def anneal(X, n, P, eligible, targets, fracs, w_size, cls_mask, min_pts, w_cov, iters, seed):
    """Assign eligible groups to 0 (train), 1 (val), 2 (test); returns (assignment, cost)."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(eligible)
    a = np.zeros(len(n), dtype=int)
    p0 = sum(targets) / n[idx].sum()
    a[idx] = np.where(rng.random(len(idx)) < p0, rng.integers(1, 3, len(idx)), 0)
    S = {s: X[a == s].sum(0) for s in (1, 2)}
    N = {s: n[a == s].sum() for s in (1, 2)}
    Pm = np.where(P > 0, P, 1)
    has_min = min_pts > 0
    m = min_pts[has_min]

    def split_cost(s, Ss, Ns):
        size = ((Ns - targets[s - 1]) / targets[s - 1]) ** 2
        share = ((Ss / Pm - fracs[s - 1]) / fracs[s - 1]) ** 2
        coverage = (np.clip(1 - Ss[has_min] / m, 0, None) ** 2).sum()
        return w_size * size + share[cls_mask].mean() + w_cov * coverage

    cost = {s: split_cost(s, S[s], N[s]) for s in (1, 2)}
    t0, t1 = 1.0, 1e-5
    for it in range(iters):
        temp = t0 * (t1 / t0) ** (it / iters)
        # Half the moves send one group to another split; the other half swap two groups in
        # different splits, which lets a large group trade places without breaking the sizes.
        g = idx[rng.integers(len(idx))]
        if rng.random() < 0.5:
            moves = [(g, (a[g] + rng.integers(1, 3)) % 3)]
        else:
            g2 = idx[rng.integers(len(idx))]
            if a[g2] == a[g]:
                continue
            moves = [(g, a[g2]), (g2, a[g])]
        dS = {s: np.zeros_like(P) for s in (1, 2)}
        dN = {1: 0.0, 2: 0.0}
        for grp, new in moves:
            for s, sign in ((a[grp], -1), (new, 1)):
                if s:
                    dS[s] += sign * X[grp]
                    dN[s] += sign * n[grp]
        delta, upd = 0.0, {}
        for s in (1, 2):
            if dN[s] or dS[s].any():
                Ss, Ns = S[s] + dS[s], N[s] + dN[s]
                c = split_cost(s, Ss, Ns)
                upd[s] = (Ss, Ns, c)
                delta += c - cost[s]
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            for grp, new in moves:
                a[grp] = new
            for s, (Ss, Ns, c) in upd.items():
                S[s], N[s], cost[s] = Ss, Ns, c
    return a, cost[1] + cost[2]


def replace_coralnet_block(config_text: str, splits: dict[str, list[int]], extra: dict[str, str]) -> str:
    """Swap the ``coralnet:`` block of a data config for one with the given whitelists."""
    lines = config_text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if re.match(r"^  coralnet:\s*$", ln))
    end = next(i for i in range(start + 1, len(lines)) if re.match(r"^  \S", lines[i]))
    block = ["  coralnet:\n"]
    for split, ids in splits.items():
        block.append(f"    {split}:\n")
        body = ", ".join(str(s) for s in ids)
        wrapped, line = [], ""
        for tok in body.split(" "):
            if len(line) + len(tok) + 1 > 76 and line:
                wrapped.append(line.rstrip())
                line = ""
            line += tok + " "
        wrapped.append(line.rstrip())
        block.append("      whitelist_sources: [" + ("\n" + " " * 26).join(wrapped) + "]\n")
        for k, v in extra.items():
            block.append(f"      {k}: {v}\n")
    return "".join(lines[:start] + block + lines[end:])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    p.add_argument("--exclude-sources", type=Path, default=DEFAULT_EXCLUDE_SOURCES,
                   help="Text file of source ids to leave out entirely (one per line).")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS, help="Where the hashing outputs are.")
    p.add_argument("--config-dir", type=Path, default=CONFIG_DIR,
                   help="Where the dedup blacklist / overrides are and source_splits.csv goes.")
    p.add_argument("--metadata", type=Path, default=DEFAULT_OUTPUTS / "coralnet_source_metadata.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS / "splits")
    p.add_argument("--out-config", type=Path, default=REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--mapping", type=Path, default=REPO_ROOT / "configs/coralnet_to_mermaid_mapping_temporary.json")
    p.add_argument("--training-config", type=Path, default=REPO_ROOT / "configs/training_config_cbm.yaml",
                   help="Its training.class_subset is the class space (with label roll-up).")
    p.add_argument("--hierarchy", type=Path, default=CONFIG_DIR / "mermaid_benthic_hierarchy.json",
                   help="MERMAID benthic hierarchy snapshot (name -> parent); fetched from the API if missing.")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--max-group-frac", type=float, default=0.25,
                   help="Largest group allowed in val/test, as a fraction of that split's target size.")
    p.add_argument("--max-id-gap", type=int, default=3)
    p.add_argument("--max-km", type=float, default=50.0)
    p.add_argument("--min-class-points", type=int, default=500)
    p.add_argument("--min-eval-points", type=int, default=100)
    p.add_argument("--w-coverage", type=float, default=1.0)
    p.add_argument("--w-size", type=float, default=10.0)
    p.add_argument("--iters", type=int, default=200_000)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.cache_root) / DEFAULT_BUCKET
    excluded = read_source_list(args.exclude_sources)
    sources = sorted(set().union(*coralnet_splits(args.split_config).values()) - excluded)

    # 1. Per-source images and mapped-class point counts on the deduplicated data.
    ann = dedup_annotations(root, sources, args.config_dir)
    id2name = load_id2name(root)
    mapping = {k.lower(): v.lower() for k, v in json.loads(args.mapping.read_text()).items() if v is not None}
    if not args.hierarchy.exists():
        args.hierarchy.write_text(json.dumps(initialize_benthic_hierarchy(), indent=0))
    hierarchy = {k.lower(): (v.lower() if v else None) for k, v in json.loads(args.hierarchy.read_text()).items()}
    subset = {c.lower() for c in yaml.safe_load(args.training_config.read_text())["training"]["class_subset"]}
    label_to_cls = {lid: roll_up_label(mapping[name], hierarchy, subset) if name in mapping else None
                    for lid, name in id2name.items()}
    ann["cls"] = ann["coralnet_id"].map(label_to_cls)
    n_img = ann.groupby("source_id")["image_id"].nunique().reindex(sources, fill_value=0)
    counts = ann.dropna(subset=["cls"]).groupby(["source_id", "cls"]).size().unstack(fill_value=0)  # noqa: PD010
    counts = counts.reindex(sources, fill_value=0)
    print(f"{len(sources)} sources ({len(excluded)} excluded), {n_img.sum():,} images, {len(ann):,} points "
          f"({ann['cls'].notna().mean():.1%} mapped to {counts.shape[1]} of {len(subset)} training classes)")

    # 2. Groups: shared images + same project.
    hashes = pd.read_parquet(args.outputs / "image_hashes_exact_all.parquet", columns=["source_id", "hash"])
    hashes = hashes[hashes["source_id"].isin(sources)].drop_duplicates()
    shared = hashes[hashes.duplicated("hash", keep=False)]
    cross_sources = set(shared.loc[shared.groupby("hash")["source_id"].transform("nunique") > 1, "source_id"])
    overlap_edges = {tuple(sorted(e)) for _, g in shared.groupby("hash") for e in zip(g["source_id"].iloc[:-1],
                                                                                     g["source_id"].iloc[1:], strict=True)}
    meta = pd.read_csv(args.metadata)
    proj = project_edges(meta, sources, args.max_id_gap, args.max_km)
    graph = nx.Graph()
    graph.add_nodes_from(sources)
    graph.add_edges_from((a, b) for a, b in overlap_edges if a != b)
    graph.add_edges_from((a, b) for a, b, _ in proj)
    comps = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))
    group_of = {s: i for i, c in enumerate(comps) for s in c}
    print(f"{len(proj)} project links ({sum(e[2] == 'id' for e in proj)} by id, "
          f"{sum(e[2] == 'location' for e in proj)} by location), {len(overlap_edges)} overlap links "
          f"-> {len(comps)} groups")

    # 3. Group features and eligibility.
    src = pd.DataFrame({"source_id": sources, "group": [group_of[s] for s in sources], "n_images": n_img.to_numpy(),
                        "shares_images": [s in cross_sources for s in sources]})
    grp = src.groupby("group").agg(n_images=("n_images", "sum"), shares_images=("shares_images", "any"),
                                   n_sources=("source_id", "size"))
    X = counts.groupby(src.set_index("source_id")["group"]).sum().reindex(grp.index).to_numpy(float)
    n = grp["n_images"].to_numpy(float)
    N = n.sum()
    fracs = (args.val_frac, args.test_frac)
    targets = tuple(f * N for f in fracs)
    cap = args.max_group_frac * min(targets)
    grp["eligible"] = ~grp["shares_images"] & (grp["n_images"] <= cap) & (grp["n_images"] > 0)
    print(f"eligible groups: {int(grp['eligible'].sum())} with {int(grp.loc[grp['eligible'], 'n_images'].sum()):,} "
          f"images (cap {cap:,.0f} images per group); targets {targets[0]:,.0f} / {targets[1]:,.0f}")

    # 4. Assignment.
    P = X.sum(0)
    cls_mask = args.min_class_points <= P
    pool = X[grp["eligible"].to_numpy()].sum(0)
    min_pts = np.minimum(args.min_eval_points, pool / 4)
    best = None
    for r in range(args.restarts):
        a, c = anneal(X, n, P, grp["eligible"].to_numpy(), targets, fracs, args.w_size, cls_mask,
                      min_pts, args.w_coverage, args.iters, args.seed + r)
        print(f"restart {r}: cost {c:.4f}")
        if best is None or c < best[1]:
            best = (a, c)
    grp["split"] = np.array(SPLITS)[best[0]]
    src["split"] = src["group"].map(grp["split"])

    # Outputs.
    meta_cols = [c for c in ("name", "admins", "affiliation", "latitude", "longitude", "created", "public")
                 if c in meta.columns]
    src = src.merge(meta[["source_id", *meta_cols]], on="source_id", how="left")
    src["eligible_group"] = src["group"].map(grp["eligible"])
    src.sort_values(["split", "group", "source_id"]).to_csv(args.config_dir / "source_splits.csv", index=False)

    split_counts = pd.DataFrame({s: X[grp["split"].to_numpy() == s].sum(0) for s in SPLITS}, index=counts.columns)
    dist = split_counts.copy()
    dist["total"] = P
    for s in SPLITS:
        dist[f"share_{s}"] = split_counts[s] / P
        dist[f"prop_{s}"] = split_counts[s] / split_counts[s].sum()
    dist["in_objective"] = cls_mask
    dist["eligible_points"] = pool
    dist["min_eval_points"] = min_pts
    dist.sort_values("total", ascending=False).to_csv(args.out_dir / "class_distribution.csv")

    def js(p, q):
        p, q = p / p.sum(), q / q.sum()
        m = (p + q) / 2
        def kl(x, y):
            return np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / np.where(y > 0, y, 1)), 0))

        return float((kl(p, m) + kl(q, m)) / 2)

    summary = {
        "split_config": str(args.split_config),
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_groups": len(grp), "n_eligible_groups": int(grp["eligible"].sum()),
        "cost": float(best[1]),
        "splits": {
            s: {
                "n_sources": int((src["split"] == s).sum()),
                "n_groups": int((grp["split"] == s).sum()),
                "n_images": int(src.loc[src["split"] == s, "n_images"].sum()),
                "frac_images": float(src.loc[src["split"] == s, "n_images"].sum() / N),
                "n_mapped_points": int(split_counts[s].sum()),
                "n_classes_present": int((split_counts[s] > 0).sum()),
                "js_divergence_vs_all": js(split_counts[s].to_numpy(), P),
            }
            for s in SPLITS
        },
        "classes_missing": {s: split_counts.index[split_counts[s] == 0].tolist() for s in ("val", "test")},
        "classes_below_min": {s: split_counts.index[split_counts[s] < min_pts].tolist() for s in ("val", "test")},
        "classes_absent_from_coralnet": sorted(subset - set(counts.columns)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("splits", "classes_missing", "classes_below_min", "classes_absent_from_coralnet")}, indent=2))

    whitelists = {s: sorted(src.loc[src["split"] == s, "source_id"].tolist()) for s in SPLITS}
    # Repo-relative when inside the repo: CoralNetDataset resolves those against the repo root.
    extra = {"image_blacklist": repo_relative(args.config_dir / "image_blacklist.parquet"),
             "annotation_overrides": repo_relative(args.config_dir / "annotation_overrides.parquet")}
    args.out_config.write_text(replace_coralnet_block(args.split_config.read_text(), whitelists, extra))
    print(f"Wrote {args.out_config}")


if __name__ == "__main__":
    main()
