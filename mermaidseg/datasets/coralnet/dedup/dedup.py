"""Deduplicate CoralNet images and merge the annotations of the duplicate copies.

Uses the content hashes from ``hashing --all-sources`` and the per-cluster source ranking
from ``conflicts``. Only sources listed in a split of ``--split-config`` are used.

1. Images. Every image is either unique (its hash occurs once) or one copy of a duplicated image.
   For each duplicated image, one copy is kept: the copy from the best-ranked source of its
   cluster, ties broken by the newer source, then by more points, then by the lower image id (the
   same order as ``canonical_images.parquet``, minus the split rules). All other copies go to the
   image blacklist.

2. Annotations. All points of all copies of a duplicated image are pooled on the kept copy
   (byte-identical images share one coordinate space). Points at a location only one copy
   annotated are added as is. Where several labels meet at one ``(row, col)``, they are compared
   through ``configs/class_to_concepts.csv`` (see ``common.compare_labels``):

   - ``same_label`` / ``equivalent`` (synonyms): keep the best-ranked copy's label;
   - ``finer``: labels are consistent and one covers all the others: keep that, the finest label;
   - ``compatible``: no clash, but no label covers the others (e.g. taxon vs condition): keep the
     label with the most taxonomic ranks specified (then most fields, then best-ranked copy);
   - ``contradictory``: keep the best-ranked copy's label;
   - labels missing from the concept CSV are ignored when a mapped label is present (``unmapped``
     when a single mapped label is left); if only unmapped labels meet (``unknown``), keep the
     best-ranked copy's label.

   ``compatible``, ``contradictory``, ``unknown`` and any location with an unmapped label are
   written to the manual review files with the label that was picked.

3. Near duplicates. The pairs labelled ``dup`` in ``--near-dup-labels`` (``Near_Duplicate_Review.ipynb``)
   are merged the same way, after mapping each copy's points onto the kept copy (see
   ``near_dup_merge``). Skipped if the labels file does not exist.

Outputs read by training, in ``--config-dir`` (``configs/coralnet/``, versioned):

- ``image_blacklist.parquet``: the copies to remove (``image_blacklist`` of ``CoralNetDataset``), with
  the copy kept in their place and the ``reason`` (``exact`` / ``near_dup``; an exact copy of an
  image removed as a near duplicate points to that image, not to the final kept copy);
- ``annotation_overrides.parquet``: the merged points of every kept copy whose annotations change
  (``annotation_overrides`` of ``CoralNetDataset``; replaces all of that image's points).

Other outputs, in ``--out-dir``:

- ``images_unique.parquet``: images whose hash occurs once;
- ``images_duplicated.parquet``: every copy of a duplicated image, with ``keep`` and the kept copy;
- ``point_provenance.parquet``: every merged location of every duplicated image, with its
  ``change`` (unchanged / replaced / added), the copy the final label came from
  (``origin_copy_rank``, see ``images_duplicated.parquet``) and every copy's label there;
- ``review_points.csv``: one row per flagged location, with every copy's label;
- ``review_images.csv``: one row per kept image with flagged locations;
- ``near_dup_copies.parquet``: one row per near-duplicate copy removed, with its kept copy, the
  transform and how many of its points were transferred;
- ``near_dup_point_provenance.parquet``: every location of every near-duplicate kept copy (as
  ``point_provenance.parquet``; ``copy_ranks`` refer to ``near_dup_copies.parquet``, 0 = kept copy);
- ``near_dup_review_points.csv``: its locations with compatible / contradictory / unmapped labels;
- ``summary.json``.

Usage:
    python -m mermaidseg.datasets.coralnet.dedup.dedup
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mermaidseg.datasets.coralnet.dedup.common import (
    CONFIG_DIR,
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONCEPTS_CSV,
    DEFAULT_EXCLUDE_SOURCES,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUTS,
    DEFAULT_SPLIT_CONFIG,
    TAX_RANKS,
    compare_labels,
    coralnet_splits,
    load_concepts,
    load_id2name,
    read_source_list,
)
from mermaidseg.datasets.coralnet.dedup.near_dup_merge import (
    DEFAULT_LABELS,
    POINT_COLUMNS,
    load_labels,
    merge_near_duplicates,
)

REVIEW_CATEGORIES = ("compatible", "contradictory", "unknown")


def make_resolver(feat: dict[int, pd.Series | None]):
    """Return ``resolve(labels) -> (label, category, has_unmapped)``, cached per label tuple.

    ``labels`` are the distinct labels at one location, ordered by copy rank (best copy first).
    """
    cache: dict[tuple[int, ...], tuple[int, str, bool]] = {}

    def n_specified(lab: int) -> tuple[int, int]:
        f = feat[lab]
        spec = f != "not_given"
        return int(spec[[r for r in TAX_RANKS if r in f.index]].sum()), int(spec.sum())

    def resolve(labels: tuple[int, ...]) -> tuple[int, str, bool]:
        if labels in cache:
            return cache[labels]
        if len(labels) == 1:
            out = (labels[0], "same_label", False)
        else:
            mapped = [lab for lab in labels if feat.get(lab) is not None]
            has_unmapped = len(mapped) < len(labels)
            rel = {}
            for a, b in combinations(mapped, 2):
                r = compare_labels(feat[a], feat[b])
                rel[a, b] = r
                rel[b, a] = {"a_finer": "b_finer", "b_finer": "a_finer"}.get(r, r)
            if not mapped:
                out = (labels[0], "unknown", True)
            elif len(mapped) == 1:
                out = (mapped[0], "unmapped", True)
            elif "contradictory" in rel.values():
                out = (labels[0], "contradictory", has_unmapped)
            else:
                covers = [a for a in mapped if all(rel[a, b] in ("a_finer", "equivalent") for b in mapped if b != a)]
                if covers:
                    cat = "finer" if any(r == "a_finer" for r in rel.values()) else "equivalent"
                    out = (covers[0], cat, has_unmapped)
                else:
                    best = min(range(len(mapped)), key=lambda i: (*(-n for n in n_specified(mapped[i])), i))
                    out = (mapped[best], "compatible", has_unmapped)
        cache[labels] = out
        return out

    return resolve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS,
                   help="Where the hashing and conflicts outputs are.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS / "dedup")
    p.add_argument("--config-dir", type=Path, default=CONFIG_DIR,
                   help="Where the files training reads (blacklist, overrides) go.")
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--concepts-csv", type=Path, default=DEFAULT_CONCEPTS_CSV)
    p.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    p.add_argument("--exclude-sources", type=Path, default=DEFAULT_EXCLUDE_SOURCES,
                   help="Text file of source ids to leave out entirely (one per line).")
    p.add_argument("--near-dup-labels", type=Path, default=DEFAULT_LABELS,
                   help="Pair labels of Near_Duplicate_Review.ipynb; pairs labelled dup are merged.")
    p.add_argument("--workers", type=int, default=32, help="Threads decoding near-duplicate images.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.cache_root) / DEFAULT_BUCKET
    id2name = load_id2name(root)
    concepts = load_concepts(args.concepts_csv)
    feat = {lid: (concepts.loc[n] if n in concepts.index else None) for lid, n in id2name.items()}

    excluded = read_source_list(args.exclude_sources)
    sources = sorted(set().union(*coralnet_splits(args.split_config).values()) - excluded)
    hashes = pd.read_parquet(args.outputs / "image_hashes_exact_all.parquet",
                             columns=["source_id", "image_id", "hash", "image_s3_key"])
    hashes["image_id"] = hashes["image_id"].astype(str)
    missing = sorted(set(sources) - set(hashes["source_id"]))
    if missing:
        raise SystemExit(f"{len(missing)} sources in {args.split_config} were never hashed: {missing[:20]}")
    hashes = hashes[hashes["source_id"].isin(sources)].reset_index(drop=True)

    ann = pq.read_table(root / DEFAULT_MANIFEST, columns=["source_id", "image_id", "row", "col", "coralnet_id"],
                        filters=[("source_id", "in", sources)]).to_pandas()
    ann["image_id"] = ann["image_id"].astype(str)
    n_points = ann.groupby(["source_id", "image_id"]).size().rename("n_points").reset_index()
    hashes = hashes.merge(n_points, on=["source_id", "image_id"], how="left")
    hashes["n_copies"] = hashes.groupby("hash")["hash"].transform("size")
    print(f"{len(sources)} sources, {len(hashes):,} images, {hashes['hash'].nunique():,} unique")

    # 1. Unique vs duplicated images, and the kept copy of each duplicated image.
    unique = hashes[hashes["n_copies"] == 1].drop(columns="n_copies")
    dup = hashes[hashes["n_copies"] > 1].copy()
    ranking = pd.read_csv(args.outputs / "annotation_source_ranking.csv")
    dup["cluster"] = dup["source_id"].map(dict(zip(ranking["source_id"], ranking["cluster"], strict=True)))
    dup["source_rank"] = dup["source_id"].map(dict(zip(ranking["source_id"], ranking["rank"], strict=True)))
    dup = dup.sort_values(["hash", "source_rank", "source_id", "n_points", "image_id"],
                          ascending=[True, True, False, False, True], na_position="last")
    dup["copy_rank"] = dup.groupby("hash").cumcount()
    dup["keep"] = dup["copy_rank"] == 0
    kept = dup.loc[dup["keep"], ["hash", "source_id", "image_id"]].set_axis(
        ["hash", "kept_source_id", "kept_image_id"], axis=1)
    dup = dup.merge(kept, on="hash")
    blacklist = dup.loc[~dup["keep"], ["source_id", "image_id", "hash", "kept_source_id", "kept_image_id"]]
    blacklist = blacklist.assign(reason="exact")

    unique.to_parquet(args.out_dir / "images_unique.parquet", index=False)
    dup.to_parquet(args.out_dir / "images_duplicated.parquet", index=False)
    print(f"unique: {len(unique):,} images; duplicated: {len(dup):,} copies of {len(kept):,} images; "
          f"blacklist: {len(blacklist):,}")

    # 2. Pool every copy's points on the kept copy and resolve each location.
    pts = ann.merge(dup[["source_id", "image_id", "hash", "copy_rank", "kept_source_id", "kept_image_id"]],
                    on=["source_id", "image_id"])
    pts = pts.drop_duplicates(["hash", "copy_rank", "row", "col", "coralnet_id"])
    pts = pts.sort_values(["hash", "row", "col", "copy_rank"]).reset_index(drop=True)
    loc_keys = ["hash", "row", "col"]
    first = ~pts.duplicated(loc_keys)
    loc_id = first.cumsum() - 1
    pts["loc"] = loc_id

    locs = pts.loc[first, ["loc", *loc_keys, "kept_source_id", "kept_image_id"]].set_index("loc")
    locs["n_copies_at_loc"] = pts.groupby("loc")["copy_rank"].nunique()
    locs["in_kept"] = pts.groupby("loc")["copy_rank"].min() == 0
    kept_label = pts[pts["copy_rank"] == 0].drop_duplicates("loc").set_index("loc")["coralnet_id"]
    locs["kept_label"] = kept_label

    n_labels = pts.groupby("loc")["coralnet_id"].nunique()
    locs["coralnet_id"] = pts[first].set_index("loc")["coralnet_id"].astype("int64")  # single-label locations
    locs["category"] = "same_label"
    locs["has_unmapped"] = False
    multi = n_labels.index[n_labels > 1]
    resolve = make_resolver(feat)
    label_tuples = (pts[pts["loc"].isin(multi)].drop_duplicates(["loc", "coralnet_id"])
                    .groupby("loc")["coralnet_id"].agg(tuple))
    resolved = [resolve(t) for t in label_tuples]
    locs.loc[label_tuples.index, "coralnet_id"] = [r[0] for r in resolved]
    locs.loc[label_tuples.index, "category"] = [r[1] for r in resolved]
    locs.loc[label_tuples.index, "has_unmapped"] = [r[2] for r in resolved]
    locs["review"] = locs["category"].isin(REVIEW_CATEGORIES) | locs["has_unmapped"]
    locs["changed"] = ~locs["in_kept"] | (locs["coralnet_id"] != locs["kept_label"])

    # Kept copies whose own (row, col, label) set differs from the merged one get an override.
    own = pts[pts["copy_rank"] == 0].groupby("hash").size()
    n_locs = locs.groupby("hash").size()
    changed_hashes = set(locs.loc[locs["changed"], "hash"]) | set(own.index[own.reindex(n_locs.index) != n_locs])
    overrides = locs[locs["hash"].isin(changed_hashes)].rename(
        columns={"kept_source_id": "source_id", "kept_image_id": "image_id"})
    overrides = overrides[["source_id", "image_id", "row", "col", "coralnet_id", "category", "has_unmapped",
                           "in_kept", "n_copies_at_loc", "hash"]].astype({"coralnet_id": "int32"})

    # Provenance of every merged location of every duplicated image (for nbs/Dedup_Review.ipynb).
    prov = locs.copy()
    prov["change"] = "unchanged"
    prov.loc[prov["in_kept"] & (prov["coralnet_id"] != prov["kept_label"]), "change"] = "replaced"
    prov.loc[~prov["in_kept"], "change"] = "added"
    final = pts["loc"].map(prov["coralnet_id"])
    prov["origin_copy_rank"] = pts[pts["coralnet_id"] == final].groupby("loc")["copy_rank"].min()
    by_loc = pts.groupby("loc")
    prov["copy_ranks"] = by_loc["copy_rank"].agg(list)
    prov["copy_labels"] = by_loc["coralnet_id"].agg(list)
    prov["overridden"] = prov["hash"].isin(changed_hashes)
    prov = prov[["hash", "kept_source_id", "kept_image_id", "row", "col", "coralnet_id", "kept_label", "change",
                 "category", "has_unmapped", "review", "in_kept", "n_copies_at_loc", "origin_copy_rank",
                 "copy_ranks", "copy_labels", "overridden"]].astype(
        {"coralnet_id": "int32", "kept_label": "Int32", "origin_copy_rank": "int16"})
    prov.to_parquet(args.out_dir / "point_provenance.parquet", index=False)

    # Manual review files.
    flagged = locs[locs["review"]]
    detail = pts[pts["loc"].isin(flagged.index)].copy()
    detail["entry"] = (detail["source_id"].astype(str) + "/" + detail["image_id"] + ": "
                       + detail["coralnet_id"].map(id2name).fillna("?"))
    review = flagged.join(detail.groupby("loc")["entry"].agg(" | ".join).rename("labels_by_copy"))
    review["chosen_label"] = review["coralnet_id"].map(id2name)
    review["kept_copy_label"] = review["kept_label"].map(id2name)
    review = review[["kept_source_id", "kept_image_id", "row", "col", "category", "has_unmapped", "chosen_label",
                     "kept_copy_label", "labels_by_copy", "coralnet_id", "hash"]]
    review.to_csv(args.out_dir / "review_points.csv", index=False)

    per_img = locs.groupby(["kept_source_id", "kept_image_id"]).agg(
        n_points=("review", "size"), n_flagged=("review", "sum"),
        n_contradictory=("category", lambda s: int((s == "contradictory").sum())),
        n_compatible=("category", lambda s: int((s == "compatible").sum())),
        n_unknown=("category", lambda s: int((s == "unknown").sum())),
        n_unmapped=("has_unmapped", "sum"),
    )
    per_img = per_img[per_img["n_flagged"] > 0].reset_index()
    copies = dup.groupby("hash").apply(
        lambda g: ", ".join(f"{s}/{i}" for s, i in zip(g["source_id"], g["image_id"], strict=True)),
        include_groups=False).rename("copies")
    per_img = per_img.merge(kept.set_index(["kept_source_id", "kept_image_id"])["hash"].reset_index(),
                            on=["kept_source_id", "kept_image_id"]).merge(copies, left_on="hash", right_index=True)
    per_img = per_img.sort_values("n_flagged", ascending=False)
    per_img.to_csv(args.out_dir / "review_images.csv", index=False)

    # 3. Near duplicates labelled in Near_Duplicate_Review.ipynb, on the images kept so far.
    n_points_after = int(unique["n_points"].sum() + len(locs))
    nd_summary = {}
    if args.near_dup_labels.exists():
        labels = load_labels(args.near_dup_labels)
        exact_kept = dict(zip(blacklist["image_id"], blacklist["kept_image_id"], strict=True))
        labelled = {exact_kept.get(i, i) for lvl in (0, 1) for i in labels.index.get_level_values(lvl)}
        alive = pd.concat([unique, dup[dup["keep"]]])[["source_id", "image_id", "hash", "image_s3_key"]]
        cand = alive[alive["image_id"].isin(labelled)]
        dup_pts = locs[locs["kept_image_id"].isin(cand["image_id"])].rename(columns={"kept_image_id": "image_id"})
        uniq_pts = ann[ann["image_id"].isin(set(cand["image_id"]) - set(dup_pts["image_id"]))].assign(
            category="same_label", has_unmapped=False, in_kept=True, n_copies_at_loc=1)
        cand_points = pd.concat([dup_pts, uniq_pts], ignore_index=True)[["image_id", *POINT_COLUMNS]]
        nd = merge_near_duplicates(cand, cand_points, labels, exact_kept, root, resolve, workers=args.workers)
        if len(nd["blacklist"]):
            nd_blacklist, nd_overrides, nd_copies, nd_prov = (
                nd[k] for k in ("blacklist", "overrides", "copies", "provenance"))
            blacklist = pd.concat([blacklist, nd_blacklist], ignore_index=True)
            overrides = pd.concat([overrides[~overrides["image_id"].isin(nd_overrides["image_id"])], nd_overrides],
                                  ignore_index=True)
            group_images = set(nd_blacklist["image_id"]) | set(nd_overrides["image_id"])
            n_points_after += len(nd_overrides) - int(cand_points["image_id"].isin(group_images).sum())
            nd_copies.to_parquet(args.out_dir / "near_dup_copies.parquet", index=False)
            nd_prov.to_parquet(args.out_dir / "near_dup_point_provenance.parquet", index=False)

            # Review file: locations where a removed copy's label met another label and was flagged.
            ids = pd.concat([nd_copies[["group", "copy_rank", "source_id", "image_id"]],
                             nd_prov.drop_duplicates("group")[["group", "source_id", "image_id"]].assign(copy_rank=0)])
            name_of = {(g, r): f"{s}/{i}" for g, r, s, i in ids[["group", "copy_rank", "source_id", "image_id"]]
                       .itertuples(index=False)}
            touched = nd_prov["copy_ranks"].map(lambda r: max(r) > 0 and len(r) > 1)
            flagged = nd_prov[touched & (nd_prov["category"].isin(REVIEW_CATEGORIES) | nd_prov["has_unmapped"])].copy()
            flagged["labels_by_copy"] = [
                " | ".join(f"{name_of[g, r]}: {id2name.get(lid, '?')}" for r, lid in zip(rs, ls, strict=True))
                for g, rs, ls in flagged[["group", "copy_ranks", "copy_labels"]].itertuples(index=False)]
            flagged["chosen_label"] = flagged["coralnet_id"].map(id2name)
            flagged["kept_copy_label"] = flagged["kept_label"].map(id2name)
            flagged[["group", "source_id", "image_id", "row", "col", "category", "has_unmapped", "chosen_label",
                     "kept_copy_label", "labels_by_copy", "coralnet_id", "hash"]].to_csv(
                args.out_dir / "near_dup_review_points.csv", index=False)
            nd_summary = {
                "near_dup_labels": str(args.near_dup_labels),
                "n_near_dup_groups": int(nd_copies["group"].nunique()),
                "n_near_dup_removed": len(nd_blacklist),
                "near_dup_transforms": nd_copies["transform"].value_counts().to_dict(),
                "n_near_dup_points_transferred": int(nd_copies["n_transferred"].sum()),
                "n_near_dup_points_joined": int(nd_copies["n_joined"].sum()),
                "n_near_dup_points_outside": int(nd_copies["n_outside"].sum()),
                "n_near_dup_points_dropped_no_transform": int(
                    nd_copies.loc[nd_copies["transform"] == "failed", "n_points"].sum()),
                "near_dup_changes": nd_prov["change"].value_counts().to_dict(),
                "n_near_dup_review_points": len(flagged),
            }
    blacklist = blacklist.astype({"source_id": "int32", "kept_source_id": "int32"})
    blacklist.to_parquet(args.config_dir / "image_blacklist.parquet", index=False)
    overrides = overrides.astype({"source_id": "int32", "row": "int16", "col": "int16", "coralnet_id": "int32",
                                  "has_unmapped": bool, "in_kept": bool, "n_copies_at_loc": "int64"})
    overrides.to_parquet(args.config_dir / "annotation_overrides.parquet", index=False)

    summary = {
        "split_config": str(args.split_config),
        "excluded_sources": sorted(excluded),
        "n_sources": len(sources),
        "n_images": len(hashes),
        "n_unique_images": int(hashes["hash"].nunique()),
        "n_images_without_duplicate": len(unique),
        "n_duplicated_images": len(kept),
        "n_duplicate_copies": len(dup),
        "n_blacklisted": len(blacklist),
        "n_blacklisted_exact": int((blacklist["reason"] == "exact").sum()),
        "n_points_before": int(n_points["n_points"].sum()),
        "n_points_after": n_points_after,
        "n_overridden_images": len(changed_hashes),
        "n_points_added_from_other_copies": int((~locs["in_kept"]).sum()),
        "n_kept_labels_replaced": int((locs["in_kept"] & (locs["coralnet_id"] != locs["kept_label"])).sum()),
        "locations_by_category": locs["category"].value_counts().to_dict(),
        "n_locations_with_unmapped": int(locs["has_unmapped"].sum()),
        "n_review_points": len(review),
        "n_review_images": len(per_img),
        **nd_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
