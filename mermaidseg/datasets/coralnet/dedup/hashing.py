"""Measure image / annotation overlap between CoralNet sources.

CoralNet ``image_id`` values are global, so the same photo uploaded to two sources shows up under
two different ids. Duplicates are therefore found by **content hash**: every image file is hashed
once (O(N) over all images), an inverted index ``hash -> [(source_id, image_id), ...]`` is built,
and only source pairs that actually share a hash are materialised. Annotation overlap is computed
the same way, by packing each point ``(row, col)`` into a single int and intersecting the per-image
integer arrays.

For each pair of sources (A, B) this reports:

- ``n_images_a`` / ``n_images_b``: images in A / B (as listed in the training manifest).
- ``n_overlap_images``: images whose content hash appears in both A and B.
- For the overlapping images (averaged over matched image pairs):
  ``mean_ann_a``, ``mean_ann_b``: annotations per image in A / B;
  ``mean_ann_overlap``: points present at the same ``(row, col)`` in both;
  ``mean_ann_overlap_same_label``: of those, points that also carry the same ``coralnet_id``.

Hash modes:

- ``exact`` (default): BLAKE2b of the file bytes. Catches byte-identical re-uploads only.
- ``perceptual``: 64-bit difference hash (dHash) of the decoded image. Also catches re-encoded or
  re-compressed copies, at the cost of decoding every image and a small false-positive risk.

Images are read from the local S3 mirror (``MERMAIDSEG_LOCAL_CACHE_DIR``), using the manifest's
``image_s3_key`` so the hashed file is the one training sees, and annotation coordinates (which
live in that file's pixel space) are comparable between duplicates.

Usage:
    python -m mermaidseg.datasets.coralnet.dedup.hashing --all-sources
    python -m mermaidseg.datasets.coralnet.dedup.hashing --sources 1645 1776 1783 3058 3466
    python -m mermaidseg.datasets.coralnet.dedup.hashing --n-sources 5 --hash-mode perceptual
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mermaidseg.datasets.coralnet.dedup.common import (
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUTS,
    DEFAULT_SPLIT_CONFIG,
    add_split_labels,
    coralnet_splits,
)

# (row, col) are int16 in the manifest, so packing into one int64 is collision free.
_COORD_SHIFT = 1 << 16


def exact_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path, size: int = 8) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im.draft("L", (size * 16, size * 16))  # fast JPEG downscale-on-decode
        small = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def load_annotations(manifest_path: Path, sources: list[int] | None) -> pd.DataFrame:
    filters = [("source_id", "in", sources)] if sources else None
    return pq.read_table(
        manifest_path,
        columns=["source_id", "image_id", "row", "col", "coralnet_id", "image_s3_key"],
        filters=filters,
    ).to_pandas()


def hash_images(df_images: pd.DataFrame, root: Path, mode: str, workers: int) -> pd.Series:
    fn = exact_hash if mode == "exact" else perceptual_hash

    def _one(key: str) -> str | None:
        path = root / key
        try:
            return fn(path)
        except (OSError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(_one, df_images["image_s3_key"], chunksize=64))
    return pd.Series(hashes, index=df_images.index, name="hash")


def per_image_points(df_ann: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Map (source_id, image_id) -> (sorted packed coords, labels aligned with them)."""
    packed = df_ann["row"].to_numpy(np.int64) * _COORD_SHIFT + df_ann["col"].to_numpy(np.int64)
    df = pd.DataFrame(
        {
            "source_id": df_ann["source_id"].to_numpy(),
            "image_id": df_ann["image_id"].to_numpy(),
            "packed": packed,
            "label": df_ann["coralnet_id"].to_numpy(),
        }
    ).sort_values(["source_id", "image_id", "packed"])
    out = {}
    for key, g in df.groupby(["source_id", "image_id"], sort=False):
        out[key] = (g["packed"].to_numpy(), g["label"].to_numpy())
    return out


def compare_points(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
    """Return (#shared (row, col) points, #shared points with the same label)."""
    _, ia, ib = np.intersect1d(a[0], b[0], assume_unique=False, return_indices=True)
    return len(ia), int((a[1][ia] == b[1][ib]).sum())


def add_overlap_pcts(
    summary: pd.DataFrame, df_pairs: pd.DataFrame, n_images: pd.Series, n_ann: pd.Series
) -> pd.DataFrame:
    """Add overlap percentages, taking the larger of the two per-source ratios.

    - ``pct_image_overlap``: shared images / images in the source.
    - ``pct_ann_overlap``: shared points / all annotations in the source.
    - ``pct_ann_overlap_in_shared``: shared points / annotations on the shared images only.
    - ``pct_ann_overlap_same_label``: shared points that also agree on the label / shared points.
    """
    sums = (
        df_pairs.groupby(["source_a", "source_b"])[["ann_a", "ann_b", "ann_overlap", "ann_overlap_same_label"]]
        .sum()
        .reset_index()
    )
    s = summary.merge(sums, on=["source_a", "source_b"], how="left")
    ov_img = s["n_overlap_images"]
    ov_ann = s["ann_overlap"].astype(float).fillna(0)
    s["pct_image_overlap"] = 100 * np.maximum(
        ov_img / s["source_a"].map(n_images), ov_img / s["source_b"].map(n_images)
    )
    s["pct_ann_overlap"] = 100 * np.maximum(ov_ann / s["source_a"].map(n_ann), ov_ann / s["source_b"].map(n_ann))
    s["pct_ann_overlap_in_shared"] = 100 * np.maximum(ov_ann / s["ann_a"], ov_ann / s["ann_b"])
    s["pct_ann_overlap_same_label"] = 100 * s["ann_overlap_same_label"] / s["ann_overlap"].where(ov_ann > 0)
    return s.drop(columns=["ann_a", "ann_b", "ann_overlap", "ann_overlap_same_label"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=int, nargs="+", help="CoralNet source ids to compare.")
    parser.add_argument(
        "--n-sources", type=int, default=5, help="If --sources is omitted, take the N largest sources."
    )
    parser.add_argument("--all-sources", action="store_true", help="Compare every source in the manifest.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest key relative to the bucket.")
    parser.add_argument("--hash-mode", choices=["exact", "perceptual"], default="exact")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG,
        help="Data config whose CoralNet train/val/test whitelists label each source's split.",
    )
    args = parser.parse_args()

    root = Path(args.cache_root) / args.bucket
    manifest_path = root / args.manifest

    if args.all_sources:
        sources = sorted(pq.read_table(manifest_path, columns=["source_id"])["source_id"].unique().to_pylist())
    elif args.sources:
        sources = sorted(set(args.sources))
    else:
        counts = pq.read_table(manifest_path, columns=["source_id", "image_id"]).to_pandas()
        counts = counts.drop_duplicates().groupby("source_id").size().sort_values(ascending=False)
        sources = sorted(counts.index[: args.n_sources].tolist())
    print(f"Sources: {sources}" if len(sources) <= 20 else f"Sources: all {len(sources)}")

    df_ann = load_annotations(manifest_path, None if args.all_sources else sources)
    df_images = df_ann[["source_id", "image_id", "image_s3_key"]].drop_duplicates(["source_id", "image_id"])
    df_images = df_images.reset_index(drop=True)
    print(f"Loaded {len(df_ann):,} annotations on {len(df_images):,} images")

    df_images["hash"] = hash_images(df_images, root, args.hash_mode, args.workers)
    n_missing = df_images["hash"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing:,} images missing/unreadable in the local cache; they are excluded")
    df_images = df_images.dropna(subset=["hash"])

    n_images = df_images.groupby("source_id").size().reindex(sources, fill_value=0)
    n_ann = df_ann.groupby("source_id").size().reindex(sources, fill_value=0)

    # Within-source duplicates: keep the first image per (source, hash) for cross-source matching.
    dup_within = df_images.duplicated(["source_id", "hash"]).groupby(df_images["source_id"]).sum()
    first = df_images.drop_duplicates(["source_id", "hash"])

    # Inverted index: only hashes that occur in >= 2 sources can produce a pair.
    hash_to_imgs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for s, i, h in zip(first["source_id"], first["image_id"], first["hash"], strict=True):
        hash_to_imgs[h].append((s, i))
    shared = {h: v for h, v in hash_to_imgs.items() if len(v) > 1}
    print(f"{len(shared):,} hashes shared by >= 2 sources")

    # Only images that take part in a cross-source match need their points grouped.
    involved = pd.DataFrame([si for v in shared.values() for si in v], columns=["source_id", "image_id"])
    df_ann = df_ann.merge(involved, on=["source_id", "image_id"])
    points = per_image_points(df_ann)
    pair_rows = []
    for h, imgs in shared.items():
        for (sa, ia), (sb, ib) in itertools.combinations(sorted(imgs), 2):
            pa, pb = points[(sa, ia)], points[(sb, ib)]
            n_pts, n_same = compare_points(pa, pb)
            pair_rows.append(
                {
                    "source_a": sa,
                    "image_a": ia,
                    "source_b": sb,
                    "image_b": ib,
                    "hash": h,
                    "ann_a": len(pa[0]),
                    "ann_b": len(pb[0]),
                    "ann_overlap": n_pts,
                    "ann_overlap_same_label": n_same,
                }
            )
    df_pairs = pd.DataFrame(
        pair_rows,
        columns=[
            "source_a", "image_a", "source_b", "image_b", "hash",
            "ann_a", "ann_b", "ann_overlap", "ann_overlap_same_label",
        ],
    )

    agg = (
        df_pairs.groupby(["source_a", "source_b"])
        .agg(
            n_overlap_images=("hash", "size"),
            mean_ann_a=("ann_a", "mean"),
            mean_ann_b=("ann_b", "mean"),
            mean_ann_overlap=("ann_overlap", "mean"),
            mean_ann_overlap_same_label=("ann_overlap_same_label", "mean"),
        )
        .reset_index()
    )
    if len(sources) <= 20:
        all_pairs = pd.DataFrame(list(itertools.combinations(sources, 2)), columns=["source_a", "source_b"])
        summary = all_pairs.merge(agg, on=["source_a", "source_b"], how="left")
    else:  # listing every zero-overlap pair is quadratic in the number of sources
        summary = agg.sort_values("n_overlap_images", ascending=False)
    summary["n_overlap_images"] = summary["n_overlap_images"].fillna(0).astype(int)
    summary.insert(2, "n_images_a", summary["source_a"].map(n_images).to_numpy())
    summary.insert(3, "n_images_b", summary["source_b"].map(n_images).to_numpy())
    summary = add_overlap_pcts(summary, df_pairs, n_images, n_ann)
    summary = add_split_labels(summary, coralnet_splits(args.split_config))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.all_sources:
        tag = f"{args.hash_mode}_all"
    elif len(sources) <= 10:
        tag = f"{args.hash_mode}_{'-'.join(map(str, sources))}"
    else:
        tag = f"{args.hash_mode}_{len(sources)}src"
    df_images.to_parquet(args.out_dir / f"image_hashes_{tag}.parquet", index=False)
    summary.to_csv(args.out_dir / f"source_overlap_{tag}.csv", index=False)
    df_pairs.to_csv(args.out_dir / f"source_overlap_pairs_{tag}.csv", index=False)

    print("\nImages per source (within-source duplicate images in brackets):")
    for s in sources if len(sources) <= 20 else []:
        print(f"  {s}: {n_images[s]:,} [{int(dup_within.get(s, 0))}]")
    per_source = pd.DataFrame(
        {"n_images": n_images, "n_annotations": n_ann, "n_dup_within": dup_within.reindex(sources, fill_value=0)}
    )
    per_source.rename_axis("source_id").to_csv(args.out_dir / f"source_images_{tag}.csv")
    print("\nPairwise overlap:")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print(summary.head(50).to_string(index=False))
    print(f"\nWrote {args.out_dir}/source_overlap_{tag}.csv and source_overlap_pairs_{tag}.csv")


if __name__ == "__main__":
    main()
