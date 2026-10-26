"""Find near-duplicate CoralNet images: copies that are not byte-identical (re-encoded, resized,
re-cropped, colour-corrected or rotated re-uploads), plus burst frames of the same scene.

Builds on ``hashing --all-sources`` (``image_hashes_exact_all.parquet``): byte-identical copies are
already grouped, so only one file per exact hash is processed.

1. ``hash`` (~25 min, disk bound): decode each representative file into a 32x32 grayscale
   thumbnail (aspect ratio squashed, so a rotated copy gives the rotated thumbnail). Resumable:
   thumbnails are written in chunks to ``<out-dir>/chunks/``.
2. ``candidates`` (minutes, GPU): every image against every other, and against the 90/180/270
   degree rotations of the other. A pair is a candidate if its 64-bit pHash (DCT of the thumbnail)
   is within ``--max-phash`` bits and the thumbnails correlate at least ``--min-thumb-corr``.
   Both are cheap but weak: images dominated by the same fixed rig (quadrat frame, settlement
   tile, colour bars) pass them too.
3. ``verify`` (~10-20 min): each candidate pair is decoded at 512 px, aligned with ORB keypoints
   and a RANSAC homography (handles crops, rescaling, rectification and rotation), high-pass
   filtered (drops colour / brightness changes) and compared patch by patch (32 px). ``score`` is
   the median patch correlation: a copy matches in nearly every patch, a different photo in the
   same rig only where the rig is.
4. ``report``: ``near_dup = score >= --dup-thr``, ``ambiguous`` for ``--ambiguous-thr <= score <
   --dup-thr``; groups (connected components of near duplicates) and leakage under
   ``--split-config``. Thresholds were set by eye: >= 0.6 is the same picture (copy or burst
   frame), 0.4-0.6 mixes cropped copies with different photos of an identical rig.

Outputs in ``<out-dir>`` (default ``data/coralnet_dedup/near_dup``):

- ``thumbs.npy`` (N x 32 x 32 uint8) and ``thumb_index.parquet`` (exact hash, representative file,
  original width / height, pHash), row i = thumbnail i;
- ``candidate_pairs.parquet`` / ``verified_pairs.parquet``: candidate pairs of exact hashes, the
  latter with ``inliers``, ``overlap`` and ``score``;
- ``near_dup_pairs.parquet``: verified pairs with the ``near_dup`` / ``ambiguous`` calls;
- ``near_dup_groups.parquet``: one row per copy (source_id, image_id) of every exact hash with a
  near duplicate, with its group, split and whether it is blacklisted;
- ``near_dup_source_pairs.csv``: source pairs linked by near duplicates, with splits;
- ``summary.json``.

Usage:
    python -m mermaidseg.datasets.coralnet.dedup.near_duplicates --stage all
    python -m mermaidseg.datasets.coralnet.dedup.near_duplicates --stage report --dup-thr 0.7
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.fft import dctn

from mermaidseg.datasets.coralnet.dedup.common import (
    CONFIG_DIR,
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_OUTPUTS,
    REPO_ROOT,
    add_split_labels,
    coralnet_splits,
    exact_blacklist,
)

THUMB = 32
CHUNK = 20_000
VERIFY_PX = 512
PATCH = 32
STAGES = ("hash", "candidates", "verify", "report")


# ----------------------------------------------------------------------------------------- hash
def thumbnail(path: Path) -> tuple[np.ndarray, int, int]:
    """32x32 grayscale thumbnail and the file's original (width, height)."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        im.draft("L", (THUMB * 4, THUMB * 4))  # JPEG downscale-on-decode
        return np.asarray(im.convert("L").resize((THUMB, THUMB), Image.Resampling.BOX)), w, h


def hash_stage(args: argparse.Namespace) -> None:
    root = Path(args.cache_root) / DEFAULT_BUCKET
    exact = pd.read_parquet(args.outputs / "image_hashes_exact_all.parquet")
    reps = exact.drop_duplicates("hash").reset_index(drop=True)[["hash", "source_id", "image_id", "image_s3_key"]]
    print(f"{len(exact):,} hashed images, {len(reps):,} distinct files to decode")
    chunk_dir = args.out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def _one(key: str):
        try:
            return thumbnail(root / key)
        except (OSError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for start in range(0, len(reps), CHUNK):
            out = chunk_dir / f"{start:07d}.npz"
            if out.exists():
                continue
            res = list(ex.map(_one, reps["image_s3_key"].iloc[start:start + CHUNK]))
            ok = np.array([r is not None for r in res])
            thumbs = np.zeros((len(res), THUMB, THUMB), np.uint8)
            wh = np.zeros((len(res), 2), np.int32)
            for i, r in enumerate(res):
                if r is not None:
                    thumbs[i], wh[i] = r[0], r[1:]
            np.savez(out, thumbs=thumbs, wh=wh, ok=ok)
            print(f"  {start + len(res):,}/{len(reps):,}", flush=True)

    parts = [np.load(chunk_dir / f"{s:07d}.npz") for s in range(0, len(reps), CHUNK)]
    thumbs = np.concatenate([p["thumbs"] for p in parts])
    wh = np.concatenate([p["wh"] for p in parts])
    ok = np.concatenate([p["ok"] for p in parts])
    print(f"{(~ok).sum():,} files unreadable; excluded")
    reps["width"], reps["height"] = wh[:, 0], wh[:, 1]
    reps, thumbs = reps[ok].reset_index(drop=True), thumbs[ok]
    reps["phash"] = pack_bits(phash_bits(thumbs))
    np.save(args.out_dir / "thumbs.npy", thumbs)
    reps.to_parquet(args.out_dir / "thumb_index.parquet", index=False)
    print(f"Wrote {args.out_dir}/thumbs.npy and thumb_index.parquet ({len(reps):,} images)")


def phash_bits(thumbs: np.ndarray) -> np.ndarray:
    """(N, 64) bool: low 8x8 DCT coefficients above their median (as in imagehash.phash)."""
    out = np.empty((len(thumbs), 64), bool)
    for s in range(0, len(thumbs), 50_000):
        low = dctn(thumbs[s:s + 50_000].astype(np.float32), axes=(1, 2), norm="ortho")[:, :8, :8]
        low = low.reshape(len(low), 64)
        out[s:s + 50_000] = low > np.median(low, axis=1, keepdims=True)
    return out


def pack_bits(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits, axis=1).view(">u8").ravel().astype(np.uint64)


def unpack_bits(h: np.ndarray) -> np.ndarray:
    return np.unpackbits(h.astype(">u8").view(np.uint8).reshape(len(h), 8), axis=1).astype(bool)


# ----------------------------------------------------------------------------------- candidates
def candidate_pairs(thumbs: np.ndarray, max_phash: int, min_corr: float, chunk: int = 2048) -> pd.DataFrame:
    """All (i, j, rot) with pHash(rot90(thumb_i, rot)) within ``max_phash`` bits of pHash(thumb_j)
    and thumbnail correlation >= ``min_corr``; i < j, best rotation per pair."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32

    def signs(b):  # +-1, so dot = 64 - 2 * hamming
        return torch.from_numpy(b.astype(np.float32) * 2 - 1).to(dev, dtype)

    def unit(t):  # zero-mean, unit-norm flattened thumbnails, so dot = Pearson correlation
        x = t.reshape(len(t), -1).astype(np.float32)
        x -= x.mean(1, keepdims=True)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
        return torch.from_numpy(x).to(dev, dtype)

    db_bits, db_thumb = signs(phash_bits(thumbs)), unit(thumbs)
    min_dot = 64 - 2 * max_phash
    found = []
    for rot in range(4):
        rt = np.ascontiguousarray(np.rot90(thumbs, rot, axes=(1, 2)))
        q_bits, q_thumb = signs(phash_bits(rt)), unit(rt)
        for s in range(0, len(thumbs), chunk):
            dot = q_bits[s:s + chunk] @ db_bits.T
            corr = q_thumb[s:s + chunk] @ db_thumb.T
            i, j = torch.nonzero((dot >= min_dot) & (corr >= min_corr), as_tuple=True)
            d, c = dot[i, j], corr[i, j]
            i = i + s
            keep = i != j
            found.append(pd.DataFrame({
                "i": i[keep].cpu().numpy(), "j": j[keep].cpu().numpy(), "rot": rot,
                "phash_dist": ((64 - d[keep].float()) / 2).round().int().cpu().numpy(),
                "thumb_corr": c[keep].float().cpu().numpy(),
            }))
            del dot, corr
    pairs = pd.concat(found, ignore_index=True)
    # Canonical order (i < j); rotating i by r equals rotating j by -r.
    swap = pairs["i"] > pairs["j"]
    pairs.loc[swap, ["i", "j"]] = pairs.loc[swap, ["j", "i"]].to_numpy()
    pairs.loc[swap, "rot"] = (4 - pairs.loc[swap, "rot"]) % 4
    pairs = pairs.sort_values(["thumb_corr"], ascending=False).drop_duplicates(["i", "j"])
    return pairs.sort_values(["i", "j"]).reset_index(drop=True)


def candidates_stage(args: argparse.Namespace) -> None:
    idx = pd.read_parquet(args.out_dir / "thumb_index.parquet")
    thumbs = np.load(args.out_dir / "thumbs.npy")
    pairs = candidate_pairs(thumbs, args.max_phash, args.min_thumb_corr)
    for side in ("i", "j"):
        rep = idx.iloc[pairs[side].to_numpy()]
        for col in ("hash", "source_id", "image_id", "width", "height"):
            pairs[f"{col}_{side}"] = rep[col].to_numpy()
    pairs.to_parquet(args.out_dir / "candidate_pairs.parquet", index=False)
    print(f"{len(pairs):,} candidate pairs (pHash <= {args.max_phash}, thumbnail corr >= {args.min_thumb_corr})")


# --------------------------------------------------------------------------------------- verify
def _load_features(path: Path):
    import cv2
    from PIL import Image

    with Image.open(path) as im:
        im.draft("L", (VERIFY_PX, VERIFY_PX))
        im = im.convert("L")
        im.thumbnail((VERIFY_PX, VERIFY_PX), Image.Resampling.BOX)
        a = np.asarray(im)
    kp, des = cv2.ORB_create(nfeatures=2000).detectAndCompute(a, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return a, pts, des


def _highpass(a: np.ndarray) -> np.ndarray:
    import cv2

    a = a.astype(np.float32)
    return a - cv2.GaussianBlur(a, (0, 0), 3)


def _patch_median(a: np.ndarray, b: np.ndarray, valid: np.ndarray, min_patches: int = 8) -> tuple[float, int]:
    """Median over fully covered PATCH x PATCH tiles of the correlation between a and b."""
    ny, nx = b.shape[0] // PATCH, b.shape[1] // PATCH

    def tiles(x):
        return x[:ny * PATCH, :nx * PATCH].reshape(ny, PATCH, nx, PATCH).transpose(0, 2, 1, 3).reshape(ny * nx, -1)

    ok = tiles(valid.astype(np.float32)).mean(1) >= 0.95
    x, y = tiles(a)[ok], tiles(b)[ok]
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
    r = (x * y).sum(1)[den > 1e-3] / den[den > 1e-3]
    return (float(np.median(r)) if len(r) >= min_patches else float("nan")), len(r)


def match_homography(fa, fb) -> tuple[np.ndarray | None, int]:
    """Homography mapping image a onto image b (in ``_load_features`` pixels) and its RANSAC
    inliers (the number of ratio-test matches if there are too few to fit one)."""
    import cv2

    (_, pa, da), (_, pb, db) = fa, fb
    if da is None or db is None or len(da) < 20 or len(db) < 20:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 12:
        return None, len(good)
    H, mask = cv2.findHomography(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]], cv2.RANSAC, 4.0)
    if H is None:
        return None, 0
    return H, int(mask.sum())


def verify_pair(fa, fb) -> tuple[int, float, float]:
    """(RANSAC inliers, overlap of the warped image a on b, aligned high-pass patch median)."""
    import cv2

    (a, _, _), (b, _, _) = fa, fb
    H, inliers = match_homography(fa, fb)
    if H is None:
        return inliers, 0.0, float("nan")
    size = (b.shape[1], b.shape[0])
    warped = cv2.warpPerspective(_highpass(a), H, size)
    valid = cv2.warpPerspective(np.ones_like(a, np.float32), H, size) > 0.5
    score, _ = _patch_median(warped, _highpass(b), valid)
    return inliers, float(valid.mean()), score


def verify_stage(args: argparse.Namespace) -> None:
    root = Path(args.cache_root) / DEFAULT_BUCKET
    pairs = pd.read_parquet(args.out_dir / "candidate_pairs.parquet")
    keys = pd.read_parquet(args.out_dir / "thumb_index.parquet").set_index("hash")["image_s3_key"]
    hashes = pd.unique(pairs[["hash_i", "hash_j"]].to_numpy().ravel())
    print(f"Verifying {len(pairs):,} pairs over {len(hashes):,} images")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        feats = dict(zip(hashes, ex.map(lambda h: _load_features(root / keys[h]), hashes), strict=True))
        # ORB features are rotation invariant, so the thumbnail rotation is not needed here.
        res = list(ex.map(lambda hh: verify_pair(feats[hh[0]], feats[hh[1]]),
                          zip(pairs["hash_i"], pairs["hash_j"], strict=True), chunksize=256))
    pairs[["inliers", "overlap", "score"]] = pd.DataFrame(res, index=pairs.index)
    pairs.to_parquet(args.out_dir / "verified_pairs.parquet", index=False)
    print(pd.cut(pairs["score"], [-1, 0, 0.2, 0.4, 0.6, 0.8, 0.9, 1]).value_counts().sort_index().to_string())


# --------------------------------------------------------------------------------------- report
def report_stage(args: argparse.Namespace) -> None:
    pairs = pd.read_parquet(args.out_dir / "verified_pairs.parquet")
    pairs["near_dup"] = pairs["score"] >= args.dup_thr
    pairs["ambiguous"] = pairs["score"].between(args.ambiguous_thr, args.dup_thr, inclusive="left")
    pairs["same_size"] = (pairs["width_i"] == pairs["width_j"]) & (pairs["height_i"] == pairs["height_j"])
    pairs["same_source"] = pairs["source_id_i"] == pairs["source_id_j"]
    pairs.to_parquet(args.out_dir / "near_dup_pairs.parquet", index=False)
    nd = pairs[pairs["near_dup"]]

    # Groups of exact hashes linked by near duplicates, expanded to every copy.
    graph = nx.Graph()
    graph.add_edges_from(nd[["hash_i", "hash_j"]].to_numpy())
    group_of = {h: g for g, comp in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True))
                for h in comp}
    exact = pd.read_parquet(args.outputs / "image_hashes_exact_all.parquet")
    copies = exact[exact["hash"].isin(group_of)].copy()
    copies["group"] = copies["hash"].map(group_of)
    splits = coralnet_splits(args.split_config)
    split_of = {s: k for k, v in splits.items() for s in v}
    copies["split"] = copies["source_id"].map(split_of).fillna("unused")
    blacklist = exact_blacklist(args.blacklist)
    copies["blacklisted"] = copies["image_id"].astype(str).isin(blacklist)
    copies.to_parquet(args.out_dir / "near_dup_groups.parquet", index=False)

    # Source pairs linked by a near-duplicate pair (any copy of either exact hash).
    srcs = exact.groupby("hash")["source_id"].agg(lambda s: sorted(set(s)))
    rows = [tuple(sorted((a, b))) for hi, hj in nd[["hash_i", "hash_j"]].itertuples(index=False)
            for a in srcs[hi] for b in srcs[hj] if a != b]
    sp = pd.DataFrame(rows, columns=["source_a", "source_b"]).value_counts().rename("n_near_dup_pairs").reset_index()
    sp = add_split_labels(sp, splits).sort_values("n_near_dup_pairs", ascending=False)
    sp.to_csv(args.out_dir / "near_dup_source_pairs.csv", index=False)

    # Leakage under the split config: groups whose surviving (non-blacklisted) copies span splits.
    live = copies[~copies["blacklisted"] & (copies["split"] != "unused")]
    n_splits = live.groupby("group")["split"].nunique()
    cross = live[live["group"].isin(n_splits.index[n_splits > 1])]
    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_candidate_pairs": int(len(pairs)),
        "n_near_dup_pairs": int(len(nd)),
        "n_ambiguous_pairs": int(pairs["ambiguous"].sum()),
        "n_near_dup_pairs_cross_source": int((~nd["same_source"]).sum()),
        "n_near_dup_pairs_rotated": int((nd["rot"] != 0).sum()),
        "n_near_dup_pairs_different_size": int((~nd["same_size"]).sum()),
        "n_groups": len(set(group_of.values())),
        "n_exact_hashes_in_groups": len(group_of),
        "n_source_pairs": int(len(sp)),
        "n_cross_split_source_pairs": int(sp["cross_split"].sum()),
        "n_groups_spanning_splits": int((n_splits > 1).sum()),
        "images_in_cross_split_groups": {
            s: int(cross.loc[cross["split"] == s, "image_id"].nunique()) for s in ("train", "val", "test")
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all",
                   help="Run one stage, or all of them in order.")
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS, help="Where hashing wrote its outputs.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS / "near_dup")
    p.add_argument("--split-config", type=Path, default=REPO_ROOT / "configs/data_config_dedup.yaml")
    p.add_argument("--blacklist", type=Path, default=CONFIG_DIR / "image_blacklist.parquet")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--max-phash", type=int, default=14, help="Candidates: pHash Hamming distance at most ...")
    p.add_argument("--min-thumb-corr", type=float, default=0.7, help="... and thumbnail correlation at least.")
    p.add_argument("--dup-thr", type=float, default=0.6, help="Near duplicate: verified score at least.")
    p.add_argument("--ambiguous-thr", type=float, default=0.4, help="Ambiguous: score in [this, --dup-thr).")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stages = {"hash": hash_stage, "candidates": candidates_stage, "verify": verify_stage, "report": report_stage}
    for name in STAGES if args.stage == "all" else (args.stage,):
        stages[name](args)


if __name__ == "__main__":
    main()
