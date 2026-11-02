"""Remove the near duplicates labelled ``dup`` in ``Near_Duplicate_Review.ipynb`` and merge their points.

Called by ``dedup`` after the exact deduplication, on the images it keeps. Near duplicates are not
byte-identical (re-encoded, resized or cropped re-uploads), so their points do not share a
coordinate space: they are mapped onto the kept copy first.

1. Groups: connected components of the pairs whose latest label is ``dup`` (``fp`` and ``unsure``
   pairs are left alone). A labelled image that exact deduplication has since removed stands for
   the copy it was merged into.
2. Kept copy: the largest image of the group (the least cropped / downscaled, so every other
   copy's points land on it at full precision), then the one with more points, the newer source
   and the lower image id. The others go to the image blacklist (``reason = "near_dup"``).
3. Transforms: for every labelled ``dup`` pair of a group, the homography between the two images
   (ORB + RANSAC, as in ``near_duplicates.verify_pair``), scaled to full resolution. Same-size
   pairs whose homography moves no image corner by more than ``identity_tol`` of the diagonal are
   taken as aligned (identity). A pair with fewer than ``min_inliers`` RANSAC inliers, or whose
   homography does not map the image to a convex quadrilateral, has no transform. Each copy is
   mapped onto the kept one along a shortest path of pairs with a transform; copies without such a
   path are still removed, but their points are dropped.
4. Points: each copy's points are mapped onto the kept copy, rounded to pixels, and dropped if they
   fall outside it. A point joins a location already annotated (by the kept copy or an earlier copy)
   at the same pixel (identity) or within ``snap_px`` (homography); otherwise it adds one. On the
   labelled pairs, aligned copies either reuse the exact coordinates or none at all, while after a
   homography the labels of the nearest points agree ~80% within 3 px and drop to the ~48% of
   unrelated nearby points beyond that: copies annotated independently get no joins.
   Where several labels meet, they are resolved with ``dedup.make_resolver`` (copies ordered kept
   first, then by path length), exactly as for byte-identical copies.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from mermaidseg.datasets.coralnet.dedup.common import CONFIG_DIR

DEFAULT_LABELS = CONFIG_DIR / "near_dup_labels.csv"
LABEL_COLUMNS = ["image_a", "image_b", "label", "pair", "group", "call", "score", "source_a", "split_a",
                 "source_b", "split_b", "labelled_at"]
POINT_COLUMNS = ["row", "col", "coralnet_id", "category", "has_unmapped", "in_kept", "n_copies_at_loc"]


def load_labels(path: Path | str) -> pd.DataFrame:
    """Latest label per pair (rows of the append-only labels CSV), keyed by ``(image_a, image_b)``."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS).set_index(["image_a", "image_b"])
    d = pd.read_csv(path, dtype={"image_a": str, "image_b": str})
    return d.drop_duplicates(["image_a", "image_b"], keep="last").set_index(["image_a", "image_b"])


def _features(path: Path):
    """``(width, height)`` of the file and its ``near_duplicates._load_features``; ``None`` if unreadable."""
    from PIL import Image

    from mermaidseg.datasets.coralnet.dedup.near_duplicates import _load_features

    try:
        with Image.open(path) as im:
            size = im.size
        return size, _load_features(path)
    except (OSError, ValueError):
        return None


def _corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def _apply(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def pair_transform(fa, fb, min_inliers: int, identity_tol: float) -> tuple[np.ndarray | None, int, str]:
    """Full-resolution map from image a onto image b: ``(H, inliers, kind)`` with ``kind`` one of
    ``identity`` / ``homography`` / ``failed`` (``H`` is ``None`` for ``failed``)."""
    from mermaidseg.datasets.coralnet.dedup.near_duplicates import match_homography

    if fa is None or fb is None:
        return None, 0, "failed"
    (size_a, feat_a), (size_b, feat_b) = fa, fb
    H, inliers = match_homography(feat_a, feat_b)
    if H is None or inliers < min_inliers:
        return None, inliers, "failed"

    def scale(feat, size):  # full resolution -> _load_features pixels
        return np.diag([feat[0].shape[1] / size[0], feat[0].shape[0] / size[1], 1.0])

    H = np.linalg.inv(scale(feat_b, size_b)) @ H @ scale(feat_a, size_a)
    quad = _apply(H, _corners(*size_a))
    edges = np.roll(quad, -1, axis=0) - quad
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (np.all(turns > 0) or np.all(turns < 0)) or not np.isfinite(quad).all():
        return None, inliers, "failed"
    if size_a == size_b and np.abs(quad - _corners(*size_a)).max() <= identity_tol * np.hypot(*size_a):
        return np.eye(3), inliers, "identity"
    return H, inliers, "homography"


def merge_near_duplicates(
    images: pd.DataFrame,
    points: pd.DataFrame,
    labels: pd.DataFrame,
    exact_kept: dict[str, str],
    bucket_root: Path,
    resolve: Callable[[tuple[int, ...]], tuple[int, str, bool]],
    *,
    min_inliers: int = 15,
    identity_tol: float = 0.01,
    snap_px: float = 3.0,
    workers: int = 32,
) -> dict[str, pd.DataFrame]:
    """Merge the labelled near-duplicate groups among ``images``.

    ``images``: the images left after exact deduplication (``source_id``, ``image_id``, ``hash``,
    ``image_s3_key``). ``points``: their points (``image_id`` + ``POINT_COLUMNS``). ``labels``:
    ``load_labels``. ``exact_kept``: exact-blacklisted image id -> the copy kept in its place.

    Returns ``blacklist`` (the removed copies, as rows of ``image_blacklist.parquet``),
    ``overrides`` (every point of every kept copy, as rows of ``annotation_overrides.parquet``),
    ``copies`` (one row per group member: kept copy, transform, points transferred / dropped) and
    ``provenance`` (one row per location of every kept copy, with every copy's label there).
    """
    src_of = dict(zip(images["image_id"], images["source_id"].astype(int), strict=True))
    hash_of = dict(zip(images["image_id"], images["hash"], strict=True))
    key_of = dict(zip(images["image_id"], images["image_s3_key"], strict=True))
    lab = labels.reset_index()
    for side in ("image_a", "image_b"):
        lab[side] = lab[side].map(lambda i: exact_kept.get(i, i))
    dup = lab[(lab["label"] == "dup") & lab["image_a"].isin(src_of) & lab["image_b"].isin(src_of)
              & (lab["image_a"] != lab["image_b"])]
    graph = nx.Graph()
    graph.add_edges_from(dup[["image_a", "image_b"]].to_numpy())
    comps = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
    members = [i for c in comps for i in c]
    if not members:
        return {k: pd.DataFrame() for k in ("blacklist", "overrides", "copies", "provenance")}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        feats = dict(zip(members, ex.map(lambda i: _features(bucket_root / key_of[i]), members),
                         strict=True))
    pts = points[points["image_id"].isin(members)].drop_duplicates(["image_id", "row", "col", "coralnet_id"])
    pts_by_img = dict(iter(pts.groupby("image_id")))
    no_points = points.iloc[:0]

    blacklist, copies, prov = [], [], []
    for gid, comp in enumerate(comps):
        def keep_order(i):
            size = feats[i][0] if feats[i] is not None else (0, 0)
            n = len(pts_by_img.get(i, no_points))
            return -size[0] * size[1], -n, -src_of[i], int(i)

        kept = min(comp, key=keep_order)
        kept_size = feats[kept][0] if feats[kept] is not None else None

        # Transforms onto the kept copy along shortest paths of usable pairs.
        tg = nx.DiGraph()
        tg.add_nodes_from(comp)
        pair_info = {}
        for a, b in sorted(tuple(sorted(e)) for e in graph.subgraph(comp).edges()):  # subgraph order is random
            H, inl, kind = pair_transform(feats[a], feats[b], min_inliers, identity_tol)
            pair_info[a, b] = pair_info[b, a] = (inl, kind)
            if H is not None:
                tg.add_edge(a, b, H=H)
                tg.add_edge(b, a, H=np.linalg.inv(H))
        paths = nx.single_source_shortest_path(tg, kept)  # kept -> ... -> copy
        to_kept = {}
        for i, path in paths.items():
            H = np.eye(3)
            for src, dst in zip(path[:0:-1], path[-2::-1], strict=True):  # copy -> ... -> kept
                H = tg.edges[src, dst]["H"] @ H
            to_kept[i] = H
        order = sorted(comp, key=lambda i: (i != kept, len(paths.get(i, comp)), keep_order(i)))
        rank = {i: r for r, i in enumerate(order)}

        # Locations: the kept copy's own points first, then every other copy's mapped points.
        kp = pts_by_img.get(kept, no_points)
        loc_xy = [kp[["col", "row"]].to_numpy(float)]
        loc_labels = [[(0, lid)] for lid in kp["coralnet_id"]]
        own = kp.reset_index(drop=True)
        for i in order[1:]:
            p = pts_by_img.get(i, no_points)
            info = {"group": gid, "image_id": i, "source_id": src_of[i], "copy_rank": rank[i],
                    "kept_image_id": kept, "kept_source_id": src_of[kept],
                    "path_length": len(paths[i]) - 1 if i in paths else None, "n_points": len(p),
                    "n_transferred": 0, "n_joined": 0, "n_outside": 0}
            nb = [n for n in graph.neighbors(i) if n in comp]
            info["pair_kinds"] = ";".join(sorted({pair_info[i, n][1] for n in nb}))
            info["transform"] = ("failed" if i not in to_kept else
                                 "identity" if np.allclose(to_kept[i], np.eye(3)) else "homography")
            copies.append(info)
            if i not in to_kept or not len(p) or kept_size is None:
                continue
            xy = np.rint(_apply(to_kept[i], p[["col", "row"]].to_numpy(float)))
            tol = 0.5 if info["transform"] == "identity" else snap_px
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < kept_size[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < kept_size[1])
            info["n_outside"] = int((~inside).sum())
            existing = np.concatenate(loc_xy)
            new = []
            for (x, y), lid in zip(xy[inside], p["coralnet_id"].to_numpy()[inside], strict=True):
                if len(existing):
                    d = np.hypot(existing[:, 0] - x, existing[:, 1] - y)
                    j = int(d.argmin())
                    if d[j] <= tol:
                        loc_labels[j].append((rank[i], lid))
                        info["n_joined"] += 1
                        continue
                new.append((x, y))
                loc_labels.append([(rank[i], lid)])
            info["n_transferred"] = int(inside.sum())
            if new:
                loc_xy.append(np.array(new))

        # Resolve every location.
        xy = np.concatenate(loc_xy).astype(int)
        rows = []
        for k, labs in enumerate(loc_labels):
            labs = sorted(labs)
            distinct = tuple(dict.fromkeys(lid for _, lid in labs))
            in_kept = k < len(own)
            if in_kept and len(labs) == 1:  # untouched point of the kept copy: keep its exact-merge record
                rec = own.loc[k, POINT_COLUMNS].to_dict()
                rec["change"] = "unchanged"
            else:
                lid, cat, unm = resolve(distinct)
                base = own.loc[k] if in_kept else None
                rec = {"row": xy[k, 1], "col": xy[k, 0], "coralnet_id": lid,
                       "category": cat if len(distinct) > 1 or base is None else base["category"],
                       "has_unmapped": bool(unm or (base is not None and base["has_unmapped"])),
                       "in_kept": bool(base["in_kept"]) if base is not None else False,
                       "n_copies_at_loc": (int(base["n_copies_at_loc"]) if base is not None else 0)
                       + len({r for r, _ in labs if r > 0})}
                rec["change"] = ("added" if base is None else
                                 "replaced" if lid != base["coralnet_id"] else "unchanged")
            rec.update(group=gid, image_id=kept, source_id=src_of[kept],
                       hash=hash_of[kept], kept_label=int(own.loc[k, "coralnet_id"]) if in_kept else None,
                       copy_ranks=[r for r, _ in labs], copy_labels=[int(lid) for _, lid in labs])
            rows.append(rec)
        prov.append(pd.DataFrame(rows))
        for i in order[1:]:
            blacklist.append({"source_id": src_of[i], "image_id": i, "hash": hash_of[i],
                              "kept_source_id": src_of[kept], "kept_image_id": kept,
                              "reason": "near_dup"})

    out_cols = ["source_id", "image_id", *POINT_COLUMNS, "hash"]
    prov = pd.concat(prov, ignore_index=True)
    return {
        "blacklist": pd.DataFrame(blacklist),
        "overrides": prov[out_cols],
        "copies": pd.DataFrame(copies),
        "provenance": prov,
    }
