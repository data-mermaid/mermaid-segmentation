"""Manual review of the perceptual near duplicates found by ``near_duplicates``.

Only images still in the dataset are reviewed: copies in ``image_blacklist.parquet`` removed as byte-identical
duplicates (``reason == "exact"``) and sources outside the splits of the data config (e.g. the low-quality
ones) are dropped first. Every near-duplicate (score >= ``dup_thr``) and ambiguous pair between two
remaining images is a review pair; review groups are the connected components of those pairs.

Groups are ordered by what is at stake: ``cross-split`` (a leak between train / val / test), then
``cross-source``, then ``same-source``; within that, groups with ambiguous pairs first.

Labels are given per pair with ``PairLabeler`` (ipywidgets: one pair at a time, buttons ``dup`` /
``fp`` (false positive) / ``unsure``). Every click appends a row to the labels CSV, so nothing is
lost if the kernel dies; the latest row of a pair wins (relabelling = clicking again). Pairs are
keyed by their CoralNet image ids, so the file stays valid if the outputs are regenerated.

Used by ``mermaidseg/datasets/coralnet/nbs/Near_Duplicate_Review.ipynb``.
"""

from __future__ import annotations

import io
import string
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from mermaidseg.datasets.coralnet.dedup.common import (
    CONFIG_DIR,
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_OUTPUTS,
    REPO_ROOT,
    coralnet_splits,
    exact_blacklist,
)
from mermaidseg.datasets.coralnet.dedup.near_dup_merge import LABEL_COLUMNS, load_labels
from mermaidseg.datasets.coralnet.dedup.near_duplicates import VERIFY_PX, _highpass, _load_features

CATEGORIES = ("cross-split", "cross-source", "same-source")
LABELS = ("dup", "fp", "unsure")


def _jpeg(fig: Figure, dpi: int = 72, quality: int = 75) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=dpi, pil_kwargs={"quality": quality})
    plt.close(fig)
    return buf.getvalue()


def _letters(n: int) -> list[str]:
    abc = string.ascii_uppercase
    return [abc[i] if i < 26 else abc[i // 26 - 1] + abc[i % 26] for i in range(n)]


class NearDupReview:
    """Loads the near-duplicate outputs once; ``select`` picks groups, ``show`` renders one."""

    def __init__(self, outputs: Path | str = DEFAULT_OUTPUTS, config_dir: Path | str = CONFIG_DIR,
                 split_config: Path | str = REPO_ROOT / "configs/data_config_dedup.yaml",
                 cache_root: str = DEFAULT_CACHE_ROOT):
        outputs = Path(outputs)
        self.root = Path(cache_root) / DEFAULT_BUCKET
        pairs = pd.read_parquet(outputs / "near_dup/near_dup_pairs.parquet")
        pairs = pairs[pairs["near_dup"] | pairs["ambiguous"]]

        # One remaining copy per exact hash (the blacklist keeps exactly one).
        exact = pd.read_parquet(outputs / "image_hashes_exact_all.parquet")
        exact["image_id"] = exact["image_id"].astype(str)
        blacklist = exact_blacklist(Path(config_dir) / "image_blacklist.parquet")
        split_of = {s: k for k, v in coralnet_splits(split_config).items() for s in v}
        live = exact[~exact["image_id"].isin(blacklist) & exact["source_id"].isin(split_of)]
        live = live.drop_duplicates("hash").set_index("hash")
        live["split"] = live["source_id"].map(split_of)
        meta = pd.read_csv(outputs / "coralnet_source_metadata.csv")
        self.source_name = dict(zip(meta["source_id"], meta["name"], strict=True))

        pairs = pairs[pairs["hash_i"].isin(live.index) & pairs["hash_j"].isin(live.index)].copy()
        for side, other in (("i", "a"), ("j", "b")):
            info = live.loc[pairs[f"hash_{side}"]]
            pairs[f"image_{other}"] = info["image_id"].to_numpy()
            pairs[f"source_{other}"] = info["source_id"].to_numpy()
            pairs[f"split_{other}"] = info["split"].to_numpy()
            pairs[f"key_{other}"] = info["image_s3_key"].to_numpy()
        pairs["call"] = np.where(pairs["near_dup"], "near_dup", "ambiguous")

        # Review groups: connected components of all review pairs.
        graph = nx.Graph()
        graph.add_edges_from(pairs[["image_a", "image_b"]].to_numpy())
        img = pd.concat([pairs[[f"{c}_{s}" for c in ("image", "source", "split", "key")]]
                         .set_axis(["image_id", "source_id", "split", "key"], axis=1) for s in "ab"])
        img = img.drop_duplicates("image_id").set_index("image_id")
        comp = {i: c for c, comp in enumerate(nx.connected_components(graph)) for i in comp}
        img["comp"] = img.index.map(comp)
        pairs["comp"] = pairs["image_a"].map(comp)
        g = img.groupby("comp").agg(n_images=("source_id", "size"), n_sources=("source_id", "nunique"),
                                    n_splits=("split", "nunique"),
                                    splits=("split", lambda s: "/".join(sorted(set(s)))),
                                    sources=("source_id", lambda s: sorted(set(s))))
        pg = pairs.groupby("comp").agg(n_pairs=("call", "size"), n_ambiguous=("ambiguous", "sum"),
                                       min_score=("score", "min"), max_score=("score", "max"))
        g = g.join(pg)
        g["category"] = np.select([g["n_splits"] > 1, g["n_sources"] > 1], CATEGORIES[:2], CATEGORIES[2])
        g["has_ambiguous"] = g["n_ambiguous"] > 0
        g["_cat"] = g["category"].map(CATEGORIES.index)
        g = g.sort_values(["_cat", "has_ambiguous", "n_images", "min_score"],
                          ascending=[True, False, False, True]).drop(columns="_cat")
        # Stable-in-order group ids G0001, ... (priority order) and pair ids G0001-1, ...
        g["group"] = [f"G{k + 1:04d}" for k in range(len(g))]
        group_of = g["group"]
        img["group"] = img["comp"].map(group_of)
        pairs["group"] = pairs["comp"].map(group_of)
        pairs = pairs.sort_values(["group", "score"], ascending=[True, False])
        pairs["pair"] = pairs["group"] + "-" + (pairs.groupby("group").cumcount() + 1).astype(str)
        self.groups = g.set_index("group")
        self.images = img.drop(columns="comp")
        self.pairs = pairs.drop(columns="comp").set_index("pair")

    # ------------------------------------------------------------------------------------ overview
    def overview(self) -> pd.DataFrame:
        """Groups, images and pairs per category, and images per split."""
        g = self.groups
        out = g.groupby("category").agg(groups=("n_images", "size"), images=("n_images", "sum"),
                                        pairs=("n_pairs", "sum"), ambiguous_pairs=("n_ambiguous", "sum"))
        out = out.reindex(CATEGORIES).fillna(0).astype(int)
        cat_of_img = self.images["group"].map(g["category"])
        out = out.join(pd.crosstab(cat_of_img, self.images["split"]).reindex(CATEGORIES).fillna(0).astype(int))
        out.loc["total"] = out.sum()
        return out

    def select(self, category: str | list[str] | None = None, split: str | None = None,
               sources: list[int] | None = None, ambiguous: bool | None = None, min_images: int = 2,
               order: str = "priority", n: int | None = None, seed: int = 0) -> pd.DataFrame:
        """Groups (rows of ``self.groups``) matching all given filters.

        ``split``: any image in that split; ``sources``: any image from these sources;
        ``ambiguous``: True = has an ambiguous pair, False = near duplicates only;
        ``order``: ``priority`` (as numbered) or ``random``.
        """
        g = self.groups
        m = g["n_images"] >= min_images
        if category is not None:
            m &= g["category"].isin([category] if isinstance(category, str) else category)
        if split is not None:
            m &= g["splits"].str.split("/").map(lambda s: split in s)
        if sources is not None:
            m &= g["sources"].map(lambda s: bool(set(s) & set(sources)))
        if ambiguous is not None:
            m &= g["has_ambiguous"] == ambiguous
        sel = g[m]
        if order == "random":
            sel = sel.sample(frac=1, random_state=seed)
        return sel if n is None else sel.head(n)

    # ------------------------------------------------------------------------------------ plotting
    @staticmethod
    def _display_jpeg(fig: plt.Figure, dpi: int = 72, quality: int = 75) -> None:
        """Show a figure as a compressed JPEG (keeps notebook output small) and close it."""
        from IPython.display import Image as IPImage
        from IPython.display import display

        display(IPImage(data=_jpeg(fig, dpi, quality), format="jpeg"))

    def _load(self, key: str, max_px: int) -> Image.Image:
        im = Image.open(self.root / key)
        im.draft("RGB", (max_px, max_px))
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        return im

    def _name(self, source_id: int) -> str:
        name = self.source_name.get(source_id)
        return "private/deleted" if pd.isna(name) else str(name)

    def plot_group(self, group: str, max_images: int | None = 12, ncol: int = 4, px: int = 400) -> plt.Figure:
        imgs = self.images[self.images["group"] == group].sort_values(["source_id", "image_id"]).head(max_images)
        ncol = min(ncol, len(imgs))
        nrow = -(-len(imgs) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, layout="constrained")
        for a, (lab, (iid, r)) in zip(axes.flat, zip(_letters(len(imgs)), imgs.iterrows(), strict=True),
                                      strict=False):
            im = self._load(r["key"], px)
            a.imshow(im)
            w, h = Image.open(self.root / r["key"]).size
            a.set_title(f"{lab}: {r['source_id']}/{iid} [{r['split']}]\n{self._name(r['source_id'])[:40]} · {w}x{h}",
                        fontsize=8, loc="left")
            a.axis("off")
        for a in axes.flat[len(imgs):]:
            a.axis("off")
        return fig

    def pair_table(self, group: str, max_rows: int | None = 20) -> pd.DataFrame:
        imgs = self.images[self.images["group"] == group].sort_values(["source_id", "image_id"])
        lab = dict(zip(imgs.index, _letters(len(imgs)), strict=True))
        p = self.pairs[self.pairs["group"] == group]
        t = pd.DataFrame({
            "images": [f"{lab[a]}–{lab[b]}" for a, b in zip(p["image_a"], p["image_b"], strict=True)],
            "call": p["call"], "score": p["score"].round(3), "inliers": p["inliers"],
            "phash_dist": p["phash_dist"], "rotated": p["rot"] != 0,
            "a": p["source_a"].astype(str) + "/" + p["image_a"], "b": p["source_b"].astype(str) + "/" + p["image_b"],
        })
        return t.head(max_rows)

    def show(self, group: str, max_images: int | None = 12, max_rows: int | None = 20, dpi: int = 72) -> None:
        """Header, all images of the group (lettered) and its pairs (ids = ``pair`` in the labels CSV)."""
        from IPython.display import Markdown, display

        g = self.groups.loc[group]
        srcs = ", ".join(f"{s} ({self._name(s)[:30]})" for s in g["sources"][:6])
        more = f", +{len(g['sources']) - 6} more" if len(g["sources"]) > 6 else ""
        display(Markdown(f"---\n### {group} · {g['category']} · {g['n_images']} images · splits: {g['splits']}\n"
                         f"Sources: {srcs}{more}. Pairs: {g['n_pairs']} ({g['n_ambiguous']} ambiguous), "
                         f"score {g['min_score']:.2f}–{g['max_score']:.2f}."))
        self._display_jpeg(self.plot_group(group, max_images), dpi)
        if max_images is not None and g["n_images"] > max_images:
            print(f"… {g['n_images'] - max_images} more images (show(..., max_images=None))")
        t = self.pair_table(group, max_rows)
        display(t)
        if g["n_pairs"] > len(t):
            print(f"… {g['n_pairs'] - len(t)} more pairs (show(..., max_rows=None))")

    def show_many(self, selection: pd.DataFrame, **kwargs) -> None:
        for group in selection.index:
            self.show(group, **kwargs)

    def page(self, k: int, per_page: int = 5, **select_kwargs) -> None:
        """Show page ``k`` (0-based) of ``select(**select_kwargs)``."""
        sel = self.select(**select_kwargs)
        n_pages = -(-len(sel) // per_page)
        print(f"Page {k + 1}/{n_pages} ({len(sel)} groups)")
        self.show_many(sel.iloc[k * per_page:(k + 1) * per_page])

    def align(self, pair: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Colour images a and b (at the verification size), a warped onto b (as in the verification)
        and the absolute high-pass difference (NaN outside the overlap); the last two are None if the
        pair cannot be aligned."""
        import cv2

        p = self.pairs.loc[pair]
        fa, fb = _load_features(self.root / p["key_a"]), _load_features(self.root / p["key_b"])
        size = (fb[0].shape[1], fb[0].shape[0])
        ca = np.asarray(self._load(p["key_a"], VERIFY_PX).resize((fa[0].shape[1], fa[0].shape[0])))
        cb = np.asarray(self._load(p["key_b"], VERIFY_PX).resize(size))
        good = []
        if fa[2] is not None and fb[2] is not None:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(fa[2], fb[2], k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        H = None
        if len(good) >= 12:
            H, _ = cv2.findHomography(fa[1][[m.queryIdx for m in good]], fb[1][[m.trainIdx for m in good]],
                                      cv2.RANSAC, 4.0)
        if H is None:
            return ca, cb, None, None
        diff = np.abs(cv2.warpPerspective(_highpass(fa[0]), H, size) - _highpass(fb[0]))
        valid = cv2.warpPerspective(np.ones_like(fa[0], np.float32), H, size) > 0.5
        return ca, cb, cv2.warpPerspective(ca, H, size), np.where(valid, diff, np.nan)

    def pair_figure(self, pair: str) -> Figure:
        """2 x 2: a and b as uploaded; a aligned onto b, and where their fine detail differs (a copy is
        dark everywhere apart from compression noise, a different shot is bright where content differs).
        Built without pyplot, so it can be rendered in a background thread."""
        p = self.pairs.loc[pair]
        ca, cb, warped, diff = self.align(pair)
        fig = Figure(figsize=(12, 7.2), layout="constrained")
        ax = fig.subplots(2, 2)
        for a, im, side in ((ax[0, 0], ca, "a"), (ax[0, 1], cb, "b")):
            src = p[f"source_{side}"]
            w, h = Image.open(self.root / p[f"key_{side}"]).size
            a.imshow(im)
            a.set_title(f"{side}: {src}/{p[f'image_{side}']} [{p[f'split_{side}']}] {w}x{h}\n"
                        f"{self._name(src)[:60]}", fontsize=9, loc="left")
        if warped is None:
            ax[1, 0].text(0.5, 0.5, "could not align", ha="center", va="center", fontsize=12)
        else:
            ax[1, 0].imshow(warped)
            ax[1, 0].set_title("a aligned onto b", fontsize=9, loc="left")
            ax[1, 1].imshow(diff, cmap="magma", vmin=0, vmax=40)
            ax[1, 1].set_title("|high-pass difference| (dark = identical detail)", fontsize=9, loc="left")
        for a in ax.flat:
            a.axis("off")
        return fig

    def zoom(self, pair: str, dpi: int = 80) -> None:
        """Show ``pair_figure`` for one pair."""
        p = self.pairs.loc[pair]
        fig = self.pair_figure(pair)
        fig.suptitle(f"{pair}: {p['call']}, score {p['score']:.3f}, {p['inliers']} inliers", fontsize=10)
        self._display_jpeg(fig, dpi)


# ---------------------------------------------------------------------------------------------- labels
def label_progress(review: NearDupReview, path: Path | str) -> pd.DataFrame:
    """Labels so far against the algorithm's call (``(open)`` = not labelled yet)."""
    lab = load_labels(path)["label"]
    key = list(zip(review.pairs["image_a"], review.pairs["image_b"], strict=True))
    got = pd.Series([lab.get(k, "(open)") for k in key], index=review.pairs.index)
    return pd.crosstab(review.pairs["call"], got, margins=True)


class PairLabeler:
    """One pair at a time with buttons; ``labeler.widget`` is the UI (display it in a notebook).

    ``pairs``: the pair ids to go through, in order (default: all, by group priority). Starts at the
    first unlabelled pair. ``dup`` / ``fp`` / ``unsure`` label the pair and move on; ``group: rest dup``
    labels every unlabelled pair of the current group ``dup`` (for groups of 5 burst-free copies, i.e.
    10 pairs) and jumps to the next group. The next pairs are rendered in the background.
    """

    def __init__(self, review: NearDupReview, path: Path | str, pairs: list[str] | None = None,
                 dpi: int = 80):
        import ipywidgets as w

        self.r, self.path, self.dpi = review, Path(path), dpi
        self.order = list(review.pairs.index if pairs is None else pairs)
        self.labels = load_labels(self.path)["label"].to_dict()
        pairs_df = review.pairs.loc[self.order]
        self._keys = dict(zip(self.order, zip(pairs_df["image_a"], pairs_df["image_b"], strict=True), strict=True))
        self._group = dict(zip(self.order, pairs_df["group"], strict=True))
        self._cache: dict[str, Future] = {}
        self._pool = ThreadPoolExecutor(max_workers=2)
        self.header = w.HTML()
        self.image = w.Image(format="jpeg", layout=w.Layout(max_width="100%"))
        style = {"dup": "danger", "fp": "success", "unsure": "warning"}
        names = {"dup": "Duplicate", "fp": "False positive", "unsure": "Unsure"}
        buttons = [w.Button(description=names[lab], button_style=style[lab]) for lab in LABELS]
        for b, lab in zip(buttons, LABELS, strict=True):
            b.on_click(lambda _, lab=lab: self.label(lab))
        back, skip = w.Button(description="← Back"), w.Button(description="Skip →")
        rest = w.Button(description="Group: rest dup", tooltip="Label every unlabelled pair of this group dup")
        nxt = w.Button(description="Next unlabelled ⇥")
        back.on_click(lambda _: self.go(self.i - 1))
        skip.on_click(lambda _: self.go(self.i + 1))
        rest.on_click(lambda _: self.label_rest_of_group())
        nxt.on_click(lambda _: self.go(self._next_open(self.i + 1)))
        self.status = w.HTML()
        self.widget = w.VBox([self.header, w.HBox([*buttons, back, skip, nxt, rest]), self.status, self.image])
        self.i = 0
        self.go(self._next_open(0))

    # -------------------------------------------------------------------------------------------- state
    def _key(self, pair: str) -> tuple[str, str]:
        return self._keys[pair]

    def _next_open(self, start: int) -> int:
        for k in range(start, len(self.order)):
            if self._key(self.order[k]) not in self.labels:
                return k
        return min(start, len(self.order) - 1)

    def _render(self, pair: str) -> Future:
        if pair not in self._cache:
            self._cache[pair] = self._pool.submit(lambda: _jpeg(self.r.pair_figure(pair), self.dpi))
        return self._cache[pair]

    def go(self, k: int) -> None:
        self.i = max(0, min(k, len(self.order) - 1))
        pair = self.order[self.i]
        p, g = self.r.pairs.loc[pair], self.r.groups.loc[self.r.pairs.loc[pair, "group"]]
        cur = self.labels.get(self._key(pair), "—")
        n_done = sum(self._key(q) in self.labels for q in self.order)
        counts = pd.Series([self.labels.get(self._key(q)) for q in self.order]).value_counts()
        self.header.value = (
            f"<b>{pair}</b> &nbsp;({self.i + 1}/{len(self.order)}) &nbsp;·&nbsp; labelled {n_done}/{len(self.order)}: "
            + ", ".join(f"{lab} {int(counts.get(lab, 0))}" for lab in LABELS)
            + f"<br>group {p['group']}: {g['category']}, {g['n_images']} images, {g['n_pairs']} pairs, "
            f"splits {g['splits']} &nbsp;·&nbsp; algorithm: <b>{p['call']}</b>, score {p['score']:.3f}, "
            f"{p['inliers']} inliers, pHash {p['phash_dist']}"
            + f"<br>current label: <b>{cur}</b>"
        )
        self.image.value = self._render(pair).result()
        for k2 in (self.i + 1, self.i + 2):  # prefetch
            if k2 < len(self.order):
                self._render(self.order[k2])
        for old in [q for q in self._cache if q not in self.order[max(0, self.i - 2):self.i + 3]]:
            del self._cache[old]

    # ------------------------------------------------------------------------------------------ labels
    def _write(self, pairs: list[str], label: str) -> None:
        rows = self.r.pairs.loc[pairs].reset_index()
        rows = rows.assign(label=label, labelled_at=pd.Timestamp.now().isoformat(timespec="seconds"))
        rows[LABEL_COLUMNS].to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        for q in pairs:
            self.labels[self._key(q)] = label

    def label(self, label: str) -> None:
        pair = self.order[self.i]
        self._write([pair], label)
        self.status.value = f"{pair}: <b>{label}</b> saved"
        self.go(self.i + 1)

    def label_rest_of_group(self) -> None:
        group = self.r.pairs.loc[self.order[self.i], "group"]
        todo = [q for q in self.order if self._group[q] == group and self._key(q) not in self.labels]
        if todo:
            self._write(todo, "dup")
        self.status.value = f"group {group}: {len(todo)} pairs labelled <b>dup</b>"
        later = [k for k in range(self.i, len(self.order)) if self._group[self.order[k]] != group]
        self.go(later[0] if later else self.i)
