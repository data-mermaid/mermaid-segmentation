"""Visual review of the CoralNet deduplication: one figure + explanation per duplicated image.

Reads the outputs of ``dedup`` (``point_provenance.parquet`` and
``images_duplicated.parquet``), the source metadata and new splits, and the images from the local
S3 cache (the same resized files training loads, so point coordinates line up).

Every merged location gets a display ``status``, derived from its ``change`` and ``category``:

- ``unchanged``: the kept copy had this point, and every copy that annotated it agrees;
- ``added``: the kept copy had no point here; it comes from another copy (union of annotations);
- ``replaced: finer``: the kept copy's label was replaced by a finer, consistent label;
- ``replaced: compatible``: no label covers the others (e.g. taxon vs condition); the label with
  the most taxonomic ranks won;
- ``replaced: unmapped label``: the kept copy's label is not in ``class_to_concepts.csv``; another
  copy's mapped label is used;
- ``kept over alternative``: the kept copy's label stayed, but another copy had a different
  (coarser, synonymous, compatible or unmapped) label here;
- ``contradictory``: the labels clash; the best-ranked copy's label is used;
- ``unknown``: only labels missing from the concept CSV meet here; the best-ranked copy's is used.

Used by ``mermaidseg/datasets/coralnet/nbs/Dedup_Review.ipynb``.
"""

from __future__ import annotations

import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from PIL import Image

from mermaidseg.datasets.coralnet.dedup.common import (
    CONFIG_DIR,
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUTS,
    load_id2name,
)

# status -> (colour, marker, marker size); Okabe-Ito colours.
STATUS_STYLE = {
    "unchanged": ("#BDBDBD", "o", 14),
    "added": ("#0072B2", "o", 40),
    "replaced: finer": ("#E69F00", "s", 40),
    "replaced: compatible": ("#CC79A7", "s", 40),
    "replaced: unmapped label": ("#009E73", "s", 40),
    "kept over alternative": ("#56B4E9", "D", 30),
    "contradictory": ("#D55E00", "X", 70),
    "unknown": ("#000000", "P", 50),
}
STATUSES = list(STATUS_STYLE)
# Order in which points get numbered / cropped (most interesting first).
_PRIORITY = ["contradictory", "replaced: compatible", "replaced: finer", "replaced: unmapped label", "unknown",
             "added", "kept over alternative", "unchanged"]


def point_status(prov: pd.DataFrame) -> pd.Series:
    status = pd.Series("unchanged", index=prov.index)
    status[prov["category"] != "same_label"] = "kept over alternative"
    status[prov["category"] == "unknown"] = "unknown"
    replaced = prov["change"] == "replaced"
    for cat, name in (("finer", "replaced: finer"), ("compatible", "replaced: compatible"),
                      ("unmapped", "replaced: unmapped label")):
        status[replaced & (prov["category"] == cat)] = name
    status[prov["change"] == "added"] = "added"
    status[prov["category"] == "contradictory"] = "contradictory"
    return status


class DedupReview:
    """Loads everything once; ``select`` picks images, ``show`` renders one."""

    def __init__(self, outputs: Path | str = DEFAULT_OUTPUTS, config_dir: Path | str = CONFIG_DIR,
                 cache_root: str = DEFAULT_CACHE_ROOT):
        outputs = Path(outputs)
        self.root = Path(cache_root) / DEFAULT_BUCKET
        self.id2name = load_id2name(self.root)

        self.prov = pd.read_parquet(outputs / "dedup/point_provenance.parquet")
        self.prov["status"] = point_status(self.prov)
        self.copies = pd.read_parquet(outputs / "dedup/images_duplicated.parquet")
        meta = pd.read_csv(outputs / "coralnet_source_metadata.csv").set_index("source_id")
        splits = pd.read_csv(Path(config_dir) / "source_splits.csv").set_index("source_id")
        self.source_name = meta["name"].to_dict()
        self.source_split = splits["split"].to_dict()

        counts = pd.crosstab(self.prov["hash"], self.prov["status"]).reindex(columns=STATUSES, fill_value=0)
        first = self.prov.drop_duplicates("hash").set_index("hash")
        n_copies = self.copies.groupby("hash").size()
        sources = self.copies.groupby("hash")["source_id"].agg(lambda s: sorted(set(s)))
        self.images = pd.concat([
            first[["kept_source_id", "kept_image_id", "overridden"]],
            n_copies.rename("n_copies"), sources.rename("sources"),
            counts.sum(axis=1).rename("n_points"), counts,
        ], axis=1)
        self.images["split"] = self.images["kept_source_id"].map(self.source_split)
        self.images["n_changed"] = self.images[[s for s in STATUSES if s != "unchanged"]].sum(axis=1)

    # ----------------------------------------------------------------------------------------- selection
    def select(self, status: str | list[str] | None = None, sources: list[int] | None = None,
               image_ids: list[str | int] | None = None, split: str | None = None, min_count: int = 1,
               n: int = 4, order: str = "random", seed: int = 0) -> pd.DataFrame:
        """Pick duplicated images to review.

        status: keep images with at least ``min_count`` points of this status (or of any of these).
        sources: keep images any of whose copies come from one of these sources.
        image_ids: keep images any of whose copies has one of these image ids (overrides the rest).
        split: split of the kept copy's source in the new config (train / val / test).
        order: ``"random"`` (reproducible with ``seed``) or ``"most"`` (most points of ``status``,
            else most changed points, first).
        """
        df = self.images
        if image_ids is not None:
            ids = {str(i) for i in image_ids}
            return df[df.index.isin(self.copies.loc[self.copies["image_id"].isin(ids), "hash"])]
        statuses = [status] if isinstance(status, str) else status
        if statuses:
            df = df[df[statuses].sum(axis=1) >= min_count]
        if sources is not None:
            wanted = set(sources)
            df = df[df["sources"].map(lambda s: bool(wanted & set(s)))]
        if split is not None:
            df = df[df["split"] == split]
        if order == "most":
            key = df[statuses].sum(axis=1) if statuses else df["n_changed"]
            return df.loc[key.sort_values(ascending=False).index[:n]]
        return df.sample(min(n, len(df)), random_state=seed)

    # ----------------------------------------------------------------------------------------- data
    @lru_cache(maxsize=64)  # noqa: B019 - one instance per notebook
    def _image_keys(self, source_id: int) -> dict[str, str]:
        t = pq.read_table(self.root / DEFAULT_MANIFEST, columns=["image_id", "image_s3_key"],
                          filters=[("source_id", "=", source_id)]).to_pandas()
        return dict(zip(t["image_id"].astype(str), t["image_s3_key"], strict=True))

    def load_image(self, source_id: int, image_id: str) -> Image.Image:
        return Image.open(self.root / self._image_keys(int(source_id))[str(image_id)]).convert("RGB")

    def image_points(self, hash_: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(points, copies) of one duplicated image. Points are numbered by interest, then position."""
        pts = self.prov[self.prov["hash"] == hash_].copy()
        pts["_prio"] = pts["status"].map(_PRIORITY.index)
        pts = pts.sort_values(["_prio", "row", "col"]).drop(columns="_prio").reset_index(drop=True)
        pts.insert(0, "nr", np.arange(1, len(pts) + 1))
        copies = self.copies[self.copies["hash"] == hash_].sort_values("copy_rank").reset_index(drop=True)
        return pts, copies

    @staticmethod
    def crop_selection(pts: pd.DataFrame, per_status: int = 5, statuses: list[str] | None = None,
                       seed: int = 0) -> pd.DataFrame:
        """Up to ``per_status`` random points of each status (unchanged originals included), by status."""
        sel = pts if statuses is None else pts[pts["status"].isin(statuses)]
        picked = [g.sample(min(per_status, len(g)), random_state=seed).sort_values("nr")
                  for s in _PRIORITY if not (g := sel[sel["status"] == s]).empty]
        return pd.concat(picked) if picked else sel.iloc[:0]

    def _copy_tag(self, copies: pd.DataFrame, rank: int) -> str:
        c = copies.loc[copies["copy_rank"] == rank].iloc[0]
        return f"{c['source_id']}/{c['image_id']}"

    def grouped_labels(self, ranks: list[int], labels: list[int], final: int, copies: pd.DataFrame,
                       max_sources: int = 4) -> list[str]:
        """One entry per distinct label, best copy first: ``→label ×n (src, src, …)``; → marks the final."""
        src = dict(zip(copies["copy_rank"], copies["source_id"], strict=True))
        groups: dict[int, list[int]] = {}
        for rank, lab in sorted(zip(ranks, labels, strict=True)):
            groups.setdefault(lab, []).append(src[rank])
        out = []
        for lab, srcs in groups.items():
            uniq = list(dict.fromkeys(srcs))
            shown = ", ".join(map(str, uniq[:max_sources])) + (", …" if len(uniq) > max_sources else "")
            count = f" ×{len(srcs)}" if len(srcs) > 1 else ""
            out.append(f"{'→' if lab == final else ' '}{self.id2name.get(lab, '?')}{count} ({shown})")
        return out

    def point_table(self, pts: pd.DataFrame, copies: pd.DataFrame, all_points: bool = False,
                    extra: set[int] | None = None) -> pd.DataFrame:
        """One row per changed point (plus the ``extra`` point numbers, e.g. cropped unchanged ones):
        final label, the kept copy's label and every copy's label."""
        rows = pts if all_points else pts[(pts["status"] != "unchanged") | pts["nr"].isin(extra or set())]
        name = self.id2name.get
        return pd.DataFrame({
            "nr": rows["nr"], "status": rows["status"], "row": rows["row"], "col": rows["col"],
            "final_label": rows["coralnet_id"].map(name),
            "from": [self._copy_tag(copies, r) for r in rows["origin_copy_rank"]],
            "kept_copy_label": rows["kept_label"].map(lambda v: "—" if pd.isna(v) else name(int(v))),
            "labels_by_copy": [" | ".join(self.grouped_labels(rs, ls, f, copies))
                               for rs, ls, f in zip(rows["copy_ranks"], rows["copy_labels"], rows["coralnet_id"],
                                                    strict=True)],
        }).set_index("nr")

    def copies_table(self, copies: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:
        own = pts.explode("copy_ranks").groupby("copy_ranks").size()
        return pd.DataFrame({
            "copy": copies["copy_rank"], "kept": copies["keep"], "source": copies["source_id"],
            "image_id": copies["image_id"],
            "name": copies["source_id"].map(self.source_name).fillna("(private/deleted)"),
            "new_split": copies["source_id"].map(self.source_split),
            "cluster": copies["cluster"].astype("Int64"), "source_rank": copies["source_rank"].astype("Int64"),
            "n_points": copies["n_points"], "points_at_merged_locs": copies["copy_rank"].map(own).fillna(0).astype(int),
        }).set_index("copy")

    def _name(self, source_id: int) -> str:
        name = self.source_name.get(source_id)
        return "private/deleted source" if pd.isna(name) else str(name)

    def explain(self, pts: pd.DataFrame, copies: pd.DataFrame) -> str:
        """Plain-language (markdown) summary of what happened to this image."""
        k = copies.iloc[0]
        lines = [f"**Kept copy: {k['source_id']}/{k['image_id']}** "
                 f"({self._name(k['source_id'])}; "
                 f"new split: {self.source_split.get(k['source_id'], '?')}), "
                 f"one of {len(copies)} byte-identical copies in "
                 f"{copies['source_id'].nunique()} source(s)."]
        if len(copies) > 1:
            o = copies.iloc[1]
            if k["source_id"] == o["source_id"]:
                why = "duplicate within one source: kept the copy with more points / lower image id"
            elif pd.notna(k["source_rank"]) and k["source_rank"] != o["source_rank"]:
                why = (f"best-ranked source in conflict cluster {int(k['cluster'])} "
                       f"(rank {int(k['source_rank'])} vs {o['source_id']}'s rank "
                       f"{'—' if pd.isna(o['source_rank']) else int(o['source_rank'])})")
            else:
                why = f"no ranking difference, so the newer source wins ({k['source_id']} > {o['source_id']})"
            lines.append(f"Why this copy: {why}.")
        c = pts["status"].value_counts()
        n_kept = int(pts["in_kept"].sum())
        lines.append(f"Merged result: {len(pts)} points ({n_kept} from the kept copy, "
                     f"{c.get('added', 0)} added from other copies).")
        desc = {
            "added": "points only another copy had, added as is",
            "replaced: finer": "kept copy's label replaced by a finer, consistent label",
            "replaced: compatible": "labels don't clash but none covers the other; label with most taxonomic ranks used",
            "replaced: unmapped label": "kept copy's label not in class_to_concepts.csv; another copy's label used",
            "kept over alternative": "kept copy's label stayed; another copy had a coarser/synonym/unmapped label",
            "contradictory": "labels clash; best-ranked copy's label used — **check these**",
            "unknown": "only unmapped labels; best-ranked copy's label used",
        }
        for s in _PRIORITY:
            if c.get(s, 0) and s in desc:
                lines.append(f"- **{s}** ({c[s]}): {desc[s]}")
        lines.append("Annotations for this image are "
                     + ("**overridden** in the dataset." if pts["overridden"].iloc[0]
                        else "unchanged in the dataset (the kept copy's own points are used)."))
        return "\n".join(lines)

    # ----------------------------------------------------------------------------------------- plotting
    def plot(self, pts: pd.DataFrame, copies: pd.DataFrame, max_copy_panels: int = 4, display_px: int = 1024,
             label_points: bool = True, highlight: set[int] | None = None) -> plt.Figure:
        """Merged annotations (large, coloured by status) next to each copy's own points.

        Changed points are numbered; so are the unchanged points in ``highlight`` (ringed), e.g. the
        original points shown as crops.
        """
        k = copies.iloc[0]
        img = self.load_image(k["source_id"], k["image_id"])
        scale = display_px / max(img.size)
        small = np.asarray(img.resize((round(img.width * scale), round(img.height * scale))))
        n_panels = min(len(copies), max_copy_panels)
        ncol = 2 if n_panels > 1 else 1
        nrow = max(2, -(-n_panels // ncol))
        fig = plt.figure(figsize=(16, 7.5 if nrow <= 2 else 3.6 * nrow), layout="constrained")
        gs = fig.add_gridspec(nrow, 2 + ncol, width_ratios=[1, 1, *[0.75] * ncol])

        ax = fig.add_subplot(gs[:, :2])
        ax.imshow(small)
        for status in reversed(_PRIORITY):
            s = pts[pts["status"] == status]
            if s.empty:
                continue
            colour, marker, size = STATUS_STYLE[status]
            ax.scatter(s["col"] * scale, s["row"] * scale, c=colour, marker=marker, s=size,
                       edgecolors="white", linewidths=0.8, label=f"{status} ({len(s)})")
        ringed = pts[(pts["status"] == "unchanged") & pts["nr"].isin(highlight or set())]
        ax.scatter(ringed["col"] * scale, ringed["row"] * scale, s=90, facecolors="none", edgecolors="white",
                   linewidths=1.5)
        if label_points:
            for _, p in pd.concat([pts[pts["status"] != "unchanged"], ringed]).iterrows():
                fc = "#616161" if p["status"] == "unchanged" else STATUS_STYLE[p["status"]][0]
                ax.annotate(str(p["nr"]), (p["col"] * scale, p["row"] * scale), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color="white",
                            bbox={"boxstyle": "round,pad=0.1", "fc": fc, "ec": "none"})
        ax.set_title(f"Merged annotations on kept copy {k['source_id']}/{k['image_id']}", fontsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.01), ncol=4, fontsize=8, frameon=False)
        ax.axis("off")

        # Per-copy panels: grey = this copy's label is the final one; blue = adds a point the kept
        # copy lacks; orange = overruled. Shown: the kept copy, then the copies that differ most.
        exploded = pts.explode(["copy_ranks", "copy_labels"])
        exploded["differs"] = (exploded["copy_labels"] != exploded["coralnet_id"]) | ~exploded["in_kept"]
        diff = exploded.groupby("copy_ranks")["differs"].sum()
        others = copies.iloc[1:].assign(diff=copies["copy_rank"].iloc[1:].map(diff).fillna(0),
                                        first_of_source=~copies["source_id"].iloc[1:].duplicated())
        others = others.sort_values(["diff", "first_of_source", "copy_rank"], ascending=[False, False, True])
        shown = pd.concat([copies.iloc[:1], others.head(n_panels - 1)])
        for i, (_, c) in enumerate(shown.iterrows()):
            a = fig.add_subplot(gs[i // ncol, 2 + i % ncol])
            a.imshow(small)
            mine = exploded[exploded["copy_ranks"] == c["copy_rank"]]
            agree = mine["copy_labels"] == mine["coralnet_id"]
            for mask, colour, lab in ((agree & mine["in_kept"], "#BDBDBD", "final label"),
                                      (agree & ~mine["in_kept"], "#0072B2", "adds point"),
                                      (~agree, "#E69F00", "overruled")):
                a.scatter(mine.loc[mask, "col"] * scale, mine.loc[mask, "row"] * scale, c=colour, s=10,
                          edgecolors="white", linewidths=0.4, label=f"{lab} ({int(mask.sum())})")
            a.set_title(f"copy {c['copy_rank']}{' (kept)' if c['keep'] else ''}: "
                        f"{c['source_id']}/{c['image_id']}", fontsize=8)
            a.legend(loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=6, frameon=False, ncol=3,
                     handletextpad=0.1, columnspacing=0.6)
            a.axis("off")
        return fig

    def plot_crops(self, pts: pd.DataFrame, copies: pd.DataFrame, sel: pd.DataFrame | None = None,
                   per_status: int = 5, crop_px: int = 192, ncol: int = 6) -> plt.Figure | None:
        """Full-resolution crops around points, captioned with every copy's label.

        ``sel``: the points to crop; default ``crop_selection(pts, per_status)``.
        """
        if sel is None:
            sel = self.crop_selection(pts, per_status)
        if sel.empty:
            return None
        k = copies.iloc[0]
        img = self.load_image(k["source_id"], k["image_id"])
        nrow = -(-len(sel) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 3.4 * nrow), squeeze=False, layout="constrained")
        half = crop_px // 2
        for a, (_, p) in zip(axes.flat, sel.iterrows(), strict=False):
            r, c = int(p["row"]), int(p["col"])
            a.imshow(img.crop((c - half, r - half, c + half, r + half)))
            colour = "white" if p["status"] == "unchanged" else STATUS_STYLE[p["status"]][0]
            a.plot([half], [half], marker="+", color=colour, markersize=22, mew=2)
            others = pts[(pts["row"] - r).abs().le(half) & (pts["col"] - c).abs().le(half)
                         & ~((pts["row"] == r) & (pts["col"] == c))]
            a.scatter(others["col"] - c + half, others["row"] - r + half, s=12, c="white", edgecolors="black",
                      linewidths=0.5)
            labels = [textwrap.shorten(s, 38, placeholder="…")
                      for s in self.grouped_labels(p["copy_ranks"], p["copy_labels"], p["coralnet_id"], copies, 2)]
            if len(labels) > 4:
                labels = [*labels[:3], f"  … {len(labels) - 3} more labels"]
            a.set_title(f"#{p['nr']} {p['status']}\n" + "\n".join(labels), fontsize=6.5, loc="left",
                        family="monospace")
            a.axis("off")
        for a in axes.flat[len(sel):]:
            a.axis("off")
        return fig

    @staticmethod
    def _display_jpeg(fig: plt.Figure, dpi: int, quality: int = 75) -> None:
        """Show a figure as a compressed JPEG (photos as PNG are several MB each) and close it."""
        import io

        from IPython.display import Image as IPImage
        from IPython.display import display

        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=dpi, pil_kwargs={"quality": quality})
        plt.close(fig)
        display(IPImage(data=buf.getvalue(), format="jpeg"))

    def show(self, hash_: str, crops: bool = True, per_status: int = 5, crop_statuses: list[str] | None = None,
             table: bool = True, max_rows: int | None = 15, max_copies: int = 6, all_points: bool = False,
             seed: int = 0, dpi: int = 72, **plot_kwargs) -> None:
        """Explanation, overview figure, crops and point table for one duplicated image (notebook).

        Crops: up to ``per_status`` random points of each status (of ``crop_statuses``, default all,
        unchanged originals included); the cropped unchanged points are also numbered in the figure
        and listed in the table. Output is kept small: figures are sent as JPEG at ``dpi``, the point
        table shows ``max_rows`` rows (``None`` for all) and the copies table ``max_copies`` copies.
        """
        from IPython.display import Markdown, display

        pts, copies = self.image_points(hash_)
        sel = self.crop_selection(pts, per_status, crop_statuses, seed) if crops else pts.iloc[:0]
        display(Markdown(f"---\n### Image `{hash_[:12]}`\n" + self.explain(pts, copies)))
        ctab = self.copies_table(copies, pts)
        display(ctab.head(max_copies))
        if len(ctab) > max_copies:
            print(f"… {len(ctab) - max_copies} more copies (max_copies=None to show all)")
        self._display_jpeg(self.plot(pts, copies, highlight=set(sel["nr"]), **plot_kwargs), dpi)
        if crops:
            fig = self.plot_crops(pts, copies, sel)
            if fig is not None:
                self._display_jpeg(fig, dpi)
        if table:
            ptab = self.point_table(pts, copies, all_points=all_points, extra=set(sel["nr"]))
            ptab["labels_by_copy"] = ptab["labels_by_copy"].map(lambda s: textwrap.shorten(s, 110, placeholder=" …"))
            # Cropped points first, so the table matches the crops above it.
            ptab = pd.concat([ptab[ptab.index.isin(sel["nr"])], ptab[~ptab.index.isin(sel["nr"])]])
            shown = ptab if max_rows is None else ptab.head(max_rows)
            with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
                display(shown)
            if len(ptab) > len(shown):
                print(f"… {len(ptab) - len(shown)} more points: r.show('{hash_}', max_rows=None) for all")

    def show_many(self, selection: pd.DataFrame, **kwargs) -> None:
        for h in selection.index:
            self.show(h, **kwargs)


def legend_figure() -> plt.Figure:
    """Stand-alone legend of the point statuses."""
    fig, ax = plt.subplots(figsize=(10, 0.6))
    handles = [Line2D([], [], color=c, marker=m, linestyle="", markersize=8, markeredgecolor="white", label=s)
               for s, (c, m, _) in STATUS_STYLE.items()]
    ax.legend(handles=handles, ncol=4, loc="center", frameon=False, fontsize=9)
    ax.axis("off")
    return fig
