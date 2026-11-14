#!/usr/bin/env python3
"""Qualitative check of predicted hard coral genera against the CoralNet point annotations.

Companion of ``qualitative_overlays.py`` (same model loading, sources, preprocessing and ground truth). Each figure:

1. the image (optionally next to a bare copy for reference);
2. regions where the model's genus argmax is a scleractinian genus, one colour per genus (filled + outlined);
   the white dashed line is P(order = scleractinia) = 0.5, the hard coral outline of the other notebook;
3. the annotated points. Marker fill = annotated genus (same colours), ring = the model's genus at the point, so a
   correct genus is a marker whose fill and ring match. Labels: CoralNet class, (MERMAID class), and for coral-relevant
   points the model's genus after an arrow.

Genus definitions follow ``genus_diversity_predict.py``: the model's genus is the argmax of the genus head over all its
channels (a non-coral genus or ``none`` means "no coral genus"); a scleractinian genus is one with
``order = scleractinia`` in the run's ``class_to_concepts.csv``; annotated dead coral (``live`` negative) has no genus.
Colours are assigned per image, most abundant genus first; genera beyond the palette share "other genera".

Usage:
    python scripts/qualitative_genus_overlays.py --run cbm_scratch100k_base_taxv2 --n-per-source 100 --show-raw \
        --out eval_out/coral_cover/qualitative_genus/base_taxv2
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualitative_overlays as qo  # noqa: E402

# Distinct colours for genera (most abundant first); avoids white/greys, which mark non-genus points.
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
           "#bcf60c", "#fabebe", "#008080", "#9a6324"]
OTHER = "#8c8c8c"  # genera beyond the palette
# Annotation categories without a genus: (label, marker, fill).
NO_GENUS = {
    "coral_no_genus": ("hard coral, genus not given", "D", "white"),
    "dead": ("dead coral", "s", "#505050"),
    "not_coral": ("not hard coral / unmapped", "o", "white"),
}


class GenusInfo:
    def __init__(self, ctx: qo.Context):
        names = ctx.concept_names
        self.idx = [i for i, n in enumerate(names) if n.startswith("genus__")]
        self.names = [names[i].removeprefix("genus__") for i in self.idx]
        csv = pd.read_csv(qo._run_paths(ctx.run, None)[2])
        coral = set(csv.loc[(csv["order"] == "scleractinia") & ~csv["genus"].isin(["not_given", "none"]), "genus"])
        self.is_coral = np.array([g in coral for g in self.names])
        self.live_channel = names.index("live")


def _gt_genus(ctx: qo.Context, gi: GenusInfo, label: str) -> tuple[str, str]:
    """(category, genus) of an annotated label: category in {'genus', 'coral_no_genus', 'dead', 'not_coral'}."""
    row = ctx.concept_lookup.row(label)
    if row[ctx.hard_channel] != 2:
        return "not_coral", ""
    if row[gi.live_channel] == 1:
        return "dead", ""
    pos = np.flatnonzero((row[gi.idx] == 2) & gi.is_coral)
    return ("genus", gi.names[pos[0]]) if len(pos) else ("coral_no_genus", "")


def sample_coral_images(ctx: qo.Context, gi: GenusInfo, n_per_source: int, seed: int = 0,
                        min_genus_points: int = 0) -> pd.DataFrame:
    """As ``qo.sample_images``, restricted to images with >= ``min_genus_points`` annotated coral-genus points."""
    ann = ctx.base.df_annotations
    labels = ann["source_label_name"].astype(str).str.lower()
    has_genus = labels.map({n: _gt_genus(ctx, gi, n)[0] == "genus" for n in labels.unique()})
    counts = has_genus.groupby(ann["image_id"]).sum().rename("n_genus_points")
    imgs = ctx.base.df_images.join(counts, on="image_id")
    imgs = imgs[imgs.n_genus_points >= min_genus_points]
    out = imgs.sample(frac=1.0, random_state=seed).groupby("source_id").head(n_per_source)
    return out.sort_values(["source_id", "image_id"]).reset_index(drop=True)


@torch.no_grad()
def predict_genus(ctx: qo.Context, gi: GenusInfo, meta: pd.Series | dict, display_max: int = 1024,
                  min_share: float = 0.005) -> dict:
    """Image, per-pixel genus groups, per-point table and per-genus summary for one image."""
    meta = dict(meta)
    image = np.asarray(ctx.base.read_image(**meta))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]
    _, concept_probs = ctx.predictor.forward_probs(ctx.predictor.preprocess(image)[None])
    probs = concept_probs[0]
    genus = probs[gi.idx]  # (G, r, r)
    coral = torch.as_tensor(gi.is_coral, device=genus.device)
    dense = genus.argmax(dim=0)
    dense_share = torch.bincount(dense.flatten(), minlength=len(gi.idx)).float().cpu().numpy() / dense.numel()

    ann = ctx.base.df_annotations
    pts = ann[ann.image_id == meta["image_id"]][["row", "col", "coralnet_name", "source_label_name"]]
    pts = pts[(pts.row >= 0) & (pts.row < h) & (pts.col >= 0) & (pts.col < w)].reset_index(drop=True)
    at = ctx.predictor.sample_at_points(probs, pts.row.to_numpy(), pts.col.to_numpy(), h, w)
    g_at = at[gi.idx]
    pred = g_at.argmax(dim=0).cpu().numpy()
    gt = [_gt_genus(ctx, gi, n) for n in pts.source_label_name]
    pts = pts.assign(
        mermaid=pts.source_label_name.map(lambda n: ctx.mermaid_of.get(n, "")),
        gt_cat=[c for c, _ in gt], gt_genus=[g for _, g in gt],
        pred_genus=[gi.names[i] if gi.is_coral[i] else "" for i in pred],
        pred_genus_raw=[gi.names[i] for i in pred],
        p_pred=g_at.max(dim=0).values.cpu().numpy().astype(np.float32),
        pred_hard=(at[ctx.order_idx].argmax(dim=0) == ctx.hard_in_order).cpu().numpy(),
    )

    # Genera to colour: annotated + predicted at points + covering >= min_share of pixels; most abundant first.
    n = max(len(pts), 1)
    score: dict[str, float] = {}
    for g in pts.gt_genus[pts.gt_cat == "genus"]:
        score[g] = score.get(g, 0) + 1 / n
    for g in pts.pred_genus[pts.pred_genus != ""]:
        score[g] = score.get(g, 0) + 1 / n
    for i in np.flatnonzero(gi.is_coral & (dense_share >= min_share)):
        score[gi.names[i]] = score.get(gi.names[i], 0) + dense_share[i]
    shown = sorted(score, key=lambda g: -score[g])
    colour = {g: (PALETTE[k] if k < len(PALETTE) else OTHER) for k, g in enumerate(shown)}

    # Display map: max probability per colour group (palette genera, other coral genera, non-coral), upsampled to
    # display size and argmaxed; for the same pixel this is the genus argmax mapped to its group.
    in_palette = [gi.names.index(g) for g in shown[:len(PALETTE)]]
    rest_coral = coral.clone()
    rest_coral[in_palette] = False
    groups = [genus[i] for i in in_palette]
    groups.append(genus[rest_coral].max(dim=0).values if rest_coral.any() else torch.zeros_like(genus[0]))
    groups.append(genus[~coral].max(dim=0).values)
    scale = min(1.0, display_max / max(h, w))
    dh, dw = max(1, round(h * scale)), max(1, round(w * scale))
    up = F.interpolate(torch.stack(groups)[None], size=(dh, dw), mode="bilinear", align_corners=False)[0]
    label_map = up.argmax(dim=0).cpu().numpy()  # 0..K-1 palette genera, K other coral, K+1 non-coral

    def share(g: str) -> float:
        return float(dense_share[gi.names.index(g)])

    genera = pd.DataFrame({
        "genus": shown, "colour": [colour[g] for g in shown],
        "gt_points": [int(((pts.gt_cat == "genus") & (pts.gt_genus == g)).sum()) for g in shown],
        "pred_points": [int((pts.pred_genus == g).sum()) for g in shown],
        "correct_points": [int(((pts.gt_genus == g) & (pts.pred_genus == g)).sum()) for g in shown],
        "dense_share": [share(g) for g in shown],
    })
    gt_g = set(pts.gt_genus[pts.gt_cat == "genus"])
    pred_g = set(pts.pred_genus[pts.pred_genus != ""])
    dense_g = {gi.names[i] for i in np.flatnonzero(gi.is_coral & (dense_share >= min_share))}
    gpts = pts[pts.gt_cat == "genus"]
    summary = {
        "source_id": int(meta["source_id"]), "image_id": int(meta["image_id"]), "H": h, "W": w,
        "n_points": len(pts), "n_genus_points": len(gpts),
        "genus_acc_points": float((gpts.gt_genus == gpts.pred_genus).mean()) if len(gpts) else float("nan"),
        "n_genera_gt": len(gt_g), "n_genera_pred_points": len(pred_g), "n_genera_dense": len(dense_g),
        "genera_gt": ",".join(sorted(gt_g)), "genera_pred_points": ",".join(sorted(pred_g)),
        "genera_dense": ",".join(sorted(dense_g)),
    }
    return {"image": image, "label_map": label_map, "n_palette": len(in_palette), "palette_genera": shown,
            "colour": colour, "p_hard": probs[ctx.hard_channel].cpu().numpy(), "points": pts,
            "genera": genera, "summary": summary}


def _label(r, width: int) -> str:
    top = textwrap.shorten(str(r.coralnet_name), width=width, placeholder="…")
    mer = textwrap.shorten(r.mermaid, width=width, placeholder="…") if r.mermaid else "unmapped"
    text = f"{top}\n({mer})"
    if r.gt_cat != "not_coral" or r.pred_genus:
        text += f"\n→ {r.pred_genus or 'no coral genus'}"
    return text


def render_genus(pred: dict, fill_alpha: float = 0.35, outline: bool = True, hard_contour: bool = True,
                 label_points: str = "all", show_raw: bool = False, fontsize: float = 6.0, marker_size: float = 90,
                 label_width: int = 30, width_in: float = 14.0, title: str | None = None) -> plt.Figure:
    """Draw image + predicted genus regions + annotated points (fill = annotated genus, ring = model genus).

    ``label_points``: "all", "coral" (only points annotated as or predicted to be hard coral genera) or "none".
    """
    image, lab, pts, s = pred["image"], pred["label_map"], pred["points"], pred["summary"]
    colour, k = pred["colour"], pred["n_palette"]
    h, w = image.shape[:2]
    ncols = 2 if show_raw else 1
    n_genus_rows = (min(len(pred["genera"]), len(PALETTE) + 1) + 1) // 2
    legend_in = 0.17 * max(n_genus_rows + 1, len(NO_GENUS) + 2) + 0.15  # legend block height (inches)
    img_in = width_in / ncols * h / w
    fig_h = img_in + 0.45 + legend_in
    fig, axes = plt.subplots(1, ncols, figsize=(width_in, fig_h), squeeze=False)
    axes = axes[0]
    ax = axes[-1]
    for a in axes:
        a.imshow(image, extent=(0, w, h, 0))

    group_colours = [colour[g] for g in pred["palette_genera"][:k]] + [OTHER]  # groups 0..k (k = other coral)
    rgba = np.zeros((*lab.shape, 4))
    for gidx, c in enumerate(group_colours):
        rgba[lab == gidx] = (*to_rgba(c)[:3], fill_alpha)
    ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest")
    if outline:
        dh, dw = lab.shape
        ys, xs = (np.arange(dh) + 0.5) * h / dh, (np.arange(dw) + 0.5) * w / dw
        for gidx, c in enumerate(group_colours):
            m = (lab == gidx).astype(float)
            if 0 < m.sum() < m.size:
                ax.contour(xs, ys, m, levels=[0.5], colors=[c], linewidths=1.1)
    if hard_contour and pred["p_hard"].max() >= 0.5:
        r = pred["p_hard"].shape[0]
        ys, xs = (np.arange(r) + 0.5) * h / r, (np.arange(r) + 0.5) * w / r
        ax.contour(xs, ys, pred["p_hard"], levels=[0.5], colors=["white"], linewidths=1.0, linestyles="--")

    def ring(g: str) -> str:
        return colour.get(g, OTHER) if g else "black"

    for target, size in ((ax, marker_size), *(((axes[0], marker_size * 0.6),) if show_raw else ())):
        d = pts[pts.gt_cat == "genus"]
        target.scatter(d.col, d.row, s=size, c=[colour.get(g, OTHER) for g in d.gt_genus], marker="o",
                       edgecolors=[ring(g) for g in d.pred_genus] if target is ax else "white",
                       linewidths=2.8 if target is ax else 1.0, zorder=3)
        for cat, (_, marker, fill) in NO_GENUS.items():
            d = pts[pts.gt_cat == cat]
            sz = size * (0.5 if cat == "not_coral" else 0.8)
            target.scatter(d.col, d.row, s=sz, c=fill, marker=marker,
                           edgecolors=[ring(g) for g in d.pred_genus] if target is ax else "black",
                           linewidths=2.4 if target is ax else 0.8, zorder=3)
    if label_points != "none":
        keep = pts if label_points == "all" else pts[(pts.gt_cat != "not_coral") | (pts.pred_genus != "")]
        for r in keep.itertuples():
            ax.annotate(_label(r, label_width), (r.col, r.row), xytext=(0, -7), textcoords="offset points",
                        ha="center", va="top", fontsize=fontsize, zorder=4, linespacing=1.0,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7})
    for a in axes:
        a.set_xlim(0, w)
        a.set_ylim(h, 0)
        a.set_axis_off()
    acc = s["genus_acc_points"]
    fig.suptitle(title or (
        f"source {s['source_id']} · image {s['image_id']} · {s['n_points']} points ({s['n_genus_points']} with a coral "
        f"genus)   |   genus accuracy at those points {'–' if np.isnan(acc) else f'{acc:.0%}'}   |   genera: "
        f"annotations {s['n_genera_gt']} · model at points {s['n_genera_pred_points']} · model all pixels "
        f"{s['n_genera_dense']}"), fontsize=10, x=0.01, ha="left")

    g = pred["genera"]
    genus_handles = [Patch(fc=to_rgba(r.colour, 0.6), ec=r.colour,
                           label=f"{r.genus}: annotated {r.gt_points}, model {r.pred_points} pts "
                                 f"({r.correct_points} right), {r.dense_share:.1%} px")
                     for r in g.itertuples() if r.colour != OTHER]
    if (g.colour == OTHER).any():
        genus_handles.append(Patch(fc=to_rgba(OTHER, 0.6), ec=OTHER, label=f"{(g.colour == OTHER).sum()} other genera"))
    key = [Line2D([], [], ls="", marker="o", ms=8, mfc="#bbbbbb", mec="black", mew=2,
                  label="fill: annotated genus · ring: model genus (black = no coral genus)")]
    key += [Line2D([], [], ls="", marker=m, ms=7, mfc=f, mec="black", label=f"annotation: {lab_}")
            for lab_, m, f in NO_GENUS.values()]
    key.append(Line2D([], [], color="white", ls="--", lw=1.2, label="P(order = scleractinia) = 0.5"))
    leg = fig.legend(handles=genus_handles, loc="lower left", bbox_to_anchor=(0.01, 0), ncol=2, fontsize=7.5,
                     frameon=False, title="model genus regions / annotated genera", title_fontsize=8)
    leg._legend_box.align = "left"
    fig.legend(handles=key, loc="lower right", bbox_to_anchor=(0.99, 0), ncol=1, fontsize=7.5, frameon=True,
               facecolor="#9a9a9a", framealpha=0.35, edgecolor="none")
    fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1 - 0.4 / fig_h), pad=0.4)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="cbm_scratch100k_base_taxv2")
    p.add_argument("--step-dir", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", default=qo.SOURCES_10)
    p.add_argument("--n-per-source", type=int, default=100)
    p.add_argument("--min-genus-points", type=int, default=0,
                   help="only sample images with at least this many annotated coral-genus points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--mermaid-map", default="csv", choices=["csv", "json"])
    p.add_argument("--fill-alpha", type=float, default=0.35)
    p.add_argument("--labels", default="all", choices=["all", "coral", "none"])
    p.add_argument("--show-raw", action="store_true")
    p.add_argument("--fontsize", type=float, default=6.0)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ctx = qo.load_context(args.run, args.sources, args.split, args.resolution, args.step_dir, args.mermaid_map)
    gi = GenusInfo(ctx)
    sample = sample_coral_images(ctx, gi, args.n_per_source, args.seed, args.min_genus_points)
    print(f"{len(sample)} images from {sample.source_id.nunique()} sources -> {args.out}")
    rows, points, genera = [], [], []
    for i, meta in sample.iterrows():
        try:
            pred = predict_genus(ctx, gi, meta)
        except Exception as e:  # noqa: BLE001  (unreadable image: skip, keep going)
            print(f"skip source {meta.source_id} image {meta.image_id}: {e}")
            continue
        s = pred["summary"]
        png = args.out / str(s["source_id"]) / f"{s['image_id']}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig = render_genus(pred, fill_alpha=args.fill_alpha, label_points=args.labels, show_raw=args.show_raw,
                           fontsize=args.fontsize)
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        rows.append({**s, "png": str(png.relative_to(args.out))})
        ids = {"source_id": s["source_id"], "image_id": s["image_id"]}
        points.append(pred["points"].assign(**ids))
        genera.append(pred["genera"].assign(**ids))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)}")
    pd.DataFrame(rows).to_csv(args.out / "images.csv", index=False)
    pd.concat(points, ignore_index=True).to_parquet(args.out / "points.parquet", index=False)
    pd.concat(genera, ignore_index=True).to_parquet(args.out / "genera.parquet", index=False)
    print(f"{len(rows)} figures, images.csv, points.parquet, genera.parquet -> {args.out}")


if __name__ == "__main__":
    main()
