#!/usr/bin/env python3
"""Qualitative check of predicted hard coral against the CoralNet point annotations.

Each figure shows one image with three layers:

1. the image itself (the 2048 px resized CoralNet image the annotations refer to);
2. the model's ``order = scleractinia`` probability as shaded bands 0.25–0.5, 0.5–0.75, 0.75–1 (more opaque higher up), with iso-lines at 0.25 (dotted), 0.5 (dashed), 0.75 (solid);
3. the annotated points, coloured by whether the point is hard coral in the ground truth, each labelled
   with its original CoralNet class and, in brackets below it, the MERMAID class it maps to.

Preprocessing, point handling and the ground truth match ``coral_cover_predict.py`` (model input resized
to ``--resolution``, hard coral = ``order__scleractinia``, concept rows from the run's ``class_to_concepts.csv``).
The MERMAID class is ``target_label_class_name`` of that same CSV (first row per CoralNet label, as in
``build_concept_lookup``); ``--mermaid-map json`` uses ``configs/coralnet_to_mermaid_mapping_temporary.json``.

The notebook ``nbs/Coral_Cover_Qualitative_10src.ipynb`` imports these functions for a few images. The CLI
samples ``--n-per-source`` images per source and writes one PNG per image plus two tables:
``images.csv`` (per image cover estimates) and ``points.parquet`` (per annotated point).

Usage:
    python scripts/qualitative_overlays.py --run cbm_scratch100k_base_taxv2 --n-per-source 100 \
        --out eval_out/coral_cover/qualitative/base_taxv2
    # live hard coral: P(scleractinia) x P(live), annotations live hard coral only
    python scripts/qualitative_overlays.py --target live_hard --n-per-source 100 \
        --out eval_out/coral_cover/qualitative/base_taxv2_live
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_REPO_ROOT), str(_REPO_ROOT / "demo")]

from inference import build_model, load_artifacts  # noqa: E402  (demo module)

from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset  # noqa: E402
from mermaidseg.datasets.local_cache import setup_local_cache  # noqa: E402
from mermaidseg.evaluation.gt_mapping import build_concept_lookup  # noqa: E402
from mermaidseg.evaluation.predictor import CBMPredictor  # noqa: E402

HARD_CORAL = "order__scleractinia"
# The 10 test sources of nbs/Coral_Cover_GT_vs_Base_10src*.ipynb.
SOURCES_10 = [6405, 6551, 6702, 4957, 2955, 1990, 3606, 3697, 4342, 7116]

# Ground-truth status of a point (concept code of order__scleractinia: 2 positive, 1 negative, 0 not given).
GT_STYLE = {
    2: ("hard coral", "#ffd23f"),
    1: ("not hard coral", "#2a78d6"),
    0: ("order not given / unmapped", "#b0b0b0"),
}
HEAT = "#ff2bd6"  # magenta: absent from reef imagery (unlike red/orange sponges and algae); opacity carries p
# Probability bands of P(order = scleractinia) as (low, high, opacity); below the first band nothing is drawn.
BANDS = ((0.25, 0.5, 0.1), (0.5, 0.75, 0.3), (0.75, 1.0, 0.45))
# Iso-lines of the same probability as (level, linestyle).
CONTOURS = ((0.25, ":"), (0.5, "--"), (0.75, "-"))
# What is mapped: name used in titles/legends and the probability shown.
#   hard:      P(order = scleractinia); a point/pixel is predicted hard coral when the order argmax is scleractinia;
#              annotation = order__scleractinia code.
#   live_hard: P(order = scleractinia) x P(live) (softmax order head x sigmoid live channel); predicted when the
#              product >= 0.5; annotation positive when scleractinia and live are both positive, negative when either
#              is negative (so dead coral is not live hard coral), otherwise not given.
# Panels that can be drawn: (probability shown, what a positive call means, how the call is made).
PANELS = {"hard": ("P(order = scleractinia)", "hard coral", "order argmax"),
          "live": ("P(live)", "live", "P ≥ 0.5"),
          "live_hard": ("P(scleractinia) × P(live)", "live hard coral", "P ≥ 0.5")}
TARGETS = {"hard": ("hard coral", "P(order = scleractinia)"),
           "live_hard": ("live hard coral", "P(scleractinia) × P(live)")}


@dataclass
class Context:
    base: CoralNetDataset
    predictor: CBMPredictor
    concept_names: list[str]
    order_idx: list[int]
    hard_channel: int
    hard_in_order: int
    concept_lookup: object
    mermaid_of: dict[str, str]
    run: str
    target: str = "hard"  # "hard": order = scleractinia; "live_hard": order = scleractinia and live
    live_channel: int = -1


def _run_paths(run: str, step_dir: str | None) -> tuple[Path, Path, Path]:
    """Checkpoint, concept map and concept CSV of a scratch100k run (as slurm/coral_cover_predict.sbatch)."""
    ckpt = _REPO_ROOT / "nbs/model_checkpoints" / (step_dir or f"{run}_cd/model_epoch0")
    csv = _REPO_ROOT / "nbs/model_checkpoints" / run / "class_to_concepts.csv"
    if csv.is_file():
        id2name = _REPO_ROOT / "eval_out/scratch100k_hier" / run / "concept_id2name.json"
    else:
        csv = _REPO_ROOT / "configs/class_to_concepts.csv"
        id2name = _REPO_ROOT / "eval_out/scratch100k_hier/concept_id2name.json"
    return ckpt, id2name, csv


def mermaid_mapping(kind: str, csv_path: Path) -> dict[str, str]:
    """Lower-cased CoralNet label -> MERMAID class name ('' when unmapped)."""
    if kind == "json":
        raw = json.loads((_REPO_ROOT / "configs/coralnet_to_mermaid_mapping_temporary.json").read_text())
        return {str(k).lower(): (str(v) if v else "") for k, v in raw.items()}
    df = pd.read_csv(csv_path)
    df = df[df["source_dataset_source"].astype(str).str.lower() == "coralnet"]
    df = df.assign(key=df["source_label_class_name"].astype(str).str.lower()).drop_duplicates("key")
    return {k: (str(v) if pd.notna(v) else "") for k, v in zip(df["key"], df["target_label_class_name"], strict=True)}


def load_context(run: str = "cbm_scratch100k_base_taxv2", sources: list[int] | None = None, split: str = "test",
                 resolution: int = 256, step_dir: str | None = None, mermaid_map: str = "csv",
                 data_config: Path = _REPO_ROOT / "configs/data_config_dedup.yaml",
                 device: str = "cuda", target: str = "hard") -> Context:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {list(TARGETS)}; got {target!r}")
    data = yaml.safe_load(Path(data_config).read_text())
    data = data.get("data", data)
    setup_local_cache(data)
    split_cfg = (data.get("coralnet") or {}).get(split) or {}
    base = CoralNetDataset(whitelist_sources=sources or SOURCES_10,
                           image_blacklist=split_cfg.get("image_blacklist"),
                           annotation_overrides=split_cfg.get("annotation_overrides"),
                           transform=None, split=split)

    ckpt, id2name_path, csv = _run_paths(run, step_dir)
    id2name = json.loads(id2name_path.read_text())
    concept_names = [id2name[k] for k in sorted(id2name, key=int)]
    order_idx = [i for i, n in enumerate(concept_names) if n.startswith("order__")]
    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    artifacts = load_artifacts(
        checkpoint=ckpt,
        model_config=_REPO_ROOT / f"configs/model_config_cbm_dpt_lora_vitl_{resolution}.yaml",
        id2label=_REPO_ROOT / "demo/id2label.json",
        concept_id2name=id2name_path,
    )
    dev = torch.device(device)
    predictor = CBMPredictor(build_model(artifacts, dev), input_size=(resolution, resolution), device=dev, amp=True)
    return Context(base=base, predictor=predictor, concept_names=concept_names, order_idx=order_idx,
                   hard_channel=concept_names.index(HARD_CORAL),
                   hard_in_order=[concept_names[i] for i in order_idx].index(HARD_CORAL),
                   concept_lookup=build_concept_lookup("coralnet", names, concept_names, csv),
                   mermaid_of=mermaid_mapping(mermaid_map, csv), run=run, target=target,
                   live_channel=concept_names.index("live"))


def sample_images(ctx: Context, n_per_source: int, seed: int = 0, min_points: int = 1) -> pd.DataFrame:
    """``n_per_source`` random images per source (all of them if fewer), sorted by source then image."""
    counts = ctx.base.df_annotations.groupby("image_id").size().rename("n_points")
    imgs = ctx.base.df_images.join(counts, on="image_id")
    imgs = imgs[imgs.n_points >= min_points]
    out = imgs.sample(frac=1.0, random_state=seed).groupby("source_id").head(n_per_source)
    return out.sort_values(["source_id", "image_id"]).reset_index(drop=True)


@torch.no_grad()
def predict_image(ctx: Context, meta: pd.Series | dict) -> dict:
    """Image, hard coral probability map (model input resolution) and per-point table for one image."""
    meta = dict(meta)
    image = np.asarray(ctx.base.read_image(**meta))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]
    _, concept_probs = ctx.predictor.forward_probs(ctx.predictor.preprocess(image)[None])
    probs = concept_probs[0]  # (C, r, r)
    order = probs[ctx.order_idx]
    live = ctx.target == "live_hard"
    p_scl, p_live = probs[ctx.hard_channel], probs[ctx.live_channel]
    maps = {"hard": p_scl, "live": p_live, "live_hard": p_scl * p_live}
    dense_call = {"hard": order.argmax(dim=0) == ctx.hard_in_order, "live": p_live >= 0.5,
                  "live_hard": maps["live_hard"] >= 0.5}

    ann = ctx.base.df_annotations
    pts = ann[ann.image_id == meta["image_id"]][["row", "col", "coralnet_name", "source_label_name"]]
    pts = pts[(pts.row >= 0) & (pts.row < h) & (pts.col >= 0) & (pts.col < w)].reset_index(drop=True)
    at = ctx.predictor.sample_at_points(probs, pts.row.to_numpy(), pts.col.to_numpy(), h, w)  # (C, n)
    pred_order = at[ctx.order_idx].argmax(dim=0).cpu().numpy()
    order_names = [ctx.concept_names[i].removeprefix("order__") for i in ctx.order_idx]
    rows = [ctx.concept_lookup.row(n) for n in pts.source_label_name]
    hard_code = np.array([int(r[ctx.hard_channel]) for r in rows], dtype=np.int64)
    live_code = np.array([int(r[ctx.live_channel]) for r in rows], dtype=np.int64)
    if live:
        gt_code = np.where((hard_code == 2) & (live_code == 2), 2, np.where((hard_code == 1) | (live_code == 1), 1, 0))
    else:
        gt_code = hard_code
    p_s = at[ctx.hard_channel].cpu().numpy().astype(np.float32)
    p_l = at[ctx.live_channel].cpu().numpy().astype(np.float32)
    pts = pts.assign(
        mermaid=pts.source_label_name.map(lambda n: ctx.mermaid_of.get(n, "")),
        gt_code=gt_code, hard_code=hard_code, live_code=live_code,
        p_scleractinia=p_s, p_live=p_l, p_live_hard=p_s * p_l,
        pred_order=[order_names[i] for i in pred_order],
    )
    # Per-panel call at the points (see PANELS); pred_hard / p_hard are those of the target.
    pts["call_hard"] = pred_order == ctx.hard_in_order
    pts["call_live"] = pts.p_live >= 0.5
    pts["call_live_hard"] = pts.p_live_hard >= 0.5
    pts["p_hard"] = pts.p_live_hard if live else pts.p_scleractinia
    pts["pred_hard"] = pts[f"call_{ctx.target}"]
    n = max(len(pts), 1)
    summary = {
        "source_id": int(meta["source_id"]), "image_id": int(meta["image_id"]), "H": h, "W": w,
        "n_points": len(pts), "gt_cover": float((pts.gt_code == 2).sum() / n),
        "model_points_cover": float(pts.pred_hard.sum() / n),
        "model_dense_cover": float(dense_call[ctx.target].float().mean().item()),
        "mean_p_hard": float(maps[ctx.target].mean().item()),
    }
    for k in PANELS:
        summary[f"points_{k}"] = float(pts[f"call_{k}"].sum() / n)
        summary[f"dense_{k}"] = float(dense_call[k].float().mean().item())
    return {"image": image, "p_hard": maps[ctx.target].cpu().numpy(), "points": pts, "summary": summary,
            "target": ctx.target, "maps": {k: v.cpu().numpy() for k, v in maps.items()}}


def _point_label(coralnet: str, mermaid: str, width: int = 30) -> str:
    top = textwrap.shorten(str(coralnet), width=width, placeholder="…")
    bottom = textwrap.shorten(mermaid, width=width, placeholder="…") if mermaid else "unmapped"
    return f"{top}\n({bottom})"


def _draw_points(ax: plt.Axes, pts: pd.DataFrame, marker_size: float, call: str | None) -> None:
    for code, (_, color) in GT_STYLE.items():
        d = pts[pts.gt_code == code]
        # Ring colour shows the panel's call at the point: magenta = positive, white = not.
        edge = np.where(d[call], HEAT, "white") if call else "white"
        ax.scatter(d.col, d.row, s=marker_size, c=color, edgecolors=edge, linewidths=1.8, zorder=3)


def _shade(ax: plt.Axes, p: np.ndarray, h: int, w: int, bands: tuple, contours: tuple) -> None:
    # Map pixel centres at (i + 0.5) * size / r: the align_corners=False convention used to read values at the points.
    ys, xs = (np.arange(p.shape[0]) + 0.5) * h / p.shape[0], (np.arange(p.shape[1]) + 0.5) * w / p.shape[1]
    for lo, hi, alpha in bands:
        if p.max() >= lo:
            ax.contourf(xs, ys, p, levels=[lo, hi + 1e-6], colors=[HEAT], alpha=alpha, antialiased=True)
    for level, ls in contours:
        if p.max() >= level:
            ax.contour(xs, ys, p, levels=[level], colors=[HEAT], linewidths=1.2, linestyles=ls)


def render(pred: dict, bands: tuple = BANDS, contours: tuple = CONTOURS, label_points: bool = True,
           show_raw: bool = False, panels: tuple[str, ...] | None = None, fontsize: float = 6.0,
           marker_size: float = 60, label_width: int = 30, width_in: float = 14.0,
           title: str | None = None) -> plt.Figure:
    """Draw image + probability shading + labelled annotation points, one panel per map.

    ``panels``: maps to draw, keys of ``PANELS`` (default: the context's target only). Each panel shades its probability
    in ``bands`` ``(low, high, opacity)`` with iso-lines ``contours`` ``(level, linestyle)``; marker rings show that
    panel's call at the point. ``show_raw`` adds a first panel with the bare image. More than two panels go on a
    two-column grid (one row for portrait images). Labels are drawn on the last panel only.
    """
    image, pts, s = pred["image"], pred["points"], pred["summary"]
    target = pred.get("target", "hard")
    what, _ = TARGETS[target]
    panels = tuple(panels or (target,))
    kinds = (("raw",) if show_raw else ()) + panels
    h, w = image.shape[:2]
    # Landscape: two columns; portrait (taller than wide): all panels in one row.
    ncols = 1 if len(kinds) == 1 else (len(kinds) if h > 1.2 * w else 2)
    nrows = -(-len(kinds) // ncols)
    panel_in = width_in / ncols * h / w
    legend_in = 0.9
    fig_h = nrows * (panel_in + 0.55) + 0.6 + legend_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, fig_h), squeeze=False)
    axes = axes.ravel()
    for a in axes[len(kinds):]:
        a.set_visible(False)
    for a, kind in zip(axes, kinds, strict=False):
        a.imshow(image, extent=(0, w, h, 0))
        if kind == "raw":
            _draw_points(a, pts, marker_size * 0.6, call=None)
            a.set_title("image", fontsize=9, loc="left")
        else:
            _shade(a, pred["maps"][kind], h, w, bands, contours)
            _draw_points(a, pts, marker_size, call=f"call_{kind}")
            p_name, pos, rule = PANELS[kind]
            a.set_title(f"{p_name}\n{pos} ({rule}): at points {s[f'points_{kind}']:.0%} · "
                        f"all pixels {s[f'dense_{kind}']:.0%}", fontsize=8.5, loc="left")
        a.set_xlim(0, w)
        a.set_ylim(h, 0)
        a.set_axis_off()
    if label_points:
        ax = axes[len(kinds) - 1]
        for r in pts.itertuples():
            ax.annotate(_point_label(r.coralnet_name, r.mermaid, label_width), (r.col, r.row),
                        xytext=(0, -7), textcoords="offset points", ha="center", va="top",
                        fontsize=fontsize, zorder=4, linespacing=1.0,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7})
    fig.suptitle(title or (
        f"source {s['source_id']} · image {s['image_id']} · {s['n_points']} points   |   {what} cover: "
        f"annotations {s['gt_cover']:.0%} · model at points {s['model_points_cover']:.0%} · "
        f"model all pixels {s['model_dense_cover']:.0%}"), fontsize=10, x=0.01, ha="left")
    gt_names = {2: what, 1: f"not {what}", 0: "not given / unmapped"}
    one = len(panels) == 1
    p_label = PANELS[panels[0]][0] if one else "panel probability"
    ring = f"model predicts {PANELS[panels[0]][1]}" if one else "the panel's call is positive"
    handles = [Line2D([], [], ls="", marker="o", ms=8, mfc=c, mec="white", label=f"annotation: {gt_names[code]}")
               for code, (_, c) in GT_STYLE.items()]
    handles += [Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec=HEAT, mew=1.8,
                       label=f"magenta ring: {ring} at the point"),
                *[Line2D([], [], ls="", marker="s", ms=10, mfc=HEAT, mec="none", alpha=alpha,
                         label=f"shading: {p_label} {lo:g}–{hi:g}") for lo, hi, alpha in bands]]
    handles += [Line2D([], [], color=HEAT, ls=ls, lw=1.2, label=f"line: P = {level:g}") for level, ls in contours]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1 - 0.55 / fig_h), h_pad=3.0)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="cbm_scratch100k_base_taxv2")
    p.add_argument("--step-dir", default=None, help="checkpoint dir under nbs/model_checkpoints "
                   "(default <RUN>_cd/model_epoch0, i.e. step 100k)")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--sources", type=int, nargs="+", default=SOURCES_10)
    p.add_argument("--n-per-source", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--mermaid-map", default="csv", choices=["csv", "json"])
    p.add_argument("--target", default="hard", choices=list(TARGETS),
                   help="hard: order = scleractinia; live_hard: scleractinia x live")
    p.add_argument("--no-contours", action="store_true", help="no probability iso-lines")
    p.add_argument("--show-raw", action="store_true", help="add a bare-image panel on the left")
    p.add_argument("--panels", nargs="+", default=None, choices=list(PANELS),
                   help="maps to draw, one panel each (default: the target), e.g. hard live live_hard")
    p.add_argument("--no-labels", action="store_true", help="draw markers without text labels")
    p.add_argument("--fontsize", type=float, default=6.0)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ctx = load_context(args.run, args.sources, args.split, args.resolution, args.step_dir, args.mermaid_map,
                       target=args.target)
    sample = sample_images(ctx, args.n_per_source, args.seed)
    print(f"{len(sample)} images from {sample.source_id.nunique()} sources -> {args.out}")
    rows, points = [], []
    for i, meta in sample.iterrows():
        try:
            pred = predict_image(ctx, meta)
        except Exception as e:  # noqa: BLE001  (unreadable image: skip, keep going)
            print(f"skip source {meta.source_id} image {meta.image_id}: {e}")
            continue
        s = pred["summary"]
        png = args.out / str(s["source_id"]) / f"{s['image_id']}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig = render(pred, contours=() if args.no_contours else CONTOURS,
                     label_points=not args.no_labels, show_raw=args.show_raw, panels=args.panels,
                     fontsize=args.fontsize)
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        rows.append({**s, "png": str(png.relative_to(args.out))})
        points.append(pred["points"].assign(source_id=s["source_id"], image_id=s["image_id"]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)}")
    pd.DataFrame(rows).to_csv(args.out / "images.csv", index=False)
    pd.concat(points, ignore_index=True).to_parquet(args.out / "points.parquet", index=False)
    print(f"{len(rows)} figures, images.csv, points.parquet -> {args.out}")


if __name__ == "__main__":
    main()
