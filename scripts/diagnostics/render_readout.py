#!/usr/bin/env python
"""Render a post-hoc readout (from posthoc_readout.py) as a reviewer-facing figure.

Reads ``<out>/readout.json`` + the source image and draws the quadrat with its point grid plus,
for the most informative points, the top-K classes and the hierarchy ladder. No model reload.

Example:
    python scripts/diagnostics/render_readout.py out_readout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

_CORAL = (
    "coral",
    "acropora",
    "porites",
    "pavona",
    "isopora",
    "galaxea",
    "montipora",
    "pocillopora",
    "favia",
    "favites",
    "goniastrea",
)
_ALGAE = ("algae", "turf", "macro", "halimeda", "dictyota", "lobophora")
_SUBSTRATE = ("sand", "rubble", "rock", "bare", "substrate")


def cat_color(label: str) -> str:
    lab = label.lower()
    if any(k in lab for k in _CORAL):
        return "#D9633B" if "hard coral" in lab else "#E8865A"
    if any(k in lab for k in _ALGAE):
        return "#5BB98C"
    if any(k in lab for k in _SUBSTRATE):
        return "#C9B27E"
    return "#8A99A6"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "out_dir", type=Path, help="dir containing readout.json (from posthoc_readout.py)"
    )
    ap.add_argument("--n-cards", type=int, default=4, help="how many points to detail on the right")
    args = ap.parse_args()

    with open(args.out_dir / "readout.json") as f:
        rep = json.load(f)
    img = np.asarray(Image.open(rep["image"]).convert("RGB").resize((512, 512)))
    pts = rep["points"]

    def coarse_gap(p):
        leaf = p["top_k"][0]["conf"]
        coarse = max((d["conf"] for d in p["hierarchy_ladder"]), default=leaf)
        return coarse - leaf, len(p["hierarchy_ladder"])

    scored = sorted(range(len(pts)), key=lambda i: -coarse_gap(pts[i])[0])
    picks = [i for i in scored if coarse_gap(pts[i])[1] > 1][: args.n_cards]
    picks = (picks + [i for i in scored if i not in picks])[: args.n_cards]

    fig = plt.figure(figsize=(13, 6.6), dpi=140)
    fig.patch.set_facecolor("#0E1817")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.06)

    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for i, p in enumerate(pts):
        x, y = p["px"]
        c = cat_color(p["top_k"][0]["label"])
        star = i in picks
        ax.scatter(
            [x],
            [y],
            s=230 if star else 90,
            marker="*" if star else "o",
            c=c,
            edgecolors="white",
            linewidths=1.6 if star else 1.0,
            zorder=3,
        )
        if star:
            ax.annotate(
                str(picks.index(i) + 1),
                (x, y),
                color="white",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )
    for s in ax.spines.values():
        s.set_color("#31423E")
    ax.set_title(
        f"{Path(rep['image']).name}  ·  {len(pts)}-point readout",
        color="#E8ECEA",
        fontsize=11,
        pad=8,
        loc="left",
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(
        0,
        0.985,
        "What the flat model reports per point",
        color="#E68A5F",
        fontsize=12,
        fontweight="bold",
        va="top",
        family="serif",
    )
    ax2.text(
        0,
        0.945,
        "old UI keeps only the top line  ·  post-hoc keeps the whole ladder",
        color="#94A6A1",
        fontsize=8.5,
        va="top",
    )
    y = 0.90
    for n, idx in enumerate(picks):
        p = pts[idx]
        ax2.add_patch(
            FancyBboxPatch(
                (0, y - 0.205),
                1.0,
                0.19,
                boxstyle="round,pad=0.006",
                fc="#141F1D",
                ec="#243330",
                lw=1,
            )
        )
        ax2.text(
            0.02, y - 0.02, str(n + 1), color="#E8ECEA", fontsize=11, fontweight="bold", va="top"
        )
        ax2.text(
            0.075, y - 0.018, "top-3", color="#6E807B", fontsize=7.5, va="top", family="monospace"
        )
        ax2.text(
            0.075,
            y - 0.05,
            "   ".join(f"{d['label']} {d['conf'] * 100:.0f}%" for d in p["top_k"]),
            color="#E8ECEA",
            fontsize=9.2,
            va="top",
            fontweight="bold",
        )
        ax2.text(
            0.075,
            y - 0.093,
            "hierarchy ladder (leaf → coarse)",
            color="#6E807B",
            fontsize=7.5,
            va="top",
            family="monospace",
        )
        lx = 0.075
        for d in p["hierarchy_ladder"][:4]:
            conf = d["conf"]
            w = 0.20 * conf + 0.045
            col = "#71B7AD" if conf > 0.8 else ("#D6A64A" if conf > 0.5 else "#94A6A1")
            ax2.add_patch(
                FancyBboxPatch(
                    (lx, y - 0.185),
                    w,
                    0.055,
                    boxstyle="round,pad=0.003",
                    fc=col,
                    ec="none",
                    alpha=0.9,
                )
            )
            ax2.text(
                lx + 0.008,
                y - 0.158,
                f"{d['level']}\n{conf * 100:.0f}%",
                color="#0E1817",
                fontsize=7.0,
                va="center",
                fontweight="bold",
                linespacing=0.95,
            )
            lx += w + 0.015
        y -= 0.225

    fig.suptitle(
        "Post-hoc per-level readout · flat DINOv3 checkpoint (no retrain)",
        color="#E8ECEA",
        fontsize=13,
        fontweight="bold",
        x=0.5,
        y=0.995,
    )
    out = args.out_dir / "readout_figure.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
