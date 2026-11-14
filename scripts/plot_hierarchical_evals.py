#!/usr/bin/env python3
"""Plot ``compile_hierarchical_evals.py`` results: each metric across fine-tune steps, one line per run.

Writes one PNG per (eval, metric family) to ``<root>/plots/``, e.g. ``coralnet_consistency.png``.
Each figure is a grid of small panels (one per rank or rank pair, each with its own y-axis),
x = fine-tune step, one line per run in a fixed colour and marker (baseline / distance / KL / partial CE).

Usage:
    python scripts/plot_hierarchical_evals.py --root eval_out/ft50k_hier
    python scripts/plot_hierarchical_evals.py --root eval_out/scratch100k_hier \
        --xlabel "Training step from scratch (75k->100k = LR cooldown)" --reference cbm_scratch100k_base
    python scripts/plot_hierarchical_evals.py --root eval_out/scratch100k_hier --runs '.*_taxv2' \
        --out-dir eval_out/scratch100k_hier/plots_taxv2 --reference cbm_scratch100k_base_taxv2
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus")
PAIRS = tuple(f"{c}->{f}" for c, f in zip(RANKS, RANKS[1:], strict=False))

# Fixed per-entity colour + marker, so a run keeps its look whatever subset is plotted.
RUN_STYLE = {
    "cbm_ft50k_baseline": ("baseline", "#2a78d6", "o"),
    "cbm_ft50k_distance": ("distance loss", "#eb6834", "s"),
    "cbm_ft50k_kl": ("KL loss", "#1baf7a", "^"),
    "cbm_ft50k_partial": ("partial CE", "#eda100", "D"),
}
OTHER_STYLE = ("#8a8984", "v")
# scratch100k runs (<RUN_PREFIX><loss>[_w<weight>][_<tag>]): hue = loss (same hues as above), weight = shade +
# line style (0.01 lighter / dotted, 0.1 base / dashed, 0.5 darker / dash-dot, 1.0 darkest / solid), so weight is
# never colour-only.
RUN_PREFIX = "cbm_scratch100k_"
LOSS_STYLE = {
    "base": ("base", "#2a78d6", "o"),
    "distance": ("distance", "#eb6834", "s"),
    "kl": ("KL", "#1baf7a", "^"),
    "partial": ("partial CE", "#eda100", "D"),
    "klpartial": ("KL + partial CE", "#8a4fd0", "P"),  # weight is "<kl>_<partial>"
}
WEIGHT_STYLE = {"0.01": (0.45, ":"), "0.1": (0.0, "--"), "0.5": (-0.2, "-."), "1.0": (-0.35, "-")}  # (mix: +white / -black, ls)


def _shade(color: str, mix: float) -> str:
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    target = 255 if mix > 0 else 0
    return "#" + "".join(f"{round(c + (target - c) * abs(mix)):02x}" for c in rgb)


_SCRATCH_RUN = re.compile(rf"{RUN_PREFIX}({'|'.join(LOSS_STYLE)})(?:_w([0-9.]+(?:_[0-9.]+)?))?(?:_(\w+))?")


def _parse_scratch(run: str) -> tuple[str, str, str] | None:
    """(loss, weight, tag) of a scratch100k run name, '' for absent parts; None otherwise."""
    m = _SCRATCH_RUN.fullmatch(run)
    return None if m is None else (m[1], m[2] or "", m[3] or "")


def run_style(run: str) -> tuple[str, str, str, str]:
    """(legend label, colour, marker, linestyle) of a run."""
    if run in RUN_STYLE:
        return (*RUN_STYLE[run], "-")
    if (parsed := _parse_scratch(run)) is not None:
        loss, weight, tag = parsed
        name, color, marker = LOSS_STYLE[loss]
        mix, ls = WEIGHT_STYLE.get(weight, (0.0, "-"))
        if tag:  # a rerun (e.g. new concept CSV) sits next to its untagged original: darker, dashed -> solid
            mix, ls = mix - 0.4, "-" if ls == "--" else ls
        label = (f"{name} w={weight.replace('_', '/')}" if weight else name) + (f" [{tag}]" if tag else "")
        return (label, _shade(color, mix), marker, ls)
    return (run, *OTHER_STYLE, "-")


def _run_order(run: str) -> tuple:
    if run in RUN_STYLE:
        return (list(RUN_STYLE).index(run), 0.0, "")
    if (parsed := _parse_scratch(run)) is not None:
        loss, weight, tag = parsed
        return (list(LOSS_STYLE).index(loss), float(weight.split("_")[0] or 0), tag)
    return (99, 0.0, run)
INK, MUTED, GRID = "#1c1c1a", "#6b6a64", "#e4e3df"
# Minimum y-span per panel, so differences in the 4th decimal do not fill a panel.
MIN_Y_SPAN = 0.02
EVAL_TITLES = {"coralnet": "CoralNet", "pacific": "Pacific"}

# family -> (title, [(column suffix, panel title)], higher_is_better)
FAMILIES = {
    "accuracy": (
        "Accuracy (living taxa for ranks)",
        [("class_accuracy", "target class")] + [(f"tax_{r}_acc_living", r) for r in RANKS],
        True,
    ),
    "consistency": (
        "Lineage consistency: argmax(fine) is a child of argmax(coarse)",
        [(f"consistency/{p}", p) for p in PAIRS] + [("consistency/overall", "overall")],
        True,
    ),
    "severity": (
        "Mistake severity: mean tree distance of mistakes / max",
        [(f"severity/{r}/normalized", r) for r in RANKS] + [("severity/mean_normalized", "mean")],
        False,
    ),
    "expected_distance": (
        "Expected tree distance of the prediction (distance-loss objective)",
        [(f"expected_distance/{r}", r) for r in RANKS],
        False,
    ),
    "partial_subtree_mass": (
        "Partial-label subtree mass: finer-head mass inside the labelled taxon's children",
        [(f"partial/{p}/subtree_mass", p) for p in PAIRS],
        True,
    ),
    "kl": (
        "Inter-head KL(coarse || aggregated fine) (KL-loss objective)",
        [(f"kl/{p}", p) for p in PAIRS],
        False,
    ),
    "calibration_rare": (
        "Calibration and rare-taxon accuracy",
        [("ece/mean", "ECE (mean over ranks)")]
        + [(f"rare/{r}/{s}_macro_acc", f"{r} {s}-taxa macro acc") for r in ("family", "genus") for s in ("head", "tail")],
        None,
    ),
}


def _load(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["step"] = int(row["step"])
        for key, value in row.items():
            if key not in ("run", "step", "resolution"):
                row[key] = float(value) if value not in ("", None) else math.nan
    return rows


def _set_limits(ax, values: list[float], steps: list[int]) -> None:
    lo, hi = min(values), max(values)
    if hi - lo < MIN_Y_SPAN:
        mid = (hi + lo) / 2
        lo, hi = mid - MIN_Y_SPAN / 2, mid + MIN_Y_SPAN / 2
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(sorted(set(steps)))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))


def _style_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v / 1000)}k"))


def plot_family(rows: list[dict], eval_name: str, family: str, out_dir: Path, xlabel: str) -> Path | None:
    title, panels, higher_better = FAMILIES[family]
    panels = [(f"{eval_name}/{col}", label) for col, label in panels]
    panels = [(c, l) for c, l in panels if any(not math.isnan(r.get(c, math.nan)) for r in rows)]
    if not panels:
        return None
    runs = sorted({r["run"] for r in rows}, key=_run_order)
    ncols = min(4, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows + 0.9), squeeze=False)
    for ax, (col, label) in zip(axes.flat, panels, strict=False):
        values, steps = [], []
        for run in runs:
            pts = sorted((r["step"], r[col]) for r in rows if r["run"] == run and not math.isnan(r.get(col, math.nan)))
            if not pts:
                continue
            values += [v for _, v in pts]
            steps += [s for s, _ in pts]
            name, color, marker, ls = run_style(run)
            ax.plot(*zip(*pts, strict=True), color=color, marker=marker, linestyle=ls, markersize=5, linewidth=2,
                    markeredgecolor="white", markeredgewidth=0.8, label=name)
        _style_axis(ax, label)
        if values:
            _set_limits(ax, values, steps)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    hint = "" if higher_better is None else ("  (higher is better)" if higher_better else "  (lower is better)")
    fig.suptitle(f"{EVAL_TITLES.get(eval_name, eval_name)} val - {title}{hint}", x=0.01, ha="left", fontsize=12, color=INK)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if len(labels) <= 4:
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
    else:  # its own row under the title
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=len(labels),
                   frameon=False, fontsize=9)
    fig.supxlabel(xlabel, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(labels) <= 4 else 0.9))
    path = out_dir / f"{eval_name}_{family}.png"
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return path


HEADLINE_PANELS = [
    ("coralscapes/miou", "CoralscapesV2 Few-Shot Linear Probe", "mIoU"),
    ("benthos/miou", "Benthos Concept Re-Mapping", "mIoU"),
    ("coralnet/class_accuracy", "CoralNet Class Accuracy", "MERMAID Accuracy"),
    ("pacific/class_accuracy", "PacificLabeledCorals Class Accuracy", "MERMAID Accuracy"),
]


def plot_headline(rows: list[dict], resolution: int, out_dir: Path, xlabel: str) -> Path | None:
    """The four headline metrics over fine-tune checkpoints, one line per run (as in the sweeps)."""
    runs = sorted({r["run"] for r in rows}, key=_run_order)
    if not runs:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    for ax, (col, title, ylabel) in zip(axes.flat, HEADLINE_PANELS, strict=True):
        values, steps = [], []
        for run in runs:
            pts = sorted((r["step"], r[col]) for r in rows
                         if r["run"] == run and not math.isnan(r.get(col, math.nan)))
            if not pts:
                continue
            name, color, marker, ls = run_style(run)
            ax.plot(*zip(*pts, strict=True), color=color, marker=marker, linestyle=ls, markersize=5, linewidth=2,
                    markeredgecolor="white", markeredgewidth=0.8, label=name)
            values += [v for _, v in pts]
            steps += [s for s, _ in pts]
        _style_axis(ax, title)
        ax.set_title("", loc="left")
        ax.set_title(title, loc="center", fontsize=11, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        if values:
            _set_limits(ax, values, steps)
        else:
            ax.text(0.5, 0.5, "not evaluated yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels(), strict=True):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    axes.flat[0].legend(handles, labels, frameon=False, fontsize=9)
    fig.suptitle(f"Eval Resolution: {resolution}", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / "headline.png"
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)
    return path


RANK_PANELS = ["kingdom", "phylum", "order", "binary", "class", "family", "genus", "bleached"]


def plot_rank_overview(rows: list[dict], eval_name: str, out_dir: Path, xlabel: str) -> Path | None:
    """Per-rank accuracy (solid = all points, dashed = living only) plus multi-hot and bleached F1,
    in the layout of the eval-resolution sweep plots, with x = fine-tune step and one colour per run."""
    runs = sorted({r["run"] for r in rows}, key=_run_order)
    fig, axes = plt.subplots(2, 4, figsize=(16, 11.5))
    drawn = False
    for ax, panel in zip(axes.flat, RANK_PANELS, strict=True):
        if panel in RANKS:
            series = [(f"{eval_name}/tax_{panel}_acc_all", "-"), (f"{eval_name}/tax_{panel}_acc_living", "--")]
            title, ylabel = panel.capitalize(), "Accuracy"
        else:
            col = "binary_macro_f1" if panel == "binary" else "bleached_f1"
            series = [(f"{eval_name}/{col}", "-")]
            title = "Multi-Hot Binary Concepts" if panel == "binary" else "Bleached"
            ylabel = "F1"
        values, steps = [], []
        for run in runs:
            name, color, marker, _ = run_style(run)
            for col, ls in series:
                pts = sorted((r["step"], r[col]) for r in rows
                             if r["run"] == run and not math.isnan(r.get(col, math.nan)))
                if pts:
                    ax.plot(*zip(*pts, strict=True), color=color, linestyle=ls, marker=marker,
                            markersize=5, linewidth=2, markeredgecolor="white", markeredgewidth=0.8)
                    values += [v for _, v in pts]
                    steps += [s for s, _ in pts]
        _style_axis(ax, title)
        ax.set_title("", loc="left")
        ax.set_title(title, loc="center", fontsize=11, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        if values:
            drawn = True
            _set_limits(ax, values, steps)
        else:
            ax.text(0.5, 0.5, "not reported", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    if not drawn:
        plt.close(fig)
        return None
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=run_style(r)[1], marker=run_style(r)[2], linewidth=2, label=run_style(r)[0])
               for r in runs]
    handles += [Line2D([], [], color=INK, linestyle="-", label="all"),
                Line2D([], [], color=INK, linestyle="--", label="living only")]
    fig.legend(handles=handles, loc="upper right", ncol=min(len(handles), 9), frameon=False, fontsize=9)
    fig.suptitle(f"{EVAL_TITLES.get(eval_name, eval_name)} val - per-rank accuracy and binary concepts (eval at 256 px)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / f"{eval_name}_rank_overview.png"
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)
    return path


DELTA_PANELS = [
    ("coralnet/class_accuracy", "CoralNet class acc"),
    ("coralnet/tax_family_acc_living", "CoralNet family acc (living)"),
    ("coralnet/tax_genus_acc_living", "CoralNet genus acc (living)"),
    ("coralnet/rare/genus/tail_macro_acc", "CoralNet genus tail macro acc"),
    ("coralnet/consistency/overall", "CoralNet consistency"),
    ("coralnet/severity/mean_normalized", "CoralNet mistake severity (lower better)"),
    ("coralnet/ece/mean", "CoralNet ECE (lower better)"),
    ("pacific/class_accuracy", "Pacific class acc"),
    ("pacific/tax_genus_acc_living", "Pacific genus acc (living)"),
    ("pacific/consistency/overall", "Pacific consistency"),
    ("benthos/miou", "Benthos mIoU"),
    ("coralscapes/miou", "CoralscapesV2 probe mIoU"),
]


def plot_delta(rows: list[dict], reference: str, out_dir: Path, xlabel: str) -> Path | None:
    """Each run minus ``reference`` at the same step (percentage points), one panel per key metric."""
    ref = {r["step"]: r for r in rows if r["run"] == reference}
    runs = sorted({r["run"] for r in rows} - {reference}, key=_run_order)
    if not ref or not runs:
        return None
    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), squeeze=False)
    for ax, (col, label) in zip(axes.flat, DELTA_PANELS, strict=True):
        ax.axhline(0, color=run_style(reference)[1], linewidth=1.2)
        for run in runs:
            pts = sorted((r["step"], 100 * (r[col] - ref[r["step"]][col])) for r in rows
                         if r["run"] == run and r["step"] in ref and not math.isnan(r.get(col, math.nan))
                         and not math.isnan(ref[r["step"]].get(col, math.nan)))
            if not pts:
                continue
            name, color, marker, ls = run_style(run)
            ax.plot(*zip(*pts, strict=True), color=color, marker=marker, linestyle=ls, markersize=5, linewidth=2,
                    markeredgecolor="white", markeredgewidth=0.8, label=name)
        _style_axis(ax, label)
        ax.set_xticks(sorted(ref))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%+.1f"))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=min(len(labels), 5), frameon=False, fontsize=9)
    fig.suptitle(f"Difference to {run_style(reference)[0]} at the same step (percentage points)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.supxlabel(xlabel, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out_dir / "delta_vs_reference.png"
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("eval_out/ft50k_hier"))
    p.add_argument("--csv", type=Path, default=None, help="default: <root>/compiled.csv")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--xlabel", default="Fine-tune step (after the 700k source)")
    p.add_argument("--reference", default=None, help="run to subtract in delta_vs_reference.png")
    p.add_argument("--runs", default=None, help="regex; plot only runs it fully matches")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <root>/plots")
    args = p.parse_args()

    rows = [r for r in _load(args.csv or args.root / "compiled.csv") if int(r["resolution"]) == args.resolution]
    if args.runs:
        rows = [r for r in rows if re.fullmatch(args.runs, r["run"])]
    out_dir = args.out_dir or args.root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        path
        for eval_name in ("coralnet", "pacific")
        for family in FAMILIES
        if (path := plot_family(rows, eval_name, family, out_dir, args.xlabel)) is not None
    ]
    written += [path for e in ("coralnet", "pacific") if (path := plot_rank_overview(rows, e, out_dir, args.xlabel))]
    if (path := plot_headline(rows, args.resolution, out_dir, args.xlabel)) is not None:
        written.insert(0, path)
    if args.reference and (path := plot_delta(rows, args.reference, out_dir, args.xlabel)) is not None:
        written.append(path)
    print(f"{len(written)} figures -> {out_dir}")


if __name__ == "__main__":
    main()
