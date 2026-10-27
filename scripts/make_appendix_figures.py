#!/usr/bin/env python3
"""Render the dataset appendix figures from the bundle written by ``export_appendix_data.py``.

Runs locally (no cluster or S3 access). Reads ``<data-dir>/`` and writes
``<output-dir>/``:

    pie_<metric>_<split>.pdf          12 pies: {images, annotations, pixels} x {total, train, val, test}
    legend_datasets.pdf               dataset legend shared by the pies and bar charts
    labels_<metric>.pdf               mapped source labels sorted by count, colored by dataset
    concepts_balance.pdf              True / False / not given shares for health, morphology,
                                      and auxiliary concepts: all mapped data, and Cnidaria only
    <dataset>/NN_image.jpg            dense: image + NN_label.png (colorized source labels)
    <dataset>/NN_overlay.jpg          sparse: points drawn over the image
    <dataset>/legend_<dataset>.pdf    labels present in the exported samples
    coralnet/s<id>/NN_overlay.jpg     one folder per CoralNet source
    coralnet/legend_s<id>.pdf         one legend per source; shared labels share a color
    coralnet/sources.txt

Pies use the ``level == "source"`` rows of ``dataset_table.csv`` (the pre-mapping
appendix table). Bar charts use the ``mapped`` rows of ``label_counts.csv`` summed
over splits. Sample images are written with PIL at native aspect ratio; matplotlib
is only used for the PDFs, set in Computer Modern without needing a LaTeX install.

Usage:
    python scripts/make_appendix_figures.py
    python scripts/make_appendix_figures.py --data-dir appendix_data --only pies bars
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dataset_table_stats import (  # noqa: E402
    DATASET_SPECS,
    DEFAULT_CONCEPTS_CSV,
    IGNORE_LABEL_NAMES,
    PAPER_SPLITS,
)

from mermaidseg.dataset_reconciliation.concepts import (  # noqa: E402
    HEALTH_CONCEPTS,
    MORPHOLOGIC_CONCEPTS,
    NONCORAL_CONCEPTS,
)

logger = logging.getLogger("make_appendix_figures")

_REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_DATA_DIR = _REPO_ROOT / "appendix_data"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "figures" / "appendix"
CORALSCAPES_V2_COLORS = _REPO_ROOT / "configs" / "coralscapes_v2_label2color.json"

DATASET_DISPLAY: dict[str, str] = {spec.key: spec.display for spec in DATASET_SPECS}
DENSE_KEYS: frozenset[str] = frozenset(s.key for s in DATASET_SPECS if s.kind == "dense")

# Paul Tol "muted" (colorblind safe), one color per dataset in table order.
DATASET_COLORS: dict[str, str] = {
    "moorea_labeled_corals": "#88CCEE",
    "pacific_labeled_corals": "#999933",
    "catlin_seaview": "#117733",
    "benthos_yuval": "#882255",
    "ucsd_mosaics": "#CC6677",
    "coralscapes_v2": "#44AA99",
    "mermaid": "#DDCC77",
    "coralnet": "#332288",
}

# Paul Tol "bright" without grey; the first colors handed out to source labels.
LABEL_SEED_COLORS: tuple[str, ...] = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
)
UNLABELED_COLOR = (255, 255, 255)
UNLABELED_EDGE = (0.45, 0.45, 0.45)
POINT_EDGE = (25, 25, 25)

METRICS: dict[str, tuple[str, str]] = {
    # metric -> (pie/bar noun, bar y-label)
    "images": ("images", "Images"),
    "annotations": ("annotations", "Annotations"),
    "pixels": ("annotated pixels", "Annotated pixels"),
}
PIE_SPLITS: tuple[str, ...] = ("total", *PAPER_SPLITS)

DEFAULT_POINT_SCALE = 1.0 / 120.0

TEXT_WIDTH_IN = 6.875
PIE_SIZE_IN = 1.3
# One third of \textwidth so the three charts sit side by side under the pies.
BAR_SIZE_IN = (TEXT_WIDTH_IN / 3, 1.6)
# Full \textwidth. Two rows: all mapped data, then the Cnidaria subset.
CONCEPT_BALANCE_SIZE_IN = (TEXT_WIDTH_IN, 5.45)
CNIDARIAN_PHYLUM = "cnidaria"

# Health, then morphology, then auxiliary. Names stay in the canonical list order.
CONCEPT_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Health", tuple(HEALTH_CONCEPTS)),
    ("Morphology", tuple(MORPHOLOGIC_CONCEPTS)),
    ("Auxiliary", tuple(NONCORAL_CONCEPTS)),
)
CONCEPT_NAMES: tuple[str, ...] = tuple(name for _, names in CONCEPT_SECTIONS for name in names)
# Stacked bottom to top. Not given stays gray; false and true use the tableau colors.
CONCEPT_STATES: tuple[tuple[str, str, str], ...] = (
    ("not_given", "Not given", "#BBBBBB"),
    ("false", "False", "tab:red"),
    ("true", "True", "tab:blue"),
)
# CoralNet legends sit beside their source's images, in the right third of \textwidth.
CORALNET_LEGEND_WIDTH_IN = 0.32 * TEXT_WIDTH_IN
CORALNET_LEGEND_FONTSIZE = 6.0
CORALNET_LEGEND_WRAP = 22
# Longer labels are wrapped so a single label cannot force a one-column legend.
LEGEND_WRAP = 30

_USETEX = False

# cmr10 is TeX-encoded: these ASCII codepoints hold other glyphs (e.g. "<" is "¡").
_CMR10_SUBSTITUTIONS: dict[str, str] = {
    "$": r"\$",
    "_": " ",
    "<": r"$<$",
    ">": r"$>$",
    "|": r"$|$",
    "{": r"$\{$",
    "}": r"$\}$",
    "\\": r"$\backslash$",
    '"': "''",
    "~": r"$\sim$",
    "^": r"$\wedge$",
}
_TEX_SUBSTITUTIONS: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def setup_style(usetex: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global _USETEX
    _USETEX = usetex
    plt.rcParams.update(
        {
            "font.family": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "ytick.minor.size": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )
    if usetex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )


def fmt_text(text: str) -> str:
    """Make arbitrary label text render correctly in the active font setup."""
    table = _TEX_SUBSTITUTIONS if _USETEX else _CMR10_SUBSTITUTIONS
    return "".join(table.get(ch, ch) for ch in str(text))


def pct_text(value: float) -> str:
    return f"{value:.0f}\\%" if _USETEX else f"{value:.0f}%"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb01(color: Sequence[int]) -> tuple[float, float, float]:
    return tuple(float(c) / 255.0 for c in color[:3])  # type: ignore[return-value]


def _relative_luminance(color: Sequence[float]) -> float:
    rgb = np.asarray(color[:3], dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(lin @ np.array([0.2126, 0.7152, 0.0722]))


# ---------------------------------------------------------------------------
# Label palettes
# ---------------------------------------------------------------------------


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65) for sRGB values in ``[0, 1]`` with shape ``(n, 3)``."""
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)
    return np.stack(
        [116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1
    )


def _candidate_colors() -> tuple[np.ndarray, np.ndarray]:
    """sRGB grid (and its CIELAB values) without near-black and near-white colors."""
    steps = np.arange(0, 256, 15)
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    lab = srgb_to_lab(grid / 255.0)
    keep = (lab[:, 0] >= 30) & (lab[:, 0] <= 92)
    return grid[keep], lab[keep]


def assign_colors(
    labels_by_frequency: Iterable[str],
    groups: Iterable[Iterable[str]] | None = None,
    *,
    min_seed_distance: float = 30.0,
) -> dict[str, tuple[int, int, int]]:
    """One color per label, distinct (CIELAB) from every label it shares a group with.

    ``groups`` are label sets shown together under one legend; by default all labels
    form one group. Labels are colored in the given order (most frequent first).
    Each label takes the first Paul Tol "bright" color at least ``min_seed_distance``
    from its conflicts, otherwise the grid color farthest from them. Black and white
    are always conflicts, so no label looks like the unlabeled mask or a highlight.
    """
    labels = list(dict.fromkeys(labels_by_frequency))
    group_sets = [set(labels)] if groups is None else [set(g) for g in groups]
    neighbors: dict[str, set[str]] = {label: set() for label in labels}
    for group in group_sets:
        for label in group & neighbors.keys():
            neighbors[label] |= group - {label}

    seeds = [_hex_to_rgb(c) for c in LABEL_SEED_COLORS]
    seeds_lab = srgb_to_lab(np.array(seeds) / 255.0)
    grid, grid_lab = _candidate_colors()
    anchors = srgb_to_lab(np.array([[0, 0, 0], [255, 255, 255]]) / 255.0)

    assigned: dict[str, tuple[int, int, int]] = {}
    assigned_lab: dict[str, np.ndarray] = {}
    for label in labels:
        taken = [assigned_lab[n] for n in neighbors[label] if n in assigned_lab]
        conflicts = np.vstack([anchors, *taken]) if taken else anchors
        seed_dist = np.linalg.norm(seeds_lab[:, None] - conflicts[None], axis=-1).min(axis=1)
        free = np.flatnonzero(seed_dist >= min_seed_distance)
        if len(free):
            index = int(free[0])
            color, color_lab = seeds[index], seeds_lab[index]
        else:
            grid_dist = np.linalg.norm(grid_lab[:, None] - conflicts[None], axis=-1).min(axis=1)
            index = int(np.argmax(grid_dist))
            color = tuple(int(v) for v in grid[index])  # type: ignore[assignment]
            color_lab = grid_lab[index]
        assigned[label] = color
        assigned_lab[label] = color_lab
    return assigned


def load_coralscapes_v2_colors(
    path: Path = CORALSCAPES_V2_COLORS,
) -> dict[str, tuple[int, int, int]]:
    raw = json.loads(Path(path).read_text())
    return {str(name): tuple(int(v) for v in rgb[:3]) for name, rgb in raw.items()}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------------


def title_label(name: str, wrap: int | None = None) -> str:
    """Capitalize every word like ``str.title`` but keep acronyms (``CCA``, ``MASE: ...``)."""
    text = re.sub(r"(^|[\s(/_-])([a-z])", lambda m: m.group(1) + m.group(2).upper(), name)
    return textwrap.fill(text, width=wrap, break_long_words=False) if wrap else text


def patch_handle(label: str, color: Sequence[float], *, edge: Sequence[float] | None = None) -> Any:
    from matplotlib.patches import Patch

    return Patch(
        facecolor=color,
        edgecolor="none" if edge is None else edge,
        linewidth=0.0 if edge is None else 0.4,
        label=fmt_text(label),
    )


def point_handle(label: str, color: Sequence[float]) -> Any:
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=5.5,
        markerfacecolor=color,
        markeredgecolor=_rgb01(POINT_EDGE),
        markeredgewidth=0.6,
        label=fmt_text(label),
    )


def save_legend(
    handles: list[Any],
    path: Path,
    *,
    max_width_in: float = TEXT_WIDTH_IN,
    fontsize: float | None = None,
) -> int:
    """Borderless legend PDF using the most columns that fit ``max_width_in``.

    Returns the number of columns used.
    """
    import matplotlib.pyplot as plt

    if not handles:
        raise ValueError(f"no legend entries for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {} if fontsize is None else {"fontsize": fontsize}
    kwargs |= {
        "loc": "center",
        "frameon": False,
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.45,
        "columnspacing": 1.3,
        "labelspacing": 0.35,
        "borderaxespad": 0.0,
        "borderpad": 0.0,
    }
    fig = plt.figure(figsize=(max_width_in, 1.0))
    legend = fig.legend(handles=handles, ncol=1, **kwargs)
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer).width for t in legend.get_texts()]
    fontsize_px = legend.get_texts()[0].get_fontsize() * fig.dpi / 72.0
    # Each column is handle + pad + its widest text + column spacing; starting from the
    # mean text width overestimates ncol, which the loop below then reduces until it fits.
    column_px = float(np.mean(widths)) + (1.0 + 0.45 + 1.3) * fontsize_px
    ncol = int(max(1, min(len(handles), (max_width_in * fig.dpi + 1.3 * fontsize_px) // column_px)))
    while True:
        legend.remove()
        legend = fig.legend(handles=handles, ncol=ncol, **kwargs)
        width_in = legend.get_window_extent(renderer).width / fig.dpi
        if width_in <= max_width_in or ncol == 1:
            break
        ncol -= 1
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    return ncol


# ---------------------------------------------------------------------------
# Pie charts
# ---------------------------------------------------------------------------


def load_source_table(path: Path) -> pd.DataFrame:
    """Per-dataset source-level counts indexed by dataset key, with ``*_total`` columns."""
    table = pd.read_csv(path)
    table = table[(table["level"] == "source") & table["dataset_key"].notna()]
    table = table[table["dataset_key"].astype(str).str.len() > 0].set_index("dataset_key")
    for metric in METRICS:
        table[f"{metric}_total"] = sum(table[f"{metric}_{s}"] for s in PAPER_SPLITS)
    order = [spec.key for spec in DATASET_SPECS if spec.key in table.index]
    return table.loc[order]


def _compact(value: float) -> str:
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if value >= threshold:
            return (
                f"{value / threshold:.1f}{suffix}"
                if value < 100 * threshold
                else (f"{value / threshold:.0f}{suffix}")
            )
    return f"{value:,.0f}"


def plot_pie(
    values: pd.Series, path: Path, *, title: str | None, min_pct_label: float = 5.0
) -> None:
    import matplotlib.pyplot as plt

    values = values[values > 0]
    fig, ax = plt.subplots(figsize=(PIE_SIZE_IN, PIE_SIZE_IN + (0.3 if title else 0.0)))
    colors = [DATASET_COLORS[k] for k in values.index]
    wedges, _texts, autotexts = ax.pie(
        values.to_numpy(dtype=float),
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 0.6},
        autopct=lambda p: pct_text(p) if p >= min_pct_label else "",
        pctdistance=0.68,
        textprops={"fontsize": 6.5},
    )
    for wedge, text in zip(wedges, autotexts, strict=True):
        dark = _relative_luminance(wedge.get_facecolor()) < 0.35
        text.set_color("white" if dark else "black")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, pad=2)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.86 if title else 0.98)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)


def make_pies(table: pd.DataFrame, out_dir: Path, *, titles: bool) -> None:
    for metric, (noun, _ylabel) in METRICS.items():
        for split in PIE_SPLITS:
            values = table[f"{metric}_{split}"].astype(float)
            prefix = "All" if split == "total" else split.capitalize()
            title = f"{prefix} {noun}\n({_compact(values.sum())})" if titles else None
            plot_pie(values, out_dir / f"pie_{metric}_{split}.pdf", title=title)
    handles = [patch_handle(DATASET_DISPLAY[key], DATASET_COLORS[key]) for key in table.index]
    save_legend(handles, out_dir / "legend_datasets.pdf")
    logger.info("wrote %d pies and legend_datasets.pdf", len(METRICS) * len(PIE_SPLITS))


# ---------------------------------------------------------------------------
# Heavy-tail bar charts
# ---------------------------------------------------------------------------


def mapped_label_totals(counts: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mapped ``(dataset, source label)`` rows summed over splits, largest first."""
    mapped = counts[counts["mapped"].astype(str).str.lower().isin({"true", "1"})]
    totals = (
        mapped.groupby(["dataset_key", "source_label"], as_index=False)[metric]
        .sum()
        .sort_values([metric, "dataset_key", "source_label"], ascending=[False, True, True])
    )
    return totals[totals[metric] > 0].reset_index(drop=True)


def plot_label_bars(counts: pd.DataFrame, metric: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, MaxNLocator

    totals = mapped_label_totals(counts, metric)
    if totals.empty:
        logger.warning("no mapped labels with %s > 0; skipping %s", metric, path.name)
        return
    x = np.arange(len(totals))
    fig, ax = plt.subplots(figsize=BAR_SIZE_IN)
    ax.bar(
        x,
        totals[metric].to_numpy(dtype=float),
        width=1.0,
        align="edge",
        color=[DATASET_COLORS[k] for k in totals["dataset_key"]],
        linewidth=0,
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.set_xlim(0, len(totals))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_xlabel("Source label rank")
    ax.set_ylabel(METRICS[metric][1])
    ax.margins(x=0)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)


def make_bars(counts: pd.DataFrame, out_dir: Path) -> None:
    for metric in METRICS:
        plot_label_bars(counts, metric, out_dir / f"labels_{metric}.pdf")
    logger.info("wrote labels_{images,annotations,pixels}.pdf")


# ---------------------------------------------------------------------------
# Concept balance
# ---------------------------------------------------------------------------


def _mapped_mask(mapped: pd.Series) -> pd.Series:
    return mapped.astype(str).str.strip().str.lower().isin({"true", "1"})


def concept_codes(values: pd.Series) -> np.ndarray:
    """Map concept cells to ``0`` not given, ``1`` false, ``2`` true.

    Only the strings ``true`` and ``false`` are given. Empty cells, ``none``,
    and ``not_given`` stay not given, matching ``_encode_concept_row``.
    """
    missing = values.isna()
    text = values.astype(str).str.strip().str.lower().mask(missing, "")
    codes = np.zeros(len(values), dtype=np.int8)
    codes[text.eq("false").to_numpy()] = 1
    codes[text.eq("true").to_numpy()] = 2
    return codes


def concept_display_name(name: str) -> str:
    """Title-case a concept id for the axis (``hard_substrate`` → ``Hard Substrate``)."""
    return fmt_text(name.title())


def concept_state_weights(
    counts: pd.DataFrame,
    concepts: pd.DataFrame,
    *,
    phylum: str | None = None,
) -> pd.DataFrame:
    """True, false, and not-given weights for each health, morphology, and auxiliary concept.

    Mapped count rows are summed over splits. Each source label is weighted by its
    annotation count and its pixel count. Labels with no concept row stay in the
    denominator as not given. Duplicate concept rows keep the first.

    ``phylum`` restricts the denominator to source labels whose mapped phylum equals
    that name (case-insensitive), for example ``cnidaria``.
    """
    required_counts = {"dataset_key", "source_label", "mapped", "annotations", "pixels"}
    required_concepts = {"source_dataset_source", "source_label_class_name"}
    missing_counts = required_counts - set(counts.columns)
    missing_concepts = required_concepts - set(concepts.columns)
    if missing_counts or missing_concepts:
        raise ValueError(
            f"concept balance inputs are missing columns: "
            f"counts {sorted(missing_counts)}, concepts {sorted(missing_concepts)}"
        )

    mapped = counts.loc[_mapped_mask(counts["mapped"])].copy()
    mapped["source"] = mapped["dataset_key"].astype(str).str.strip().str.lower()
    mapped["label"] = mapped["source_label"].astype(str).str.strip().str.lower()
    weights = mapped.groupby(["source", "label"], as_index=False, sort=False)[
        ["annotations", "pixels"]
    ].sum()

    table = concepts.copy()
    table["source"] = table["source_dataset_source"].astype(str).str.strip().str.lower()
    table["label"] = table["source_label_class_name"].astype(str).str.strip().str.lower()
    present = [name for name in CONCEPT_NAMES if name in table.columns]
    table = table.drop_duplicates(["source", "label"], keep="first")
    keep_cols = ["source", "label", *present]
    if "phylum" in table.columns:
        keep_cols.append("phylum")
    joined = weights.merge(table[keep_cols], on=["source", "label"], how="left")
    if phylum is not None:
        if "phylum" not in joined.columns:
            joined = joined.iloc[0:0]
        else:
            phylum_text = joined["phylum"].astype(str).str.strip().str.lower()
            phylum_text = phylum_text.mask(joined["phylum"].isna(), "")
            joined = joined.loc[phylum_text.eq(phylum.strip().lower())]

    annotations = joined["annotations"].to_numpy(dtype=np.int64)
    pixels = joined["pixels"].to_numpy(dtype=np.int64)
    rows: list[dict[str, int | str]] = []
    for group, names in CONCEPT_SECTIONS:
        for name in names:
            codes = (
                concept_codes(joined[name])
                if name in joined.columns
                else np.zeros(len(joined), dtype=np.int8)
            )
            rows.append(
                {
                    "concept": name,
                    "group": group,
                    "annotations_not_given": int(annotations[codes == 0].sum()),
                    "annotations_false": int(annotations[codes == 1].sum()),
                    "annotations_true": int(annotations[codes == 2].sum()),
                    "pixels_not_given": int(pixels[codes == 0].sum()),
                    "pixels_false": int(pixels[codes == 1].sum()),
                    "pixels_true": int(pixels[codes == 2].sum()),
                }
            )
    return pd.DataFrame(rows)


def _concept_section_marks() -> tuple[list[float], list[tuple[str, float]]]:
    """Vertical-line positions and ``(section, tick center)`` for the concept axis."""
    lines: list[float] = []
    centers: list[tuple[str, float]] = []
    start = 0
    for index, (group, names) in enumerate(CONCEPT_SECTIONS):
        center = start + (len(names) - 1) / 2
        centers.append((group, center))
        start += len(names)
        if index < len(CONCEPT_SECTIONS) - 1:
            lines.append(start - 0.5)
    return lines, centers


def plot_concept_balance(
    weights: pd.DataFrame,
    cnidarian: pd.DataFrame,
    path: Path,
) -> None:
    """Draw all-data and Cnidaria-only shares for each binary concept.

    Each panel is labeled under its concept names: all annotations, all annotated
    pixels, cnidarian annotations, and cnidarian annotated pixels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    for frame in (weights, cnidarian):
        if list(frame["concept"]) != list(CONCEPT_NAMES):
            raise ValueError("concept balance rows must follow CONCEPT_SECTIONS order")

    fig, axes = plt.subplots(2, 2, figsize=CONCEPT_BALANCE_SIZE_IN, sharex=True, sharey=True)
    panels = (
        (axes[0, 0], weights, "annotations", "(a) All Annotations"),
        (axes[0, 1], weights, "pixels", "(b) All Annotated Pixels"),
        (axes[1, 0], cnidarian, "annotations", "(c) Cnidarian Annotations"),
        (axes[1, 1], cnidarian, "pixels", "(d) Cnidarian Annotated Pixels"),
    )
    x = np.arange(len(weights))
    lines, centers = _concept_section_marks()
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for ax, frame, metric, letter in panels:
        bottom = np.zeros(len(frame), dtype=float)
        state_counts = [
            frame[f"{metric}_{state}"].to_numpy(dtype=float) for state, _, _ in CONCEPT_STATES
        ]
        total = np.sum(state_counts, axis=0)
        for (_, label, color), count in zip(CONCEPT_STATES, state_counts, strict=True):
            percent = np.divide(count, total, out=np.zeros_like(count), where=total > 0) * 100.0
            bars = ax.bar(
                x,
                percent,
                width=0.92,
                bottom=bottom,
                align="center",
                color=color,
                linewidth=0,
                label=label,
                zorder=2,
            )
            bottom += percent
            if ax is axes[0, 0]:
                legend_handles.append(bars)
                legend_labels.append(label)
        for line in lines:
            ax.axvline(line, color="0.35", linestyle="--", linewidth=0.6, zorder=3)
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for group, center in centers:
            ax.text(
                center,
                1.02,
                fmt_text(group),
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=6.5,
                clip_on=False,
            )
        ax.set_xlim(-0.5, len(frame) - 0.5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [concept_display_name(name) for name in frame["concept"]],
            rotation=90,
            ha="center",
            va="top",
            fontsize=4.5,
        )
        # sharex hides the top-row names; put them back under both rows.
        ax.tick_params(axis="x", length=2, pad=1, labelbottom=True)
        ax.set_xlabel(fmt_text(letter), fontsize=8, labelpad=3)
    for ax in axes.flat:
        ax.tick_params(axis="x", labelbottom=True)
    axes[0, 0].set_ylabel("Share (%)")
    axes[1, 0].set_ylabel("Share (%)")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.17, top=0.94, wspace=0.08, hspace=0.62)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def make_concept_balance(counts: pd.DataFrame, concepts: pd.DataFrame, path: Path) -> None:
    weights = concept_state_weights(counts, concepts)
    cnidarian = concept_state_weights(counts, concepts, phylum=CNIDARIAN_PHYLUM)
    plot_concept_balance(weights, cnidarian, path)
    logger.info("wrote %s", path.name)


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------


def _open_resized(path: Path, max_edge: int) -> tuple[Any, float]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = min(1.0, max_edge / max(w, h)) if max_edge > 0 else 1.0
    if scale < 1.0:
        image = image.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS
        )
    return image, scale


def _stems(directory: Path, suffix: str) -> list[str]:
    return sorted(p.name[: -len(suffix)] for p in directory.glob(f"*{suffix}"))


def draw_points(
    image: Any,
    points: pd.DataFrame,
    colors: Mapping[str, Sequence[int]],
    *,
    point_scale: float = DEFAULT_POINT_SCALE,
) -> Any:
    """Filled discs with a thin dark outline; radius is ``point_scale`` x the longest edge."""
    from PIL import ImageDraw

    image = image.copy()
    draw = ImageDraw.Draw(image)
    radius = max(3.0, max(image.size) * point_scale)
    outline = max(1, round(radius / 3.5))
    for row, col, label in points[["row", "col", "source_label_name"]].itertuples(index=False):
        x, y = float(col), float(row)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=tuple(colors[str(label)]),
            outline=POINT_EDGE,
            width=outline,
        )
    return image


def _read_points(path: Path, scale: float) -> pd.DataFrame:
    points = pd.read_csv(path, dtype={"source_label_name": str}, keep_default_na=False)
    points["row"] = points["row"].astype(float) * scale
    points["col"] = points["col"].astype(float) * scale
    return points


def _point_labels_by_frequency(point_files: Iterable[Path]) -> list[str]:
    frames = [
        pd.read_csv(p, usecols=["source_label_name"], dtype=str, keep_default_na=False)
        for p in point_files
    ]
    if not frames:
        return []
    counts = pd.concat(frames)["source_label_name"].value_counts()
    return sorted(counts.index, key=lambda name: (-int(counts[name]), name))


def _sparse_legend(
    labels: Iterable[str], colors: Mapping[str, Sequence[int]], *, wrap: int | None = None
) -> list[Any]:
    return [
        point_handle(title_label(name, wrap), _rgb01(colors[name]))
        for name in sorted(set(labels), key=str.casefold)
    ]


def render_sparse(
    in_dir: Path,
    out_dir: Path,
    colors: Mapping[str, Sequence[int]],
    *,
    max_edge: int,
    point_scale: float = DEFAULT_POINT_SCALE,
) -> set[str]:
    """Write ``NN_overlay.jpg`` for every exported sample. Returns the labels drawn."""
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for stem in _stems(in_dir, "_points.csv"):
        image, scale = _open_resized(in_dir / f"{stem}_image.jpg", max_edge)
        points = _read_points(in_dir / f"{stem}_points.csv", scale)
        overlay = draw_points(image, points, colors, point_scale=point_scale)
        overlay.save(out_dir / f"{stem}_overlay.jpg", quality=93)
        used.update(points["source_label_name"].astype(str))
    return used


def make_sparse_dataset(
    key: str,
    in_dir: Path,
    out_dir: Path,
    *,
    max_edge: int,
    point_scale: float = DEFAULT_POINT_SCALE,
) -> None:
    point_files = sorted(in_dir.glob("*_points.csv"))
    colors = assign_colors(_point_labels_by_frequency(point_files))
    used = render_sparse(in_dir, out_dir, colors, max_edge=max_edge, point_scale=point_scale)
    save_legend(_sparse_legend(used, colors, wrap=LEGEND_WRAP), out_dir / f"legend_{key}.pdf")
    logger.info("%s: %d overlays, %d legend entries", key, len(point_files), len(used))


def _dense_colors(
    key: str, id_to_name: Mapping[int, str], labels: Sequence[np.ndarray]
) -> dict[str, tuple[int, int, int]]:
    if key == "coralscapes_v2" and CORALSCAPES_V2_COLORS.is_file():
        return load_coralscapes_v2_colors()
    pixels: dict[str, int] = {}
    for label in labels:
        ids, n = np.unique(label, return_counts=True)
        for class_id, count in zip(ids.tolist(), n.tolist(), strict=True):
            name = id_to_name.get(int(class_id))
            if class_id > 0 and name is not None and not _is_ignore(name):
                pixels[name] = pixels.get(name, 0) + int(count)
    return assign_colors(sorted(pixels, key=lambda name: (-pixels[name], name)))


def _is_ignore(name: str) -> bool:
    return name.strip().lower() in IGNORE_LABEL_NAMES


def colorize_mask(
    label: np.ndarray, id_to_name: Mapping[int, str], colors: Mapping[str, Sequence[int]]
) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:] = UNLABELED_COLOR
    for class_id, name in id_to_name.items():
        if 0 < class_id < 256 and not _is_ignore(name) and name in colors:
            lut[class_id] = colors[name][:3]
    return lut[label.astype(np.uint8)]


def make_dense_dataset(key: str, in_dir: Path, out_dir: Path, *, max_edge: int) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_name = {
        int(k): str(v) for k, v in json.loads((in_dir / "classes.json").read_text()).items()
    }
    stems = _stems(in_dir, "_label.png")
    labels = [np.asarray(Image.open(in_dir / f"{s}_label.png")) for s in stems]
    labels = [lab[..., 0] if lab.ndim == 3 else lab for lab in labels]
    colors = _dense_colors(key, id_to_name, labels)

    present: set[str] = set()
    has_unlabeled = False
    for stem, label in zip(stems, labels, strict=True):
        image, _ = _open_resized(in_dir / f"{stem}_image.jpg", max_edge)
        image.save(out_dir / f"{stem}_image.jpg", quality=95)
        mask = Image.fromarray(colorize_mask(label, id_to_name, colors))
        mask.resize(image.size, Image.Resampling.NEAREST).save(out_dir / f"{stem}_label.png")
        for class_id in np.unique(label).tolist():
            name = id_to_name.get(int(class_id))
            if class_id == 0 or name is None or _is_ignore(name):
                has_unlabeled = True
            elif name in colors:
                present.add(name)

    handles = [
        patch_handle(title_label(name, LEGEND_WRAP), _rgb01(colors[name]))
        for name in sorted(present, key=str.casefold)
    ]
    if has_unlabeled:
        handles.append(patch_handle("Unlabeled", _rgb01(UNLABELED_COLOR), edge=UNLABELED_EDGE))
    save_legend(handles, out_dir / f"legend_{key}.pdf")
    logger.info("%s: %d image/label pairs, %d legend entries", key, len(stems), len(handles))


def make_coralnet(
    in_dir: Path, out_dir: Path, *, max_edge: int, point_scale: float = DEFAULT_POINT_SCALE
) -> None:
    """Shared labels share a color; labels within one source are mutually distinct."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads((in_dir / "sources.json").read_text())
    point_files = sorted(in_dir.glob("s*/*_points.csv"))
    per_source = [
        set(_point_labels_by_frequency((in_dir / f"s{int(e['source_id'])}").glob("*_points.csv")))
        for e in sources
    ]
    colors = assign_colors(_point_labels_by_frequency(point_files), groups=per_source)
    lines = ["source_id\tsplit\timages_in_source\turl"]
    for entry in sources:
        sid = int(entry["source_id"])
        used = render_sparse(
            in_dir / f"s{sid}",
            out_dir / f"s{sid}",
            colors,
            max_edge=max_edge,
            point_scale=point_scale,
        )
        if used:
            save_legend(
                _sparse_legend(used, colors, wrap=CORALNET_LEGEND_WRAP),
                out_dir / f"legend_s{sid}.pdf",
                max_width_in=CORALNET_LEGEND_WIDTH_IN,
                fontsize=CORALNET_LEGEND_FONTSIZE,
            )
        lines.append(f"{sid}\t{entry['split']}\t{entry['n_images_in_source']}\t{entry['url']}")
        logger.info("coralnet s%d: %d legend entries", sid, len(used))
    (out_dir / "sources.txt").write_text("\n".join(lines) + "\n")


def make_samples(
    samples_dir: Path,
    out_dir: Path,
    *,
    max_edge: int,
    point_scale: float = DEFAULT_POINT_SCALE,
) -> None:
    for spec in DATASET_SPECS:
        in_dir = samples_dir / spec.key
        if not in_dir.is_dir():
            logger.warning("no samples for %s in %s", spec.key, samples_dir)
            continue
        if spec.key == "coralnet":
            make_coralnet(in_dir, out_dir / spec.key, max_edge=max_edge, point_scale=point_scale)
        elif spec.key in DENSE_KEYS:
            make_dense_dataset(spec.key, in_dir, out_dir / spec.key, max_edge=max_edge)
        else:
            make_sparse_dataset(
                spec.key, in_dir, out_dir / spec.key, max_edge=max_edge, point_scale=point_scale
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="dataset_table.csv (default: <data-dir>/dataset_table.csv, then dataset_stats/).",
    )
    parser.add_argument("--max-edge", type=int, default=1024, help="Longest edge of sample images.")
    parser.add_argument(
        "--point-scale",
        type=float,
        default=DEFAULT_POINT_SCALE,
        help="Sparse point radius as a fraction of the longest image edge.",
    )
    parser.add_argument(
        "--concepts-csv",
        type=Path,
        default=DEFAULT_CONCEPTS_CSV,
        help="class_to_concepts.csv used for the concept-balance figure.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=("pies", "bars", "samples", "concepts"),
        default=None,
    )
    parser.add_argument("--titles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--usetex", action="store_true", help="Render text with a LaTeX install.")
    return parser.parse_args(argv)


def _resolve_table(args: argparse.Namespace) -> Path | None:
    candidates = (
        [args.table]
        if args.table
        else [
            args.data_dir / "dataset_table.csv",
            _REPO_ROOT / "dataset_stats" / "dataset_table.csv",
        ]
    )
    return next((p for p in candidates if p is not None and p.is_file()), None)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    setup_style(args.usetex)
    parts = set(args.only or ("pies", "bars", "samples", "concepts"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "pies" in parts:
        table_path = _resolve_table(args)
        if table_path is None:
            logger.warning("no dataset_table.csv found; skipping pies")
        else:
            make_pies(load_source_table(table_path), args.output_dir, titles=args.titles)
    if "bars" in parts:
        counts_path = args.data_dir / "label_counts.csv"
        if counts_path.is_file():
            make_bars(pd.read_csv(counts_path, keep_default_na=False), args.output_dir)
        else:
            logger.warning("%s missing; skipping bar charts", counts_path)
    if "concepts" in parts:
        counts_path = args.data_dir / "label_counts.csv"
        if counts_path.is_file() and args.concepts_csv.is_file():
            make_concept_balance(
                pd.read_csv(counts_path, keep_default_na=False),
                pd.read_csv(args.concepts_csv),
                args.output_dir / "concepts_balance.pdf",
            )
        else:
            logger.warning(
                "skipping concept balance; need %s and %s", counts_path, args.concepts_csv
            )
    if "samples" in parts:
        make_samples(
            args.data_dir / "samples",
            args.output_dir,
            max_edge=args.max_edge,
            point_scale=args.point_scale,
        )
    logger.info("done: %s", args.output_dir)


if __name__ == "__main__":
    main()
