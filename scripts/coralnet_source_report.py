#!/usr/bin/env python3
"""Build a static HTML report on the quality of every CoralNet source.

Reads the CoralNet training manifest (the same parquet ``CoralNetDataset`` uses),
joins source labels to ``configs/class_to_concepts.csv`` for genus and class, and
writes a self-contained folder that opens offline in any browser:

    <output-dir>/
      index.html                  sortable / filterable table + 6 histograms
      source_stats.csv            same table, machine-readable
      hist/<metric>.png           distribution of each metric over sources
      sources/s<id>.html          per-source page
      sources/s<id>/*.png|*.jpg   label / genus / coral genus bars + overlays

Per-source metrics:

* images — unique ``image_id``.
* annotated pixels — point-annotation rows (one point is one pixel, as in
  ``scripts/dataset_table_stats.py``).
* source labels — unique ``source_label_name``.
* genera — unique ``genus`` in ``class_to_concepts.csv``; ``none``, ``not_given``
  and labels missing from the CSV do not count.
* coral genera — genera whose ``class`` is ``hexacorallia`` or ``octocorallia``.
* avg pixels per image — annotated pixels / images.

Images are read local-cache-first and fall back to S3. The cache is never
written to.

Usage:
    python scripts/coralnet_source_report.py --workers 32
    python scripts/coralnet_source_report.py --splits train val test --n-images 10
    python scripts/coralnet_source_report.py --sources 23 57 --output-dir /tmp/report
    python scripts/coralnet_source_report.py --skip-images
"""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import logging
import multiprocessing as mp
import os
import traceback
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("coralnet_source_report")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = _REPO_ROOT / "configs" / "data_config_dedup.yaml"
DEFAULT_CONCEPTS_CSV = _REPO_ROOT / "configs" / "class_to_concepts.csv"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "coralnet_source_report"
DEFAULT_BUCKET = "dev-datamermaid-sm-sources"
ID2NAME_KEY = "coralnet-public-images/temporary/coralnet_id2name.json"

MISSING_TAXON = frozenset({"", "none", "not_given", "nan"})
CORAL_CLASSES = frozenset({"hexacorallia", "octocorallia"})
SPLITS: tuple[str, ...] = ("train", "val", "test")

MANIFEST_COLUMNS = ["source_id", "image_id", "row", "col", "coralnet_id", "image_s3_key"]
WORKER_COLUMNS = ["image_id", "row", "col", "label", "image_key", "genus", "coral_genus"]

MAX_BARS = 200


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    log_x: bool
    fmt: str


METRICS: tuple[Metric, ...] = (
    Metric("n_images", "Images", True, "{:,.0f}"),
    Metric("n_pixels", "Annotated pixels", True, "{:,.0f}"),
    Metric("n_labels", "Source labels", False, "{:,.0f}"),
    Metric("n_genera", "Genera", False, "{:,.0f}"),
    Metric("n_coral_genera", "Coral genera", False, "{:,.0f}"),
    Metric("avg_pixels_per_image", "Avg pixels / image", True, "{:,.1f}"),
)
EXTRA_COLUMNS: tuple[Metric, ...] = (
    Metric("pct_genus_level", "% pixels genus-level", False, "{:.1f}"),
    Metric("pct_unmapped", "% pixels unmapped", False, "{:.1f}"),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def resolve_cache_dir(data_config: Path, override: str | None) -> str | None:
    if override:
        return override
    env = os.environ.get("MERMAIDSEG_LOCAL_CACHE_DIR")
    if env:
        return env
    cfg = yaml.safe_load(data_config.read_text())
    return (cfg.get("data") or {}).get("local_cache_dir")


def configure_cache(cache_dir: str | None) -> None:
    from mermaidseg.datasets.local_cache import LocalS3Cache

    LocalS3Cache.configure(cache_dir, write_through=False)


def load_splits(data_config: Path) -> dict[int, str]:
    """Map source id -> yaml split for every whitelisted CoralNet source."""
    cfg = yaml.safe_load(data_config.read_text())
    coralnet = (cfg.get("data") or {}).get("coralnet") or {}
    out: dict[int, str] = {}
    for split in SPLITS:
        for sid in (coralnet.get(split) or {}).get("whitelist_sources") or []:
            out[int(sid)] = split
    return out


def select_sources(
    splits: dict[int, str],
    *,
    only_splits: list[str] | None,
    only_sources: list[int] | None,
) -> list[int] | None:
    """Source ids to load. ``None`` means the whole manifest."""
    if not only_splits:
        return list(only_sources) if only_sources is not None else None
    allowed = {sid for sid, split in splits.items() if split in set(only_splits)}
    if only_sources is not None:
        allowed &= {int(sid) for sid in only_sources}
    return sorted(allowed)


def _read_manifest(bucket: str, manifest: str, sources: list[int] | None) -> pd.DataFrame:
    """Read the training manifest from a local path or S3.

    S3 reads go through boto3. PyArrow's ``s3://`` filesystem does not use AWS SSO
    profiles, which is how this bucket is authenticated.
    """
    import io

    from mermaidseg.datasets.local_cache import LocalS3Cache, parse_storage_ref

    filters = [("source_id", "in", sources)] if sources else None
    parsed = parse_storage_ref(manifest, bucket)
    if parsed is None:
        logger.info("reading manifest %s", manifest)
        df = pd.read_parquet(manifest, columns=MANIFEST_COLUMNS, filters=filters)
    else:
        cache = LocalS3Cache.get()
        local = cache.local_path(*parsed) if cache.enabled else None
        if local is not None and local.is_file():
            logger.info("reading manifest %s", local)
            df = pd.read_parquet(local, columns=MANIFEST_COLUMNS, filters=filters)
        else:
            logger.info("reading manifest s3://%s/%s", *parsed)
            data = cache.read_bytes(*parsed)
            df = pd.read_parquet(io.BytesIO(data), columns=MANIFEST_COLUMNS, filters=filters)
    if sources:
        df = df[df["source_id"].isin(sources)]
    return df


def load_concepts(path: Path) -> pd.DataFrame:
    """CoralNet rows of ``class_to_concepts.csv`` indexed by lowercased label name."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[frame["source_dataset_source"].str.strip().str.lower() == "coralnet"]
    frame = frame.assign(
        label=frame["source_label_class_name"].str.strip().str.lower(),
        genus=frame["genus"].str.strip().str.lower(),
        cls=frame["class"].str.strip().str.lower(),
    )
    frame = frame.drop_duplicates("label").set_index("label")[["genus", "cls"]]
    frame["genus"] = frame["genus"].where(~frame["genus"].isin(MISSING_TAXON))
    frame["coral_genus"] = frame["genus"].where(frame["cls"].isin(CORAL_CLASSES))
    frame["mapped"] = 1
    return frame


def load_annotations(
    bucket: str, manifest: str, concepts_csv: Path, sources: list[int] | None
) -> pd.DataFrame:
    """One row per point with label, genus, coral genus and image key."""
    from mermaidseg.datasets.local_cache import LocalS3Cache

    df = _read_manifest(bucket, manifest, sources)
    id2name = LocalS3Cache.get().read_json(bucket, ID2NAME_KEY)
    logger.info("manifest: %d points, %d sources", len(df), df["source_id"].nunique())

    codes = df["coralnet_id"].astype("category")
    names = pd.Series(codes.cat.categories).map(lambda x: id2name.get(str(x)))
    label_cats = names.astype(str).str.strip().str.lower()

    concepts = load_concepts(concepts_csv).reindex(label_cats.values)
    out = pd.DataFrame(
        {
            "source_id": df["source_id"].astype(int).to_numpy(),
            "image_id": df["image_id"].to_numpy(),
            "row": df["row"].to_numpy(),
            "col": df["col"].to_numpy(),
            "image_key": pd.Categorical(df["image_s3_key"]),
        }
    )
    idx = codes.cat.codes.to_numpy()
    for column, values in (
        ("label", label_cats.to_numpy()),
        ("genus", concepts["genus"].to_numpy()),
        ("coral_genus", concepts["coral_genus"].to_numpy()),
        ("mapped", concepts["mapped"].notna().to_numpy()),
    ):
        taken = values[idx]
        out[column] = taken if column == "mapped" else pd.Categorical(taken)
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def compute_source_stats(df: pd.DataFrame, splits: dict[int, str]) -> pd.DataFrame:
    grouped = df.groupby("source_id", observed=True)
    stats = pd.DataFrame(
        {
            "n_images": grouped["image_id"].nunique(),
            "n_pixels": grouped.size(),
            "n_labels": grouped["label"].nunique(),
            "n_genera": grouped["genus"].nunique(),
            "n_coral_genera": grouped["coral_genus"].nunique(),
            "pct_genus_level": 100.0 * grouped["genus"].count() / grouped.size(),
            "pct_unmapped": 100.0 * (1.0 - grouped["mapped"].mean()),
            "n_images_with_key": df.dropna(subset=["image_key"])
            .groupby("source_id", observed=True)["image_id"]
            .nunique(),
        }
    )
    stats["n_images_with_key"] = stats["n_images_with_key"].fillna(0).astype(int)
    stats["avg_pixels_per_image"] = stats["n_pixels"] / stats["n_images"]
    stats["split"] = [splits.get(int(s), "none") for s in stats.index]
    stats["coralnet_url"] = [f"https://coralnet.ucsd.edu/source/{s}/" for s in stats.index]
    stats.index.name = "source_id"
    ordered = [m.key for m in METRICS] + [m.key for m in EXTRA_COLUMNS]
    return stats[["split", *ordered, "n_images_with_key", "coralnet_url"]].sort_values(
        "n_images", ascending=False
    )


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; color: #222; }
h1, h2 { font-weight: 600; }
a { color: #0b62a4; text-decoration: none; }
a:hover { text-decoration: underline; }
.hists { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px;
         max-width: 1600px; }
.hists figure { margin: 0; }
.hists img { width: 100%; border: 1px solid #ddd; }
.controls { margin: 16px 0; display: flex; gap: 12px; align-items: center; white-space: nowrap; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; }
th, td { padding: 4px 10px; border-bottom: 1px solid #eee; text-align: right; }
th { position: sticky; top: 0; background: #f5f5f5; cursor: pointer; user-select: none; }
th.sorted-asc::after { content: " \\25B2"; }
th.sorted-desc::after { content: " \\25BC"; }
td.l, th.l { text-align: left; }
tr:hover td { background: #fafafa; }
.stats { display: flex; flex-wrap: wrap; gap: 24px; margin: 12px 0 24px; }
.stat { background: #f5f5f5; padding: 8px 14px; border-radius: 6px; }
.stat b { display: block; font-size: 20px; }
.bars img { max-width: 100%; border: 1px solid #ddd; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 12px; }
.grid figure { margin: 0; }
.grid img { width: 100%; border: 1px solid #ddd; }
.grid figcaption { font-size: 12px; color: #555; }
.note { color: #a33; }
details { margin: 16px 0; }
"""

SORT_JS = """
const table = document.getElementById('t');
const tbody = table.tBodies[0];
const headers = table.tHead.rows[0].cells;
let sortCol = -1, sortDir = 1;
for (let i = 0; i < headers.length; i++) {
  headers[i].addEventListener('click', () => {
    sortDir = (sortCol === i) ? -sortDir : (headers[i].dataset.type === 'num' ? -1 : 1);
    sortCol = i;
    const num = headers[i].dataset.type === 'num';
    const rows = Array.from(tbody.rows);
    rows.sort((a, b) => {
      const x = a.cells[i].dataset.v, y = b.cells[i].dataset.v;
      return sortDir * (num ? parseFloat(x) - parseFloat(y) : x.localeCompare(y));
    });
    rows.forEach(r => tbody.appendChild(r));
    for (const h of headers) h.classList.remove('sorted-asc', 'sorted-desc');
    headers[i].classList.add(sortDir > 0 ? 'sorted-asc' : 'sorted-desc');
  });
}
const q = document.getElementById('q'), split = document.getElementById('split');
const count = document.getElementById('count');
function applyFilter() {
  const text = q.value.trim().toLowerCase(), s = split.value;
  let shown = 0;
  for (const r of tbody.rows) {
    const ok = (!text || r.textContent.toLowerCase().includes(text))
      && (!s || r.dataset.split === s);
    r.style.display = ok ? '' : 'none';
    shown += ok;
  }
  count.textContent = shown + ' sources shown';
}
q.addEventListener('input', applyFilter);
split.addEventListener('change', applyFilter);
applyFilter();
"""


def _page(title: str, body: str, script: str = "") -> str:
    js = f"<script>{script}</script>" if script else ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}{js}</body></html>"
    )


def render_histograms(stats: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    hist_dir = out_dir / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        values = stats[metric.key].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6, 3.6), dpi=110)
        if metric.log_x:
            positive = values[values > 0]
            lo, hi = max(positive.min(), 1e-1), positive.max()
            bins = np.logspace(np.log10(lo), np.log10(hi * 1.0001), 40)
            ax.hist(positive, bins=bins, color="#3a7dbf", edgecolor="white", linewidth=0.4)
            ax.set_xscale("log")
        else:
            top = int(values.max()) if len(values) else 0
            width = max(1, int(np.ceil((top + 1) / 60)))
            bins = np.arange(-0.5, top + width + 0.5, width)
            ax.hist(values, bins=bins, color="#3a7dbf", edgecolor="white", linewidth=0.4)
        median = float(np.median(values)) if len(values) else 0.0
        ax.axvline(median, color="#d9534f", linestyle="--", linewidth=1)
        ax.set_title(f"{metric.title} (median {metric.fmt.format(median)})", fontsize=10)
        ax.set_xlabel(metric.title)
        ax.set_ylabel("Sources")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(hist_dir / f"{metric.key}.png")
        plt.close(fig)


def _cell(value: Any, fmt: str) -> str:
    v = float(value)
    return f"<td data-v='{v}'>{fmt.format(v)}</td>"


def render_index(stats: pd.DataFrame, out_dir: Path, manifest: str) -> None:
    columns = (*METRICS, *EXTRA_COLUMNS)
    header = (
        "<tr><th class='l' data-type='num'>Source</th><th class='l' data-type='str'>Split</th>"
        + "".join(f"<th data-type='num'>{html.escape(m.title)}</th>" for m in columns)
        + "<th class='l' data-type='str'>CoralNet</th></tr>"
    )
    rows = []
    for sid, row in stats.iterrows():
        cells = "".join(_cell(row[m.key], m.fmt) for m in columns)
        rows.append(
            f"<tr data-split='{row['split']}'>"
            f"<td class='l' data-v='{sid}'><a href='sources/s{sid}.html'>s{sid}</a></td>"
            f"<td class='l' data-v='{row['split']}'>{row['split']}</td>{cells}"
            f"<td class='l' data-v='{sid}'><a href='{row['coralnet_url']}' target='_blank'>"
            "coralnet.ucsd.edu</a></td></tr>"
        )
    hists = "".join(
        f"<figure><img src='hist/{m.key}.png' alt='{html.escape(m.title)}'></figure>"
        for m in METRICS
    )
    split_options = "".join(f"<option value='{s}'>{s}</option>" for s in (*SPLITS, "none"))
    body = (
        "<h1>CoralNet source quality</h1>"
        f"<p>{len(stats):,} sources, {int(stats['n_images'].sum()):,} images, "
        f"{int(stats['n_pixels'].sum()):,} annotated pixels (one per point). "
        f"Manifest: <code>{html.escape(manifest)}</code>. "
        "Genera and coral genera (hexacorallia + octocorallia) come from "
        "<code>configs/class_to_concepts.csv</code>.</p>"
        f"<h2>Distributions over sources</h2><div class='hists'>{hists}</div>"
        "<h2>Sources</h2>"
        "<div class='controls'><input id='q' placeholder='Filter...' size='30'>"
        f"<select id='split'><option value=''>all splits</option>{split_options}</select>"
        "<span id='count'></span><a href='source_stats.csv'>source_stats.csv</a></div>"
        f"<div class='scroll'><table id='t'><thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    (out_dir / "index.html").write_text(_page("CoralNet source quality", body, SORT_JS))


# ---------------------------------------------------------------------------
# Per-source pages (run in worker processes)
# ---------------------------------------------------------------------------


def _palette(n: int) -> list[tuple[float, float, float]]:
    import matplotlib as mpl

    base: list[tuple[float, float, float]] = []
    for name in ("tab20", "tab20b", "tab20c"):
        base.extend(tuple(c[:3]) for c in mpl.colormaps[name].colors)
    extra = [
        colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.75, 0.95)
        for i in range(max(0, n - len(base)))
    ]
    return (base + extra)[:n]


def _bar_chart(counts: pd.Series, title: str, path: Path, colors: dict[str, Any] | None) -> bool:
    import matplotlib.pyplot as plt

    if counts.empty:
        return False
    hidden = max(0, len(counts) - MAX_BARS)
    shown = counts.iloc[:MAX_BARS]
    height = 1.2 + 0.2 * len(shown)
    fig, ax = plt.subplots(figsize=(9, height), dpi=90)
    y = np.arange(len(shown))
    bar_colors = [colors.get(k, "#3a7dbf") for k in shown.index] if colors else "#3a7dbf"
    ax.barh(y, shown.to_numpy(), color=bar_colors, edgecolor="#333", linewidth=0.3)
    ax.set_yticks(y, [str(k) for k in shown.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    total = counts.sum()
    for yi, v in zip(y, shown.to_numpy(), strict=True):
        ax.text(v, yi, f" {v:,} ({100 * v / total:.1f}%)", va="center", fontsize=7)
    suffix = f" ({hidden} more not shown)" if hidden else ""
    ax.set_title(f"{title}: {len(counts)} distinct{suffix}", fontsize=10, loc="left")
    ax.set_xlabel("Annotated pixels (log)")
    ax.set_xlim(0.8, shown.max() * 8)
    ax.margins(y=0.01)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _render_overlay(
    image: Any,
    points: pd.DataFrame,
    colors: dict[str, Any],
    max_edge: int,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    w, h = image.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    w2, h2 = image.size

    counts = points["label"].value_counts()
    rows_per_column = max(1, int(max(h2, 200) / 15))
    ncol = int(np.ceil(len(counts) / rows_per_column))
    legend_px = 260 * max(1, ncol)
    dpi = 100
    fig = plt.figure(figsize=((w2 + legend_px) / dpi, max(h2, 200) / dpi), dpi=dpi)
    ax = fig.add_axes((0, 1 - h2 / max(h2, 200), w2 / (w2 + legend_px), h2 / max(h2, 200)))
    ax.imshow(image)
    ax.set_xlim(-0.5, w2 - 0.5)
    ax.set_ylim(h2 - 0.5, -0.5)
    ax.axis("off")

    radius_px = max(3.0, max(w2, h2) / 160)
    size_pt2 = (2 * radius_px * 72 / dpi) ** 2
    ax.scatter(
        points["col"].to_numpy() * scale,
        points["row"].to_numpy() * scale,
        s=size_pt2,
        c=[colors[label] for label in points["label"]],
        edgecolors="black",
        linewidths=0.8,
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=colors[label],
            markeredgecolor="black",
            label=f"{label} ({n})",
        )
        for label, n in counts.items()
    ]
    leg_ax = fig.add_axes((w2 / (w2 + legend_px), 0, legend_px / (w2 + legend_px), 1))
    leg_ax.axis("off")
    leg_ax.legend(
        handles=handles,
        loc="upper left",
        ncol=max(1, ncol),
        fontsize=7,
        frameon=False,
        borderaxespad=0.3,
        handletextpad=0.3,
    )
    fig.savefig(path, pil_kwargs={"quality": 85})
    plt.close(fig)


def _source_table(df: pd.DataFrame) -> str:
    table = (
        df.groupby("label", observed=True)
        .agg(
            pixels=("label", "size"),
            genus=("genus", "first"),
            coral=("coral_genus", "first"),
        )
        .sort_values("pixels", ascending=False)
    )
    rows = "".join(
        f"<tr><td class='l'>{html.escape(str(label))}</td>"
        f"<td class='l'>{'' if pd.isna(r.genus) else html.escape(str(r.genus))}</td>"
        f"<td class='l'>{'yes' if not pd.isna(r.coral) else ''}</td>"
        f"<td>{r.pixels:,}</td></tr>"
        for label, r in table.iterrows()
    )
    return (
        "<details><summary>Label table (label, genus, coral, pixels)</summary>"
        "<table><thead><tr><th class='l'>Source label</th><th class='l'>Genus</th>"
        "<th class='l'>Coral</th><th>Pixels</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )


def _init_worker(cache_dir: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    configure_cache(cache_dir)


def _source_nav(prev_id: int | None, next_id: int | None) -> str:
    parts = ["<a href='../index.html'>&larr; all sources</a>"]
    if prev_id is not None:
        parts.append(f"<a href='s{prev_id}.html'>&larr; s{prev_id}</a>")
    if next_id is not None:
        parts.append(f"<a href='s{next_id}.html'>s{next_id} &rarr;</a>")
    return "<p>" + " &middot; ".join(parts) + "</p>"


def render_source(
    source_id: int,
    df: pd.DataFrame,
    stats_row: dict[str, Any],
    out_dir: Path,
    bucket: str,
    n_images: int,
    max_edge: int,
    seed: int,
    skip_images: bool,
    resume: bool = False,
    prev_id: int | None = None,
    next_id: int | None = None,
) -> dict[str, Any]:
    """Write ``sources/s<id>.html`` and its figures. Never raises."""
    page = out_dir / "sources" / f"s{source_id}.html"
    if resume and page.is_file():
        return {"source_id": source_id, "n_rendered": None, "notes": [], "skipped": True}
    src_dir = out_dir / "sources" / f"s{source_id}"
    src_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    n_rendered = 0
    bars_html = ""
    grid_html = ""
    table_html = ""
    try:
        label_counts = df["label"].value_counts()
        palette = _palette(len(label_counts))
        colors = dict(zip(label_counts.index, palette, strict=True))

        genus_counts = df["genus"].value_counts()
        coral_counts = df["coral_genus"].value_counts()
        for name, counts, title, cmap in (
            ("label_hist", label_counts, "Source labels", colors),
            ("genus_hist", genus_counts, "Genera", None),
            ("coral_genus_hist", coral_counts, "Coral genera", None),
        ):
            if _bar_chart(counts, title, src_dir / f"{name}.png", cmap):
                bars_html += f"<img src='s{source_id}/{name}.png' alt='{title}'><br>"
            else:
                bars_html += f"<p>{title}: none in this source.</p>"
        table_html = _source_table(df)

        if not skip_images:
            from mermaidseg.datasets.utils import get_image_s3

            images = (
                df.dropna(subset=["image_key"])
                .drop_duplicates("image_id")[["image_id", "image_key"]]
                .reset_index(drop=True)
            )
            rng = np.random.default_rng([seed, int(source_id)])
            order = rng.permutation(len(images))
            by_image = dict(tuple(df.groupby("image_id", observed=True, sort=False)))
            figures: list[str] = []
            failures = 0
            for i in order:
                if n_rendered >= n_images or failures >= max(5, n_images):
                    break
                image_id, image_key = images.iloc[i]
                try:
                    image = get_image_s3(s3=None, bucket=bucket, key=str(image_key)).convert("RGB")
                    _render_overlay(
                        image,
                        by_image[image_id],
                        colors,
                        max_edge,
                        src_dir / f"img_{n_rendered:02d}.jpg",
                    )
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.warning("s%s image %s failed: %s", source_id, image_id, exc)
                    continue
                n_points = len(by_image[image_id])
                figures.append(
                    f"<figure><a href='s{source_id}/img_{n_rendered:02d}.jpg' target='_blank'>"
                    f"<img loading='lazy' src='s{source_id}/img_{n_rendered:02d}.jpg'></a>"
                    f"<figcaption>image {html.escape(str(image_id))} &middot; "
                    f"{n_points} points</figcaption></figure>"
                )
                n_rendered += 1
            if failures:
                notes.append(f"{failures} image(s) failed to load or render.")
            if not len(images):
                notes.append("No image in the manifest has an image key.")
            grid_html = f"<div class='grid'>{''.join(figures)}</div>"
    except Exception:  # noqa: BLE001
        notes.append("Rendering failed; see job log.")
        logger.error("s%s failed:\n%s", source_id, traceback.format_exc())

    stat_boxes = "".join(
        f"<div class='stat'>{html.escape(m.title)}<b>{m.fmt.format(float(stats_row[m.key]))}</b>"
        "</div>"
        for m in (*METRICS, *EXTRA_COLUMNS)
    )
    notes_html = "".join(f"<p class='note'>{html.escape(n)}</p>" for n in notes)
    images_section = (
        ""
        if skip_images
        else f"<h2>{n_rendered} random images</h2>"
        "<p>Each point is one annotation, colored by source label. Colors match the "
        f"source label chart above.</p>{grid_html}"
    )
    nav = _source_nav(prev_id, next_id)
    body = (
        f"{nav}<h1>CoralNet source s{source_id}</h1>"
        f"<p>Split: <b>{html.escape(str(stats_row['split']))}</b> &middot; "
        f"<a href='{stats_row['coralnet_url']}' target='_blank'>{stats_row['coralnet_url']}</a>"
        f"</p><div class='stats'>{stat_boxes}</div>{notes_html}"
        f"<h2>Distributions within this source</h2><div class='bars'>{bars_html}</div>"
        f"{table_html}{images_section}{nav}"
    )
    page.write_text(_page(f"CoralNet s{source_id}", body))
    return {"source_id": source_id, "n_rendered": n_rendered, "notes": notes, "skipped": False}


def _iter_sources(df: pd.DataFrame, order: list[int]) -> Iterator[tuple[int, pd.DataFrame]]:
    """Per-source slices with plain object columns, so workers never see global categories."""
    positions = df.groupby("source_id", observed=True).indices
    for sid in order:
        part = df.iloc[positions[sid]][WORKER_COLUMNS].reset_index(drop=True)
        for column in ("label", "image_key", "genus", "coral_genus"):
            part[column] = part[column].astype(object)
        yield sid, part


def _walk_order(stats: pd.DataFrame) -> list[int]:
    """Train, then val, then test, each in source-id order. Used for prev/next links."""
    rank = {split: i for i, split in enumerate(SPLITS)}
    return sorted(
        (int(sid) for sid in stats.index),
        key=lambda sid: (rank.get(str(stats.loc[sid, "split"]), len(SPLITS)), sid),
    )


def render_sources(
    df: pd.DataFrame, stats: pd.DataFrame, args: argparse.Namespace, cache_dir: str | None
) -> None:
    order = list(stats.sort_values("n_pixels", ascending=False).index)
    walk = _walk_order(stats)
    prev_of = {sid: walk[i - 1] if i else None for i, sid in enumerate(walk)}
    next_of = {sid: walk[i + 1] if i + 1 < len(walk) else None for i, sid in enumerate(walk)}
    records = dict(zip(stats.index, stats.to_dict("records"), strict=True))
    common = (
        args.output_dir,
        args.bucket,
        args.n_images,
        args.max_edge,
        args.seed,
        args.skip_images,
        args.resume,
    )
    progress = _Progress(len(order))
    if args.workers <= 1:
        _init_worker(cache_dir)
        for sid, part in _iter_sources(df, order):
            progress.record(
                render_source(
                    sid, part, records[sid], *common, prev_of[int(sid)], next_of[int(sid)]
                )
            )
        return

    max_in_flight = 2 * args.workers
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(cache_dir,),
    ) as pool:
        pending: set[Future] = set()
        for sid, part in _iter_sources(df, order):
            if len(pending) >= max_in_flight:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                progress.collect(finished)
            pending.add(
                pool.submit(
                    render_source,
                    sid,
                    part,
                    records[sid],
                    *common,
                    prev_of[int(sid)],
                    next_of[int(sid)],
                )
            )
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            progress.collect(finished)


class _Progress:
    def __init__(self, total: int, every: int = 25) -> None:
        self.total = total
        self.every = every
        self.done = 0

    def record(self, result: dict[str, Any]) -> None:
        for note in result["notes"]:
            logger.warning("s%s: %s", result["source_id"], note)
        self.done += 1
        if self.done % self.every == 0 or self.done == self.total:
            logger.info("rendered %d / %d sources", self.done, self.total)

    def collect(self, finished: set[Future]) -> None:
        for fut in finished:
            try:
                self.record(fut.result())
            except Exception:  # noqa: BLE001
                logger.error("worker crashed:\n%s", traceback.format_exc())
                self.done += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from mermaidseg.datasets.coralnet.coralnet_dataset import _resolve_default_manifest_path

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest",
        default=_resolve_default_manifest_path(),
        help="Manifest key in --bucket, s3:// URI, or local parquet path.",
    )
    parser.add_argument("--bucket", default=os.getenv("MERMAID_CORALNET_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--concepts-csv", type=Path, default=DEFAULT_CONCEPTS_CSV)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local S3 cache root (default: $MERMAIDSEG_LOCAL_CACHE_DIR or the data config).",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sources", type=int, nargs="*", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=None,
        help="Only sources whitelisted in these data-config splits. Default: the whole manifest.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Write stats, histograms and per-source bar charts, but no image overlays.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a source when its HTML page already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    import matplotlib

    matplotlib.use("Agg")

    cache_dir = resolve_cache_dir(args.data_config, args.cache_dir)
    if cache_dir and not Path(cache_dir).is_dir():
        logger.info("cache dir %s does not exist; reading from S3 only", cache_dir)
        cache_dir = None
    configure_cache(cache_dir)

    splits = load_splits(args.data_config)
    sources = select_sources(splits, only_splits=args.splits, only_sources=args.sources)
    if sources is not None and not sources:
        raise SystemExit("no CoralNet sources selected")
    if sources is not None:
        logger.info("selected %d sources", len(sources))
    df = load_annotations(args.bucket, args.manifest, args.concepts_csv, sources)
    if df.empty:
        raise SystemExit("no annotations for the requested sources")
    present = set(df["source_id"].astype(int).unique())
    if sources is not None:
        missing = [sid for sid in sources if sid not in present]
        if missing:
            logger.warning("%d selected sources have no manifest rows", len(missing))
    stats = compute_source_stats(df, splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output_dir / "source_stats.csv")
    render_histograms(stats, args.output_dir)
    render_index(stats, args.output_dir, args.manifest)
    (args.output_dir / "run.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
    )
    logger.info("wrote index for %d sources to %s", len(stats), args.output_dir)

    render_sources(df, stats, args, cache_dir)
    logger.info("done: open %s", args.output_dir / "index.html")


if __name__ == "__main__":
    main()
