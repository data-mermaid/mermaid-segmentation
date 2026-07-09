"""Palettes, overlays, HTML panels, and taxonomy tree rendering."""

from __future__ import annotations

import colorsys
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from mermaidseg.dataset_reconciliation.concepts import (
    MORPHOLOGIC_CONCEPTS,
    NONCORAL_CONCEPTS,
    TAXONOMIC_CONCEPTS,
    parse_concept_rank,
)

RANK_ORDER: tuple[str, ...] = tuple(TAXONOMIC_CONCEPTS)

DISPLAY_SIZE = 720

ONEHOT_MODE_LABELS: dict[str, str] = {
    "classes": "MERMAID Classification",
    **{rank: rank.capitalize() for rank in RANK_ORDER},
}

BACKGROUND_IDS: frozenset[int] = frozenset({0})
SAND_IDS: frozenset[int] = frozenset({58})
HARD_SUBSTRATE_IDS: frozenset[int] = frozenset({7, 55, 56, 57})
ALGAE_IDS: frozenset[int] = frozenset({9, 10, 12, 26, 34, 36, 47, 60, 67, 70})
SPONGE_IDS: frozenset[int] = frozenset({64})
CORAL_IDS: frozenset[int] = frozenset(
    {
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        35,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        61,
        62,
        63,
        65,
        66,
        68,
        69,
    }
)
SAND_RGB: tuple[int, int, int] = (237, 201, 145)
OTHER_FALLBACK_RGB: tuple[int, int, int] = (80, 60, 100)

_RANK_HUE_RANGES: dict[str, tuple[float, float]] = {
    "kingdom": (0, 360),
    "phylum": (30, 300),
    "class": (180, 540),
    "order": (60, 330),
    "family": (0, 720),
    "genus": (0, 1080),
}


def _hsv_ramp(
    n: int,
    hue_deg_range: tuple[float, float],
    s_range: tuple[float, float] = (0.55, 0.9),
    v_range: tuple[float, float] = (0.55, 0.95),
) -> NDArray[np.uint8]:
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    out = np.zeros((n, 3), dtype=np.uint8)
    h0, h1 = hue_deg_range
    s_lo, s_hi = s_range
    v_lo, v_hi = v_range
    for i in range(n):
        t = i / max(n - 1, 1)
        hue = ((h0 + (h1 - h0) * t) % 360) / 360.0
        sat = s_lo + (s_hi - s_lo) * (0.5 + 0.5 * np.sin(i * 1.7))
        val = v_lo + (v_hi - v_lo) * (0.5 + 0.5 * np.cos(i * 2.3))
        r, g, b = colorsys.hsv_to_rgb(hue, float(sat), float(val))
        out[i] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def _gray_ramp(n: int, v_range: tuple[int, int] = (80, 180)) -> NDArray[np.uint8]:
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = v_range
    values = np.linspace(lo, hi, n, dtype=np.int32)
    return np.stack([values, values, values], axis=1).astype(np.uint8)


def make_color_palette(num_classes: int) -> NDArray[np.uint8]:
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    def _assign(ids: frozenset[int], colors: NDArray[np.uint8]) -> None:
        present = sorted(i for i in ids if 0 <= i < num_classes)
        for idx, cid in enumerate(present):
            palette[cid] = colors[idx]

    _assign(HARD_SUBSTRATE_IDS, _gray_ramp(len(HARD_SUBSTRATE_IDS)))
    _assign(
        ALGAE_IDS,
        _hsv_ramp(
            len(ALGAE_IDS), hue_deg_range=(90, 135), s_range=(0.75, 0.95), v_range=(0.55, 0.9)
        ),
    )
    _assign(SPONGE_IDS, _hsv_ramp(len(SPONGE_IDS), hue_deg_range=(180, 215)))
    _assign(CORAL_IDS, _hsv_ramp(len(CORAL_IDS), hue_deg_range=(285, 390)))

    for sid in SAND_IDS:
        if 0 <= sid < num_classes:
            palette[sid] = SAND_RGB

    assigned = BACKGROUND_IDS | SAND_IDS | HARD_SUBSTRATE_IDS | ALGAE_IDS | SPONGE_IDS | CORAL_IDS
    for cid in range(num_classes):
        if cid not in assigned and cid not in BACKGROUND_IDS:
            palette[cid] = OTHER_FALLBACK_RGB
    for bid in BACKGROUND_IDS:
        if 0 <= bid < num_classes:
            palette[bid] = (0, 0, 0)
    return palette


def resize_for_display(image_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    pil = Image.fromarray(image_rgb).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def draw_click_marker(
    image_rgb: NDArray[np.uint8], xy: tuple[int, int] | None
) -> NDArray[np.uint8]:
    if xy is None:
        return image_rgb
    x, y = int(xy[0]), int(xy[1])
    pil = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = 8
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 255, 240), width=2
    )
    draw.ellipse(
        (x - 3, y - 3, x + 3, y + 3), fill=(255, 60, 60, 255), outline=(255, 255, 255, 240), width=1
    )
    return np.asarray(Image.alpha_composite(pil, overlay).convert("RGB"), dtype=np.uint8)


def build_rank_index(concept_names: list[str]) -> dict[str, list[tuple[int, str]]]:
    index: dict[str, list[tuple[int, str]]] = {rank: [] for rank in RANK_ORDER}
    for idx, name in enumerate(concept_names):
        rank, value = parse_concept_rank(name)
        if rank in index:
            index[rank].append((idx, value))
    return index


def find_concept_channel(concept_names: list[str], name: str) -> int | None:
    try:
        return concept_names.index(name)
    except ValueError:
        return None


@functools.lru_cache(maxsize=4)
def load_taxonomy_parents(csv_path: str | Path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    parents: dict[str, str] = {}
    placeholders = {"not_given", "none", "", None}
    for child_rank, parent_rank in zip(RANK_ORDER[1:], RANK_ORDER[:-1], strict=True):
        if child_rank not in df.columns or parent_rank not in df.columns:
            continue
        sub = df[[child_rank, parent_rank]].dropna()
        for child_val, parent_val in sub.itertuples(index=False, name=None):
            if child_val in placeholders or parent_val in placeholders:
                continue
            parents.setdefault(f"{child_rank}__{child_val}", f"{parent_rank}__{parent_val}")
    return parents


def make_rank_palette(values: list[str], rank: str) -> dict[str, NDArray[np.uint8]]:
    sorted_values = sorted(values)
    hue_range = _RANK_HUE_RANGES.get(rank, (0, 360))
    colors = _hsv_ramp(len(sorted_values), hue_deg_range=hue_range)
    return {value: colors[i] for i, value in enumerate(sorted_values)}


def _blend(
    display_rgb: NDArray[np.uint8],
    color_per_pixel: NDArray[np.uint8],
    alpha: NDArray[np.float32],
) -> NDArray[np.uint8]:
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blended = (
        display_rgb.astype(np.float32) * (1.0 - alpha) + color_per_pixel.astype(np.float32) * alpha
    )
    return blended.astype(np.uint8)


def compose_onehot_overlay(
    display_rgb: NDArray[np.uint8],
    class_probs: NDArray[np.float32] | None,
    concept_probs: NDArray[np.float32] | None,
    class_palette: NDArray[np.uint8],
    mode: str,
    rank_index: dict[str, list[tuple[int, str]]],
    rank_palettes: dict[str, dict[str, NDArray[np.uint8]]],
    opacity: float,
) -> NDArray[np.uint8]:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if mode == "classes":
        if class_probs is None:
            return display_rgb
        argmax = class_probs.argmax(axis=0)
        alpha = np.take_along_axis(class_probs, argmax[None, ...], axis=0)[0] * opacity
        return _blend(display_rgb, class_palette[argmax], alpha)

    if concept_probs is None or mode not in RANK_ORDER:
        return display_rgb
    entries = rank_index.get(mode, [])
    palette = rank_palettes.get(mode, {})
    if not entries or not palette:
        return display_rgb

    channel_idxs = np.asarray([idx for idx, _ in entries], dtype=np.int64)
    values = [val for _, val in entries]
    rank_probs = concept_probs[channel_idxs]
    argmax = rank_probs.argmax(axis=0)
    alpha = np.take_along_axis(rank_probs, argmax[None, ...], axis=0)[0] * opacity

    color_lut = np.zeros((len(values), 3), dtype=np.uint8)
    for i, value in enumerate(values):
        if value != "none" and value in palette:
            color_lut[i] = palette[value]
    return _blend(display_rgb, color_lut[argmax], alpha)


def compose_multihot_overlay(
    display_rgb: NDArray[np.uint8],
    concept_probs: NDArray[np.float32] | None,
    channel_idx: int,
    opacity: float,
    cmap: str = "viridis",
) -> NDArray[np.uint8]:
    if concept_probs is None:
        return display_rgb
    import matplotlib

    opacity = float(np.clip(opacity, 0.0, 1.0))
    prob = np.clip(concept_probs[channel_idx], 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](prob)
    color_per_pixel = (rgba[..., :3] * 255.0).astype(np.uint8)
    return _blend(display_rgb, color_per_pixel, prob * opacity)


def build_morph_concept_choices(concept_names: list[str]) -> list[str]:
    name_set = set(concept_names)
    return [name for name in (*MORPHOLOGIC_CONCEPTS, *NONCORAL_CONCEPTS) if name in name_set]


def overlay_legend_items(
    class_probs: NDArray[np.float32] | None,
    concept_probs: NDArray[np.float32] | None,
    mode: str,
    rank_index: dict[str, list[tuple[int, str]]],
    rank_palettes: dict[str, dict[str, NDArray[np.uint8]]],
    class_palette: NDArray[np.uint8],
    id2label: dict[int, str],
    top_n: int = 12,
) -> list[tuple[str, tuple[int, int, int], float]]:
    """Categories present in the one-hot overlay's argmax, sorted by pixel cover.

    Mirrors the argmax that ``compose_onehot_overlay`` draws so the legend matches the overlay
    exactly. Returns ``(label, (r, g, b), coverage_fraction)`` triples.
    """
    items: list[tuple[str, tuple[int, int, int], float]] = []
    if mode == "classes":
        if class_probs is None:
            return []
        argmax = class_probs.argmax(axis=0)
        total = argmax.size
        ids, counts = np.unique(argmax, return_counts=True)
        for cid, cnt in zip(ids.tolist(), counts.tolist(), strict=True):
            r, g, b = class_palette[cid]
            label = id2label.get(int(cid), f"class_{cid}")
            items.append((label, (int(r), int(g), int(b)), cnt / total))
    else:
        entries = rank_index.get(mode, [])
        palette = rank_palettes.get(mode, {})
        if concept_probs is None or not entries or not palette:
            return []
        channel_idxs = np.asarray([idx for idx, _ in entries], dtype=np.int64)
        values = [val for _, val in entries]
        argmax = concept_probs[channel_idxs].argmax(axis=0)
        total = argmax.size
        ids, counts = np.unique(argmax, return_counts=True)
        for i, cnt in zip(ids.tolist(), counts.tolist(), strict=True):
            value = values[int(i)]
            if value == "none" or value not in palette:
                continue
            r, g, b = palette[value]
            items.append((value, (int(r), int(g), int(b)), cnt / total))
    items.sort(key=lambda t: t[2], reverse=True)
    return items[:top_n]


def render_multihot_legend(
    concept_name: str | None,
    title: str = "Overlay color key",
    cmap: str = "viridis",
    stops: int = 12,
) -> str:
    """Colorbar key for the multi-hot heatmap: low→high sigmoid activation.

    The bar samples the actual colormap so it matches the overlay; only the concept name changes
    between concepts (the 0→1 scale is constant).
    """
    import matplotlib

    colormap = matplotlib.colormaps[cmap]
    ramp = ", ".join(
        "rgb({},{},{})".format(*(int(c * 255) for c in colormap(i / (stops - 1))[:3]))
        for i in range(stops)
    )
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    name = concept_name or "concept"
    return (
        f'<div class="panel">{title_html}'
        f'<div class="hint" style="margin-bottom:4px">sigmoid activation for <b>{name}</b></div>'
        f'<div style="height:14px;border-radius:3px;border:1px solid rgba(128,128,128,0.4);'
        f'background:linear-gradient(to right, {ramp})"></div>'
        '<div style="display:flex;justify-content:space-between;font-size:11px;opacity:0.7;margin-top:2px">'
        "<span>0.0 (low)</span><span>1.0 (high)</span></div></div>"
    )


def render_overlay_legend(
    items: list[tuple[str, tuple[int, int, int], float]],
    title: str = "Overlay color key",
) -> str:
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    if not items:
        return (
            f'<div class="panel">{title_html}'
            '<div class="hint">Run segmentation to see the color key.</div></div>'
        )
    chips: list[str] = []
    for label, (r, g, b), _cover in items:
        chips.append(
            '<span style="display:inline-flex;align-items:center;margin:0 12px 6px 0">'
            f'<span style="width:14px;height:14px;border-radius:3px;background:rgb({r},{g},{b});'
            'display:inline-block;margin-right:6px;border:1px solid rgba(128,128,128,0.4)"></span>'
            f"{label}</span>"
        )
    return f'<div class="panel">{title_html}<div>{"".join(chips)}</div></div>'


def render_top_classes_html(
    items: list[tuple[str, float]],
    title: str = "Top classes at clicked pixel",
    colors: list[tuple[int, int, int]] | None = None,
) -> str:
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    if not items:
        return (
            f'<div class="panel">{title_html}'
            '<div class="hint">Click a pixel to see top classes.</div></div>'
        )
    spans: list[str] = []
    for i, (name, p) in enumerate(items):
        p_clamped = float(np.clip(p, 0.0, 1.0))
        size_px = 14.0 + 42.0 * p_clamped
        opacity_pct = 35 + int(65 * p_clamped)
        if colors is not None and i < len(colors):
            r, g, b = colors[i]
            prob_style = f"color:rgb({r},{g},{b});font-weight:600"
        else:
            prob_style = "opacity:0.7"
        spans.append(
            f'<span style="font-size:{size_px:.0f}px;opacity:{opacity_pct / 100:.2f};margin-right:18px">'
            f'{name} <small style="{prob_style}">{p_clamped:.2f}</small></span>'
        )
    return f'<div class="panel">{title_html}<div>{"".join(spans)}</div></div>'


def render_top_bottom_other_html(
    top_items: list[tuple[str, float]],
    bottom_items: list[tuple[str, float]],
    title: str = "Predicted Concepts: Other",
    empty_hint: str = "Click a pixel to see other predicted concepts.",
) -> str:
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    if not top_items and not bottom_items:
        return f'<div class="panel">{title_html}<div class="hint">{empty_hint}</div></div>'

    def _chip(name: str, p: float, *, bottom: bool = False) -> str:
        p_clamped = float(np.clip(p, 0.0, 1.0))
        if bottom:
            return (
                f'<span style="font-size:18px;color:#c0392b;font-weight:600;margin-right:16px">'
                f"{name} <small>{p_clamped:.2f}</small></span>"
            )
        size_px = 12.0 + 32.0 * p_clamped
        opacity_pct = 35 + int(65 * p_clamped)
        return (
            f'<span style="font-size:{size_px:.0f}px;opacity:{opacity_pct / 100:.2f};margin-right:16px">'
            f"{name} <small>{p_clamped:.2f}</small></span>"
        )

    rows: list[str] = []
    if top_items:
        rows.append(
            f"<div><strong>Top {len(top_items)}</strong> {''.join(_chip(n, p) for n, p in top_items)}</div>"
        )
    if bottom_items:
        rows.append(
            f'<div><strong style="color:#c0392b">Bottom {len(bottom_items)}</strong> '
            f"{''.join(_chip(n, p, bottom=True) for n, p in bottom_items)}</div>"
        )
    return f'<div class="panel">{title_html}{"".join(rows)}</div>'


def _rank_candidates(
    concept_probs_at_pixel: NDArray[np.float32],
    rank_index: dict[str, list[tuple[int, str]]],
    rank: str,
    top_k: int,
) -> list[tuple[str, float]]:
    entries = rank_index.get(rank, [])
    if not entries:
        return []
    idxs = np.asarray([idx for idx, _ in entries], dtype=np.int64)
    values = [val for _, val in entries]
    probs = concept_probs_at_pixel[idxs]
    order = np.argsort(probs)[::-1][:top_k]
    return [(values[int(j)], float(probs[int(j)])) for j in order if values[int(j)] != "none"]


def render_taxonomy_tree(
    concept_probs_at_pixel: NDArray[np.float32] | None,
    rank_index: dict[str, list[tuple[int, str]]],
    _parents: dict[str, str],
    top_k: int = 3,
    title: str = "Taxonomy at clicked pixel",
) -> str:
    """Vertical HTML readout: one row per rank, top candidates as fixed-size chips."""
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    if concept_probs_at_pixel is None or concept_probs_at_pixel.size == 0:
        return (
            f'<div class="panel taxonomy-panel">{title_html}'
            '<div class="hint">Click a pixel to see the taxonomy.</div></div>'
        )

    rows: list[str] = []
    for i, rank in enumerate(RANK_ORDER):
        candidates = _rank_candidates(concept_probs_at_pixel, rank_index, rank, top_k)
        if not candidates:
            continue
        primary, primary_p = candidates[0]
        primary_p = float(np.clip(primary_p, 0.0, 1.0))
        alt_html = ""
        if len(candidates) > 1:
            alt_chips = []
            for name, p in candidates[1:]:
                p_clamped = float(np.clip(p, 0.0, 1.0))
                alt_chips.append(
                    f'<span class="taxonomy-alt" style="opacity:{0.45 + 0.55 * p_clamped:.2f}">'
                    f"{name}</span>"
                )
            alt_html = f'<div class="taxonomy-alts">{"".join(alt_chips)}</div>'

        connector = ""
        if i > 0:
            connector = '<div class="taxonomy-connector" aria-hidden="true"></div>'

        rows.append(
            f"{connector}"
            '<div class="taxonomy-row">'
            f'<div class="taxonomy-rank">{rank}</div>'
            '<div class="taxonomy-candidates">'
            f'<div class="taxonomy-primary">{primary}</div>'
            f'<div class="taxonomy-bar" style="width:{primary_p * 100:.0f}%"></div>'
            f"{alt_html}"
            "</div></div>"
        )

    if not rows:
        return (
            f'<div class="panel taxonomy-panel">{title_html}'
            '<div class="hint">No taxonomy concepts available for this pixel.</div></div>'
        )

    return f'<div class="panel taxonomy-panel">{title_html}<div class="taxonomy-tree">{"".join(rows)}</div></div>'
