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

# MERMAID classes for which the taxonomy ladder is hidden (non-biological / non-target).
TAXONOMY_SKIP_CLASS_LABELS: frozenset[str] = frozenset(
    {
        "background",
        "sand",
        "bare substrate",
        "anthropogenic",
        "human",
        "dark",
    }
)

ONEHOT_MODE_LABELS: dict[str, str] = {
    "classes": "MERMAID Classification",
    **{rank: rank.capitalize() for rank in RANK_ORDER},
}

BACKGROUND_IDS: frozenset[int] = frozenset({0})
SAND_RGB: tuple[int, int, int] = (237, 201, 145)
OTHER_FALLBACK_RGB: tuple[int, int, int] = (80, 60, 100)
ANTHRO_RGB: tuple[int, int, int] = (140, 140, 150)
BG_RGB: tuple[int, int, int] = (0, 0, 0)

_RANK_HUE_RANGES: dict[str, tuple[float, float]] = {
    "kingdom": (0, 360),
    "phylum": (30, 300),
    "class": (180, 540),
    "order": (60, 330),
    "family": (0, 720),
    "genus": (0, 1080),
}

# Name-based semantic groups for MERMAID classes (ids in this file went stale vs id2label).
_SUBSTRATE_LABELS: frozenset[str] = frozenset(
    {"bare substrate", "rock", "rubble", "algae covered substrate"}
)
_ALGAE_LABELS: frozenset[str] = frozenset(
    {
        "crustose coralline algae",
        "cyanobacteria",
        "dictyota",
        "halimeda",
        "lobophora",
        "macroalgae",
        "padina",
        "sargassum",
        "seagrass",
        "turbinaria-algae",
        "turf algae",
    }
)
_SPONGE_LABELS: frozenset[str] = frozenset({"sponge"})
_ANTHRO_LABELS: frozenset[str] = frozenset(
    {"human", "human-made structure", "transect tools", "anthropogenic"}
)
_BG_LABELS: frozenset[str] = frozenset({"ignore", "background", "dark"})
_SAND_LABELS: frozenset[str] = frozenset({"sand"})
# Non-coral biota (new ``fish`` class must not share the scleractinian magenta ramp).
_FAUNA_LABELS: frozenset[str] = frozenset({"fish", "sea urchin", "tridacna giant clam"})
# Broad MERMAID groups kept out of the per-genus coral ramp.
_COARSE_LABELS: frozenset[str] = frozenset({"hard coral", "soft coral", "gorgonian", "zoanthid"})
# Optional taxonomy-rank values that should reuse a MERMAID class color (name ≠ class).
_TAXONOMY_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "fish": ("chordata",),
    "sponge": ("porifera",),
    "sea urchin": ("echinodermata",),
    "hard coral": ("scleractinia", "hexacorallia"),
    "soft coral": ("octocorallia",),
    "zoanthid": ("zoantharia",),
}


def _normalize_label(name: str) -> str:
    return name.strip().lower()


def _semantic_bucket(label: str) -> str:
    n = _normalize_label(label)
    if n in _BG_LABELS:
        return "bg"
    if n in _SAND_LABELS:
        return "sand"
    if n in _SUBSTRATE_LABELS:
        return "substrate"
    if n in _ALGAE_LABELS:
        return "algae"
    if n in _SPONGE_LABELS:
        return "sponge"
    if n in _ANTHRO_LABELS:
        return "anthro"
    if n in _FAUNA_LABELS:
        return "fauna"
    if n in _COARSE_LABELS:
        return "coarse"
    return "coral"


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


def build_shared_name_colors(id2label: dict[int, str]) -> dict[str, NDArray[np.uint8]]:
    """Assign RGB colors keyed by normalized label name for MERMAID ↔ taxonomy sharing.

    Exact name matches (``acropora`` class ↔ ``acropora`` genus) reuse the same RGB.
    Binomial MERMAID labels without a same-named genus-level class also register their
    first token (``agaricia agaricites`` → ``agaricia``) so the genus overlay matches.
    Multi-species owners of the same token (``orbicella annularis`` / ``orbicella
    faveolata``) still seed the genus when they share a semantic bucket; mixed buckets
    (``turbinaria-algae`` / ``turbinaria-coral``) do not.
    """
    by_bucket: dict[str, list[tuple[int, str]]] = {
        "bg": [],
        "sand": [],
        "substrate": [],
        "algae": [],
        "sponge": [],
        "anthro": [],
        "fauna": [],
        "coarse": [],
        "coral": [],
    }
    for cid, name in sorted(id2label.items()):
        by_bucket[_semantic_bucket(name)].append((cid, name))

    id_colors: dict[int, NDArray[np.uint8]] = {}
    for cid, _ in by_bucket["bg"]:
        id_colors[cid] = np.asarray(BG_RGB, dtype=np.uint8)
    for cid, _ in by_bucket["sand"]:
        id_colors[cid] = np.asarray(SAND_RGB, dtype=np.uint8)
    for cid, _ in by_bucket["anthro"]:
        id_colors[cid] = np.asarray(ANTHRO_RGB, dtype=np.uint8)

    def _paint(bucket: str, colors: NDArray[np.uint8]) -> None:
        entries = by_bucket[bucket]
        for i, (cid, _) in enumerate(entries):
            id_colors[cid] = (
                colors[i] if i < len(colors) else np.asarray(OTHER_FALLBACK_RGB, dtype=np.uint8)
            )

    _paint("substrate", _gray_ramp(len(by_bucket["substrate"])))
    _paint(
        "algae",
        _hsv_ramp(
            len(by_bucket["algae"]),
            hue_deg_range=(90, 135),
            s_range=(0.75, 0.95),
            v_range=(0.55, 0.9),
        ),
    )
    _paint("sponge", _hsv_ramp(len(by_bucket["sponge"]), hue_deg_range=(180, 215)))
    _paint(
        "fauna",
        _hsv_ramp(
            len(by_bucket["fauna"]),
            hue_deg_range=(195, 250),
            s_range=(0.7, 0.95),
            v_range=(0.65, 0.95),
        ),
    )
    _paint(
        "coarse",
        _hsv_ramp(
            len(by_bucket["coarse"]),
            hue_deg_range=(15, 55),
            s_range=(0.55, 0.85),
            v_range=(0.55, 0.9),
        ),
    )
    _paint("coral", _hsv_ramp(len(by_bucket["coral"]), hue_deg_range=(285, 390)))

    name_colors: dict[str, NDArray[np.uint8]] = {}
    exact_keys = {_normalize_label(name) for name in id2label.values()}
    # Count first-token candidates so we don't map conflicting splits
    # (e.g. turbinaria-algae + turbinaria-coral) onto one genus color.
    token_owners: dict[str, list[str]] = {}
    for name in id2label.values():
        key = _normalize_label(name)
        parts = key.replace("-", " ").split()
        if len(parts) >= 2 and parts[0] not in exact_keys:
            token_owners.setdefault(parts[0], []).append(key)

    for cid, name in sorted(id2label.items()):
        color = id_colors.get(cid, np.asarray(OTHER_FALLBACK_RGB, dtype=np.uint8))
        key = _normalize_label(name)
        name_colors[key] = color
        parts = key.replace("-", " ").split()
        if len(parts) >= 2 and parts[0] not in exact_keys:
            owners = token_owners.get(parts[0], [])
            owner_buckets = {_semantic_bucket(owner) for owner in owners}
            # Seed when owners agree on bucket (orbicella spp.); skip mixed (turbinaria-*).
            if len(owner_buckets) == 1:
                name_colors.setdefault(parts[0], color)
        for alias in _TAXONOMY_COLOR_ALIASES.get(key, ()):
            name_colors.setdefault(alias, color)
    return name_colors


def make_color_palette(
    num_classes: int,
    id2label: dict[int, str] | None = None,
    shared_name_colors: dict[str, NDArray[np.uint8]] | None = None,
) -> NDArray[np.uint8]:
    """Build the MERMAID-class RGB LUT.

    Prefer ``shared_name_colors`` (from ``build_shared_name_colors``) so class overlays
    stay aligned with taxonomy overlays for matching names. Falls back to a name-based
    rebuild when ``id2label`` is given; legacy ID-group assignment is no longer used.
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    if shared_name_colors is None and id2label is not None:
        shared_name_colors = build_shared_name_colors(id2label)
    if shared_name_colors is not None and id2label is not None:
        for cid in range(num_classes):
            name = id2label.get(cid)
            if name is None:
                palette[cid] = OTHER_FALLBACK_RGB
                continue
            color = shared_name_colors.get(_normalize_label(name))
            palette[cid] = color if color is not None else OTHER_FALLBACK_RGB
        return palette

    # Minimal fallback if called without labels (tests / legacy).
    for cid in range(num_classes):
        palette[cid] = BG_RGB if cid == 0 else OTHER_FALLBACK_RGB
    return palette


def resize_for_display(image_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    pil = Image.fromarray(image_rgb).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def draw_click_marker(
    image_rgb: NDArray[np.uint8],
    xy: tuple[int, int] | None,
) -> NDArray[np.uint8]:
    if xy is None:
        return image_rgb
    x, y = int(xy[0]), int(xy[1])
    pil = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for radius, width in ((11, 2), (6, 2), (2, 1)):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 255, 235),
            width=width,
        )
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 60, 60, 255))
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


def make_rank_palette(
    values: list[str],
    rank: str,
    shared_name_colors: dict[str, NDArray[np.uint8]] | None = None,
) -> dict[str, NDArray[np.uint8]]:
    """Build a per-taxon RGB map for one taxonomic rank.

    Values whose normalized name appear in ``shared_name_colors`` (MERMAID class names,
    plus first-token seeds for species-only classes) reuse that RGB so the Taxonomy
    overlay matches the MERMAID-class overlay. Remaining taxa keep the rank HSV ramp.
    Higher ranks (family/order/…) almost never share names with MERMAID classes, so they
    stay mostly independent by design.
    """
    sorted_values = sorted(set(values))
    hue_range = _RANK_HUE_RANGES.get(rank, (0, 360))
    shared = shared_name_colors or {}
    unmatched = [v for v in sorted_values if _normalize_label(v) not in shared]
    ramp = _hsv_ramp(len(unmatched), hue_deg_range=hue_range)
    ramp_by_value = {v: ramp[i] for i, v in enumerate(unmatched)}

    palette: dict[str, NDArray[np.uint8]] = {}
    for value in sorted_values:
        key = _normalize_label(value)
        if key in shared:
            palette[value] = shared[key]
        else:
            palette[value] = ramp_by_value[value]
    return palette


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

    Mirrors the argmax that ``compose_onehot_overlay`` draws so the legend matches the
    overlay exactly. Returns ``(label, (r, g, b), coverage_fraction)`` triples.
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
    concept_probs: NDArray[np.float32] | None = None,
    _channel_idx: int | None = None,
    pixel_prob: float | None = None,
    title: str = "Overlay color key",
    cmap: str = "viridis",
    stops: int = 12,
) -> str:
    """Color ramp for the selected multi-hot concept, with optional clicked-pixel
    readout."""
    import matplotlib

    colormap = matplotlib.colormaps[cmap]
    ramp = ", ".join(
        "rgb({},{},{})".format(*(int(c * 255) for c in colormap(i / (stops - 1))[:3]))
        for i in range(stops)
    )
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    name = concept_name or "concept"

    if concept_probs is None:
        extra_html = '<div class="hint">Run segmentation to see the overlay key.</div>'
    elif pixel_prob is not None:
        p = float(np.clip(pixel_prob, 0.0, 1.0))
        extra_html = f'<div class="hint">Clicked Pixel: <strong>{p:.2f}</strong></div>'
    else:
        extra_html = '<div class="hint">Click a pixel to read activation at that point.</div>'

    return (
        f'<div class="panel">{title_html}'
        f'<div class="hint" style="margin-bottom:4px"><b>{name}</b></div>'
        f"{extra_html}"
        f'<div style="height:14px;border-radius:3px;border:1px solid rgba(128,128,128,0.4);'
        f'background:linear-gradient(to right, {ramp});margin-top:6px"></div>'
        '<div style="display:flex;justify-content:space-between;font-size:11px;opacity:0.7;margin-top:2px">'
        "<span>unlikely</span><span>likely</span></div></div>"
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


def rank_highlight_rgb(
    highlight_rank: str | None,
    concept_probs_at_pixel: NDArray[np.float32],
    rank_index: dict[str, list[tuple[int, str]]],
    rank_palettes: dict[str, dict[str, NDArray[np.uint8]]],
) -> tuple[int, int, int] | None:
    """RGB for the top taxon at ``highlight_rank``, matching the taxonomy overlay
    key."""
    if not highlight_rank:
        return None
    candidates = _rank_candidates(concept_probs_at_pixel, rank_index, highlight_rank, top_k=1)
    if not candidates:
        return None
    primary, _ = candidates[0]
    color = rank_palettes.get(highlight_rank, {}).get(primary)
    if color is None:
        return None
    return int(color[0]), int(color[1]), int(color[2])


def class_accent_rgb(
    class_probs_at_pixel: NDArray[np.float32],
    class_palette: NDArray[np.uint8],
    id2label: dict[int, str],
) -> tuple[tuple[int, int, int], str]:
    """Top MERMAID-class color + label at a pixel (matches the MERMAID overlay key)."""
    idx = int(class_probs_at_pixel.argmax())
    color = class_palette[idx]
    label = id2label.get(idx, f"class_{idx}")
    return (int(color[0]), int(color[1]), int(color[2])), label


def top_class_skips_taxonomy(
    class_probs_at_pixel: NDArray[np.float32],
    id2label: dict[int, str],
) -> tuple[bool, str]:
    idx = int(class_probs_at_pixel.argmax())
    label = id2label.get(idx, f"class_{idx}")
    return label.strip().lower() in TAXONOMY_SKIP_CLASS_LABELS, label


def render_taxonomy_skipped(
    class_label: str,
    title: str = "Taxonomy at clicked pixel",
) -> str:
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    return (
        f'<div class="panel taxonomy-panel">{title_html}'
        f'<div class="hint">No taxonomy for <b>{class_label}</b>.</div></div>'
    )


def _swatch_html(r: int, g: int, b: int) -> str:
    return (
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
        f"background:rgb({r},{g},{b});margin-right:6px;vertical-align:middle;"
        f'border:1px solid rgba(128,128,128,0.35)"></span>'
    )


def render_taxonomy_tree(
    concept_probs_at_pixel: NDArray[np.float32] | None,
    rank_index: dict[str, list[tuple[int, str]]],
    _parents: dict[str, str],
    top_k: int = 3,
    title: str = "Taxonomy at clicked pixel",
    highlight_rank: str | None = None,
    highlight_rgb: tuple[int, int, int] | None = None,
    accent_rgb: tuple[int, int, int] | None = None,
    accent_label: str | None = None,
) -> str:
    """Vertical HTML readout: one row per rank, top candidates as fixed-size chips.

    ``highlight_rank`` / ``highlight_rgb`` mark the Taxonomy-tab overlay rank (same
    palette as that overlay). ``accent_rgb`` / ``accent_label`` tint the panel to the
    MERMAID-class mask color without implying a taxonomic rank is selected.
    """
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    if concept_probs_at_pixel is None or concept_probs_at_pixel.size == 0:
        return (
            f'<div class="panel taxonomy-panel">{title_html}'
            '<div class="hint">Click a pixel to see the taxonomy.</div></div>'
        )

    caption = (
        '<div class="hint taxonomy-caption">'
        "Bar length = model confidence (0–1) for the top taxon at each rank.</div>"
    )
    if accent_rgb is not None and accent_label and highlight_rank is None:
        ar, ag, ab = accent_rgb
        caption = (
            f'<div class="hint taxonomy-caption">{_swatch_html(ar, ag, ab)}'
            f"Accent color matches MERMAID class <b>{accent_label}</b> "
            f"(not a taxonomic overlay).</div>"
        )

    panel_style = ""
    if accent_rgb is not None and highlight_rank is None:
        ar, ag, ab = accent_rgb
        panel_style = (
            f' style="border-left:4px solid rgb({ar},{ag},{ab});padding-left:10px;margin-left:-2px"'
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

        active = rank == highlight_rank
        if active and highlight_rgb is not None:
            r, g, b = highlight_rgb
            swatch = _swatch_html(r, g, b)
            row_open = (
                f'<div class="taxonomy-row" style="background:rgba({r},{g},{b},0.14);'
                f"border-radius:8px;margin:0 -8px;padding:6px 8px;"
                f'box-shadow:inset 0 0 0 1px rgba({r},{g},{b},0.5)">'
            )
            rank_cell = (
                f'<div class="taxonomy-rank" style="color:rgb({r},{g},{b})">{swatch}{rank}</div>'
            )
            bar = (
                f'<div class="taxonomy-bar" style="width:{primary_p * 100:.0f}%;'
                f'background:rgb({r},{g},{b})"></div>'
            )
        elif active:
            row_open = '<div class="taxonomy-row taxonomy-row-active">'
            rank_cell = f'<div class="taxonomy-rank">{rank}</div>'
            bar = f'<div class="taxonomy-bar" style="width:{primary_p * 100:.0f}%"></div>'
        elif accent_rgb is not None and highlight_rank is None:
            r, g, b = accent_rgb
            row_open = '<div class="taxonomy-row">'
            rank_cell = f'<div class="taxonomy-rank">{rank}</div>'
            bar = (
                f'<div class="taxonomy-bar" style="width:{primary_p * 100:.0f}%;'
                f'background:rgb({r},{g},{b})"></div>'
            )
        else:
            row_open = '<div class="taxonomy-row">'
            rank_cell = f'<div class="taxonomy-rank">{rank}</div>'
            bar = f'<div class="taxonomy-bar" style="width:{primary_p * 100:.0f}%"></div>'

        rows.append(
            f"{connector}{row_open}"
            f"{rank_cell}"
            '<div class="taxonomy-candidates">'
            f'<div class="taxonomy-primary">{primary} <small>({primary_p:.2f})</small></div>'
            f"{bar}"
            f"{alt_html}"
            "</div></div>"
        )

    if not rows:
        return (
            f'<div class="panel taxonomy-panel"{panel_style}>{title_html}'
            '<div class="hint">No taxonomy concepts available for this pixel.</div></div>'
        )

    return (
        f'<div class="panel taxonomy-panel"{panel_style}>{title_html}{caption}'
        f'<div class="taxonomy-tree">{"".join(rows)}</div></div>'
    )
