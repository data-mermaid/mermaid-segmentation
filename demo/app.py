"""MERMAID Concept Bottleneck demo — Gradio app entrypoint for `gradio deploy`."""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import spaces  # ZeroGPU: must be imported before torch initializes CUDA; no-op elsewhere
import torch
from inference import (
    DemoArtifacts,
    build_model,
    build_transforms,
    default_model_config,
    default_taxonomy_csv,
    load_artifacts,
    predict,
    preprocess,
    resolve_checkpoint,
)
from PIL import Image
from rendering import (
    DISPLAY_SIZE,
    ONEHOT_MODE_LABELS,
    build_morph_concept_choices,
    build_rank_index,
    build_shared_name_colors,
    class_accent_rgb,
    compose_multihot_overlay,
    compose_onehot_overlay,
    draw_click_marker,
    find_concept_channel,
    load_taxonomy_parents,
    make_color_palette,
    make_rank_palette,
    overlay_legend_items,
    rank_highlight_rgb,
    render_multihot_legend,
    render_overlay_legend,
    render_taxonomy_skipped,
    render_taxonomy_tree,
    render_top_bottom_other_html,
    render_top_classes_html,
    resize_for_display,
    top_class_skips_taxonomy,
)

from mermaidseg.dataset_reconciliation.concepts import (
    MORPHOLOGIC_CONCEPTS,
    parse_concept_rank,
)

TOP_K_CLASSES = 3
TOP_K_TREE = 3
TOP_K_OTHER = 5
BOTTOM_K_OTHER = 5

# Readout title for the Concept Bottleneck tab (top/bottom concept activations at a pixel).
CONCEPT_READOUT_TITLE = "Concept activations at clicked pixel"

ONEHOT_DROPDOWN_MODES: tuple[str, ...] = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "classes",
)

DEFAULT_MULTIHOT: tuple[str, ...] = (
    "massive",
    "plating",
    "brain",
    "branching",
    "tabular",
    "bleached",
    "algae",
    "background",
    "anthropogenic",
    "dark",
)

PRIMARY_BTN_LABEL = "Run segmentation"

CSS = """
/* Scoped to .gradio-container: Blocks(elem_id=...) does not reach the DOM in Gradio 6. */
.gradio-container .gap, .gradio-container .form { gap: 8px !important; }
.gradio-container .block { padding: 8px !important; }
.gradio-container .section-title { font-size: 1.1rem; font-weight: 700; margin: 0 0 6px 0; }
.gradio-container .hint { color: #888; font-style: italic; font-size: 12px; }
.gradio-container .panel { padding: 2px; }
/* Fixed dark header (not theme vars) so the white-filled logo reads in both themes. */
#mermaid-header { padding: 0 !important; }
#mermaid-header .mermaid-header-bar {
    display: flex; align-items: center; gap: 16px;
    background: #0d1117; color: #ffffff;
    padding: 14px 20px; border-radius: 10px;
}
#mermaid-header .mermaid-header-logo svg { width: 46px; height: 48px; display: block; flex: 0 0 auto; }
#mermaid-header .mermaid-header-title {
    font-size: 1.35rem; font-weight: 700; line-height: 1.25; color: #ffffff !important;
}
#mermaid-header .mermaid-header-title-short { display: none; }
@media (max-width: 640px) {
    #mermaid-header .mermaid-header-title-full { display: none; }
    #mermaid-header .mermaid-header-title-short { display: inline; }
}
#mermaid-header .mermaid-header-subtitle {
    margin-top: 2px; color: rgba(255, 255, 255, 0.85) !important; font-size: 0.95rem;
}
/* Gradio's base CSS colors <b> near-black; keep it readable on the dark bar. */
#mermaid-header .mermaid-header-subtitle b { color: #ffffff; }
#mermaid-segment-btn, #mermaid-segment-btn button {
    width: 100%; max-width: 340px; margin-left: auto; margin-right: auto;
}
#mermaid-cls-img img, #mermaid-tax-img img, #mermaid-growth-img img {
    aspect-ratio: 1 / 1 !important;
    max-height: 70vh !important;
    object-fit: contain !important;
}
/* Keep sample thumbs from stretching non-square reef photos. */
.gradio-container .gallery-item img,
.gradio-container .thumbnail-item img {
    object-fit: contain !important;
}
/* Footer logo strip on a light card so the dark-ink logos read in both themes. */
#mermaid-footer { padding: 0 !important; }
#mermaid-footer .mermaid-footer {
    margin-top: 8px; padding: 16px 20px; border-radius: 10px;
    background: #ffffff; border: 1px solid rgba(128, 128, 128, 0.2); text-align: center;
}
#mermaid-footer .mermaid-footer-label {
    font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #667085; margin-bottom: 12px;
}
#mermaid-footer .mermaid-footer-logos {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
    gap: 20px 32px;
}
/* Uniform bounding box + contain so every logo occupies ~the same footprint
   regardless of its native aspect ratio. */
#mermaid-footer .mermaid-footer-logo {
    width: 150px; height: 52px; object-fit: contain; opacity: 0.85;
}
#mermaid-footer .mermaid-footer-logo.logo-exeter {
    /* PNG content is left-heavy (562px wide, ~325px ink); pin visual center in the slot. */
    width: 185px; height: 64px; object-position: 29.3% center;
}
#mermaid-footer .mermaid-footer-logo.logo-epfl {
    width: 105px; height: 38px;
}
@media (max-width: 640px) {
    #mermaid-footer .mermaid-footer-logos { gap: 16px 24px; }
    #mermaid-footer .mermaid-footer-logo { width: 120px; height: 42px; }
    #mermaid-footer .mermaid-footer-logo.logo-exeter { width: 148px; height: 50px; }
    #mermaid-footer .mermaid-footer-logo.logo-epfl { width: 88px; height: 32px; }
}
/* Third tab label is shortened to "CBM" via RESPONSIVE_JS on narrow viewports. */
/* Taxonomy readout: vertical ladder with fixed typography (replaces matplotlib Plot). */
.gradio-container .taxonomy-panel { min-height: 220px; }
.gradio-container .taxonomy-tree { margin-top: 4px; }
.gradio-container .taxonomy-row {
    display: flex; align-items: flex-start; gap: 14px; padding: 6px 0;
}
.gradio-container .taxonomy-row-active {
    background: rgba(59, 130, 246, 0.12); border-radius: 8px;
    margin: 0 -8px; padding: 6px 8px;
    box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.35);
}
.gradio-container .taxonomy-row-active .taxonomy-rank { color: #2563eb; }
.gradio-container .taxonomy-row-active .taxonomy-bar { background: #2563eb; }
.gradio-container .taxonomy-rank {
    flex: 0 0 76px; font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: #667085; padding-top: 3px;
}
.gradio-container .taxonomy-candidates { flex: 1; min-width: 0; }
.gradio-container .taxonomy-primary {
    font-size: 15px; font-weight: 600; line-height: 1.25; color: inherit;
    word-break: break-word;
}
.gradio-container .taxonomy-bar {
    height: 5px; border-radius: 3px; background: #3b82f6; margin: 5px 0 6px 0;
    max-width: 100%;
}
.gradio-container .taxonomy-alts { display: flex; flex-wrap: wrap; gap: 6px; }
.gradio-container .taxonomy-alt {
    font-size: 12px; line-height: 1.3; padding: 3px 9px; border-radius: 999px;
    background: rgba(128, 128, 128, 0.14); color: inherit;
}
.gradio-container .taxonomy-connector {
    width: 2px; height: 10px; margin-left: 37px; background: rgba(128, 128, 128, 0.35);
}
.gradio-container .taxonomy-caption { margin-bottom: 8px; }
"""

RESPONSIVE_JS = """
(() => {
  const FULL = "Concept Bottleneck";
  const SHORT = "CBM";
  const mq = window.matchMedia("(max-width: 640px)");
  let pending = false;
  function relabelTabs() {
    pending = false;
    const narrow = mq.matches;
    document.querySelectorAll(".gradio-container button[role='tab']").forEach((btn) => {
      const current = btn.textContent.trim();
      if (!btn.dataset.cbmFull) {
        btn.dataset.cbmFull = current === SHORT ? FULL : current;
      }
      if (btn.dataset.cbmFull !== FULL) return;
      const next = narrow ? SHORT : FULL;
      if (current !== next) btn.textContent = next;
    });
  }
  function scheduleRelabel() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(relabelTabs);
  }
  relabelTabs();
  mq.addEventListener("change", relabelTabs);
  const root = document.querySelector(".gradio-container") || document.body;
  new MutationObserver(scheduleRelabel).observe(root, { childList: true, subtree: true });
  window.addEventListener("load", () => setTimeout(relabelTabs, 250));
})();
"""

logger = logging.getLogger(__name__)

_LOGO_SVG_PATH = Path(__file__).resolve().parent / "static" / "mermaid-logo.svg"


def _header_html() -> str:
    logo = _LOGO_SVG_PATH.read_text(encoding="utf-8") if _LOGO_SVG_PATH.is_file() else ""
    return (
        '<div class="mermaid-header-bar">'
        f'<div class="mermaid-header-logo" aria-hidden="true">{logo}</div>'
        "<div>"
        '<div class="mermaid-header-title">'
        '<span class="mermaid-header-title-full">Concept Bottleneck Demo</span>'
        '<span class="mermaid-header-title-short">CBM Demo</span>'
        "</div>"
        '<div class="mermaid-header-subtitle">'
        f"Upload an image or pick a sample, click <b>{PRIMARY_BTN_LABEL}</b>, "
        "then click any overlay pixel to inspect classes, taxonomy, and concepts.</div>"
        "</div></div>"
    )


# Partner/collaborator logos shown in the footer strip (file, alt text, optional CSS class).
_FOOTER_LOGOS: tuple[tuple[str, str, str], ...] = (
    ("logo_wcs.png", "Wildlife Conservation Society", ""),
    ("logo_exeter.png", "University of Exeter", "logo-exeter"),
    ("logo_queensland.png", "University of Queensland", ""),
    ("logo_mit.png", "Massachusetts Institute of Technology", ""),
    ("logo_epfl.png", "EPFL", "logo-epfl"),
    ("logo_sparkgeo.png", "Sparkgeo", ""),
)


def _footer_html() -> str:
    """Centered strip of collaborator logos, base64-embedded so it is self-contained.

    Sits on a light card (see CSS) so the dark-ink logos stay legible in both the light
    and dark Gradio themes.
    """
    # In demo/logos/ (outside static/) so the sample-image glob, which scans static/,
    # never surfaces these as selectable sample images.
    logos_dir = Path(__file__).resolve().parent / "logos"
    imgs: list[str] = []
    for filename, alt, css_class in _FOOTER_LOGOS:
        path = logos_dir / filename
        if not path.is_file():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        class_attr = (
            f' class="mermaid-footer-logo {css_class}"'
            if css_class
            else ' class="mermaid-footer-logo"'
        )
        imgs.append(
            f'<img{class_attr} src="data:image/png;base64,{b64}" alt="{alt}" title="{alt}">'
        )
    if not imgs:
        return ""
    return (
        '<div class="mermaid-footer">'
        '<div class="mermaid-footer-label">In collaboration with</div>'
        f'<div class="mermaid-footer-logos">{"".join(imgs)}</div>'
        "</div>"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", help="Path to model checkpoint (or set DEMO_CHECKPOINT)")
    p.add_argument("--model-config", help="Path to model config YAML (or set DEMO_MODEL_CONFIG)")
    p.add_argument("--id2label", help="Path to id2label.json (default: demo/id2label.json)")
    p.add_argument("--concept-id2name", help="Path to concept_id2name.json")
    p.add_argument(
        "--taxonomy-csv",
        help="Path to class_to_concepts CSV for taxonomy tree edges (or set DEMO_TAXONOMY_CSV)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p


def _resolve_taxonomy_csv(args: argparse.Namespace) -> str:
    if args.taxonomy_csv:
        return args.taxonomy_csv
    if os.environ.get("DEMO_TAXONOMY_CSV"):
        return os.environ["DEMO_TAXONOMY_CSV"]
    return default_taxonomy_csv()


def _load(args: argparse.Namespace) -> DemoArtifacts:
    # Checkpoint: explicit/local path if present, else downloaded from the HF model repo.
    checkpoint = resolve_checkpoint(args.checkpoint)
    # Model config: explicit, else env, else the bundled demo copy.
    model_config = (
        args.model_config or os.environ.get("DEMO_MODEL_CONFIG") or default_model_config()
    )
    return load_artifacts(
        checkpoint=checkpoint,
        model_config=model_config,
        id2label=args.id2label,
        concept_id2name=args.concept_id2name,
    )


def _input_size(artifacts: DemoArtifacts) -> tuple[int, int]:
    size = artifacts.model_cfg.get("input_size", [512, 512])
    return int(size[0]), int(size[1])


def build_ui(
    artifacts: DemoArtifacts,
    model: Any,
    device: torch.device,
    taxonomy_csv: str,
) -> gr.Blocks:
    model_transform, display_transform = build_transforms(_input_size(artifacts))
    num_classes = max(artifacts.id2label.keys()) + 1
    shared_name_colors = build_shared_name_colors(artifacts.id2label)
    class_palette = make_color_palette(
        num_classes, id2label=artifacts.id2label, shared_name_colors=shared_name_colors
    )

    num_concepts = model.concept_classifier.in_channels
    concept_names = [
        name for _, name in sorted(artifacts.concept_id2name.items(), key=lambda kv: int(kv[0]))
    ]
    if len(concept_names) < num_concepts:
        concept_names.extend(f"concept_{i}" for i in range(len(concept_names), num_concepts))

    rank_index = build_rank_index(concept_names)
    rank_palettes = {
        rank: make_rank_palette(
            [value for _, value in entries], rank, shared_name_colors=shared_name_colors
        )
        for rank, entries in rank_index.items()
    }

    multihot_choices = [n for n in DEFAULT_MULTIHOT if n in set(concept_names)]
    if not multihot_choices:
        multihot_choices = build_morph_concept_choices(concept_names) or ["(none)"]
    multihot_channel_by_name = {
        name: find_concept_channel(concept_names, name)
        for name in multihot_choices
        if name != "(none)"
    }

    # Split the multi-hot concepts into the two groups the Concept Bottleneck tab shows.
    morph_set = set(MORPHOLOGIC_CONCEPTS)
    growth_form_choices = [n for n in multihot_choices if n in morph_set and n != "(none)"]
    other_group_choices = [n for n in multihot_choices if n not in morph_set and n != "(none)"]
    default_trait = (growth_form_choices or other_group_choices or ["(none)"])[0]

    other_concept_channels = [
        (idx, name) for idx, name in enumerate(concept_names) if parse_concept_rank(name)[0] is None
    ]

    try:
        parents = load_taxonomy_parents(taxonomy_csv)
    except FileNotFoundError:
        logger.warning("Taxonomy CSV not found at %s; taxonomy edges disabled.", taxonomy_csv)
        parents = {}

    static_dir = Path(__file__).resolve().parent / "static"
    static_examples = (
        sorted(
            str(p)
            for p in static_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        if static_dir.is_dir()
        else []
    )

    # Taxonomy-tab "stepper": the taxonomic ranks only (the final class is its own tab).
    rank_choices = [
        (ONEHOT_MODE_LABELS[m], m)
        for m in ONEHOT_DROPDOWN_MODES
        if m in ONEHOT_MODE_LABELS and m != "classes"
    ]
    rank_values = [v for _, v in rank_choices]
    default_rank = (
        "genus" if "genus" in rank_values else (rank_values[-1] if rank_values else "genus")
    )

    def _compose_onehot_base(display_image, class_probs, concept_probs, mode, opacity):
        """Composite the one-hot overlay (no click marker) + its color-key legend.

        Split from marker drawing so a pixel click only redraws the marker on the cached
        base instead of recomposing the full argmax+blend over the big arrays.
        """
        composite = compose_onehot_overlay(
            display_image,
            class_probs,
            concept_probs,
            class_palette,
            mode,
            rank_index,
            rank_palettes,
            opacity,
        )
        legend = render_overlay_legend(
            overlay_legend_items(
                class_probs,
                concept_probs,
                mode,
                rank_index,
                rank_palettes,
                class_palette,
                artifacts.id2label,
            )
        )
        return resize_for_display(composite), legend

    def _compose_multihot_base(display_image, concept_probs, multihot_name, opacity):
        channel_idx = multihot_channel_by_name.get(multihot_name)
        composite = (
            compose_multihot_overlay(display_image, concept_probs, channel_idx, opacity)
            if channel_idx is not None and concept_probs is not None
            else display_image
        )
        return resize_for_display(composite)

    def _empty_taxonomy():
        return render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE)

    def _taxonomy_at(
        class_probs,
        concept_probs,
        x_src,
        y_src,
        *,
        highlight_rank: str | None = None,
        accent_from_class: bool = False,
    ):
        if class_probs is not None:
            skip, label = top_class_skips_taxonomy(class_probs[:, y_src, x_src], artifacts.id2label)
            if skip:
                return render_taxonomy_skipped(label)
        at_pixel = concept_probs[:, y_src, x_src]
        # Taxonomy tab: highlight the selected rank with that rank's overlay palette.
        # MERMAID / Concept tabs: accent with the MERMAID-class mask color (separate
        # palette from taxonomic ranks — do not mark a rank row as "selected").
        highlight_rgb = (
            rank_highlight_rgb(highlight_rank, at_pixel, rank_index, rank_palettes)
            if highlight_rank
            else None
        )
        accent_rgb = accent_label = None
        if accent_from_class and class_probs is not None:
            accent_rgb, accent_label = class_accent_rgb(
                class_probs[:, y_src, x_src], class_palette, artifacts.id2label
            )
        return render_taxonomy_tree(
            at_pixel,
            rank_index,
            parents,
            top_k=TOP_K_TREE,
            highlight_rank=highlight_rank,
            highlight_rgb=highlight_rgb,
            accent_rgb=accent_rgb,
            accent_label=accent_label,
        )

    def _multihot_legend(trait, concept_probs=None, pixel_prob=None):
        channel_idx = multihot_channel_by_name.get(trait) if trait else None
        return render_multihot_legend(trait, concept_probs, channel_idx, pixel_prob)

    def _empty_other():
        return render_top_bottom_other_html(
            [],
            [],
            title=CONCEPT_READOUT_TITLE,
            empty_hint="Click a pixel to see concept activations.",
        )

    def _coords(evt: gr.SelectData, probs):
        index = evt.index if evt.index and None not in evt.index[:2] else (-1, -1)
        x_disp, y_disp = int(index[0]), int(index[1])
        if not (0 <= x_disp < DISPLAY_SIZE and 0 <= y_disp < DISPLAY_SIZE):
            return None
        src_h, src_w = probs.shape[1], probs.shape[2]
        x_src = min(src_w - 1, max(0, x_disp * src_w // DISPLAY_SIZE))
        y_src = min(src_h - 1, max(0, y_disp * src_h // DISPLAY_SIZE))
        return (x_disp, y_disp), x_src, y_src

    def _top_classes_at(class_probs, x_src, y_src):
        at = class_probs[:, y_src, x_src]
        items, colors = [], []
        for i in np.argsort(at)[::-1][:TOP_K_CLASSES]:
            idx = int(i)
            items.append((artifacts.id2label.get(idx, f"class_{idx}"), float(at[idx])))
            r, g, b = class_palette[idx]
            colors.append((int(r), int(g), int(b)))
        return render_top_classes_html(items, colors=colors)

    def _other_at(concept_probs, x_src, y_src):
        if not other_concept_channels:
            return _empty_other()
        at = concept_probs[:, y_src, x_src]
        idxs = np.asarray([idx for idx, _ in other_concept_channels], dtype=np.int64)
        names = [name for _, name in other_concept_channels]
        probs = at[idxs]
        order = np.argsort(probs)
        top_k = min(TOP_K_OTHER, len(order))
        bot_k = min(BOTTOM_K_OTHER, len(order) - top_k)
        return render_top_bottom_other_html(
            [(names[int(j)], float(probs[int(j)])) for j in order[::-1][:top_k]],
            [(names[int(j)], float(probs[int(j)])) for j in order[:bot_k]],
            title=CONCEPT_READOUT_TITLE,
        )

    # Sized from measured calls on the Space: ~11s in-context + cold weight
    # streaming; smaller reservations rank higher in the ZeroGPU queue.
    @spaces.GPU(duration=30)
    def run_predict(image, tax_rank, growth_form, other_group, cls_op, tax_op, growth_op):
        trait = growth_form or other_group or default_trait
        empty_tree = _empty_taxonomy()
        if image is None:
            return (
                None,
                render_overlay_legend([]),
                render_top_classes_html([]),
                empty_tree,
                None,
                render_overlay_legend([]),
                empty_tree,
                None,
                _multihot_legend(trait),
                _empty_other(),
                empty_tree,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        start = time.perf_counter()
        image_tensor, display_image = preprocess(image, model_transform, display_transform)
        # predict() returns float16 prob maps already (cast on-device before .cpu()).
        class_probs, concept_probs, _ = predict(model, image_tensor.to(device))
        logger.info("predict wall time: %.1fs", time.perf_counter() - start)
        cls_base, cls_legend = _compose_onehot_base(
            display_image, class_probs, concept_probs, "classes", cls_op
        )
        tax_base, tax_legend = _compose_onehot_base(
            display_image, class_probs, concept_probs, tax_rank, tax_op
        )
        growth_base = _compose_multihot_base(display_image, concept_probs, trait, growth_op)
        return (
            cls_base,
            cls_legend,
            render_top_classes_html([]),
            empty_tree,
            tax_base,
            tax_legend,
            empty_tree,
            growth_base,
            _multihot_legend(trait, concept_probs),
            _empty_other(),
            empty_tree,
            display_image,
            class_probs,
            concept_probs,
            None,
            cls_base,
            tax_base,
            growth_base,
        )

    def _unpack_click(click_state):
        if click_state is None:
            return None, None, None
        click_xy, x_src, y_src = click_state
        return click_xy, x_src, y_src

    def _readouts_at(class_probs, concept_probs, x_src, y_src, trait, tax_rank=None):
        empty_tree = _empty_taxonomy()
        if concept_probs is None:
            cls_tree = tax_tree = growth_tree = empty_tree
        else:
            # Shared ladders, different accents: MERMAID-class color vs rank overlay color.
            cls_tree = _taxonomy_at(
                class_probs, concept_probs, x_src, y_src, accent_from_class=True
            )
            tax_tree = _taxonomy_at(
                class_probs, concept_probs, x_src, y_src, highlight_rank=tax_rank
            )
            growth_tree = cls_tree
        channel_idx = multihot_channel_by_name.get(trait) if trait else None
        pixel_prob = (
            float(concept_probs[channel_idx, y_src, x_src])
            if concept_probs is not None and channel_idx is not None
            else None
        )
        return (
            _top_classes_at(class_probs, x_src, y_src),
            cls_tree,
            tax_tree,
            growth_tree,
            _other_at(concept_probs, x_src, y_src) if concept_probs is not None else _empty_other(),
            _multihot_legend(trait, concept_probs, pixel_prob),
        )

    def _marked_overlays(cls_base, tax_base, growth_base, click_xy):
        return (
            draw_click_marker(cls_base, click_xy) if cls_base is not None else None,
            draw_click_marker(tax_base, click_xy) if tax_base is not None else None,
            draw_click_marker(growth_base, click_xy) if growth_base is not None else None,
        )

    def pixel_click(
        display_image,
        class_probs,
        concept_probs,
        cls_base,
        tax_base,
        growth_base,
        growth_form,
        other_group,
        click_state,
        tax_rank,
        evt: gr.SelectData,
    ):
        trait = growth_form or other_group or default_trait
        empty_tree = _empty_taxonomy()
        empty_readouts = (
            render_top_classes_html([]),
            empty_tree,
            empty_tree,
            empty_tree,
            _empty_other(),
            _multihot_legend(trait, concept_probs),
        )

        def _pack(
            cls_top, cls_tree, tax_tree, growth_tree, growth_other, growth_legend, overlays, state
        ):
            cls_img, tax_img, growth_img = overlays
            return (
                cls_img,
                cls_top,
                cls_tree,
                tax_img,
                tax_tree,
                growth_img,
                growth_other,
                growth_tree,
                growth_legend,
                state,
            )

        if display_image is None or class_probs is None or cls_base is None:
            return _pack(
                empty_readouts[0],
                empty_readouts[1],
                empty_readouts[2],
                empty_readouts[3],
                empty_readouts[4],
                empty_readouts[5],
                (cls_base, tax_base, growth_base),
                None,
            )

        hit = _coords(evt, class_probs)
        if hit is None:
            click_xy, x_src, y_src = _unpack_click(click_state)
            if click_xy is None:
                return _pack(
                    empty_readouts[0],
                    empty_readouts[1],
                    empty_readouts[2],
                    empty_readouts[3],
                    empty_readouts[4],
                    empty_readouts[5],
                    (cls_base, tax_base, growth_base),
                    None,
                )
            cls_top, cls_tree, tax_tree, growth_tree, growth_other, growth_legend = _readouts_at(
                class_probs, concept_probs, x_src, y_src, trait, tax_rank
            )
            return _pack(
                cls_top,
                cls_tree,
                tax_tree,
                growth_tree,
                growth_other,
                growth_legend,
                _marked_overlays(cls_base, tax_base, growth_base, click_xy),
                click_state,
            )

        click_xy, x_src, y_src = hit
        cls_top, cls_tree, tax_tree, growth_tree, growth_other, growth_legend = _readouts_at(
            class_probs, concept_probs, x_src, y_src, trait, tax_rank
        )
        return _pack(
            cls_top,
            cls_tree,
            tax_tree,
            growth_tree,
            growth_other,
            growth_legend,
            _marked_overlays(cls_base, tax_base, growth_base, click_xy),
            (click_xy, x_src, y_src),
        )

    def recompose_cls(display_image, class_probs, concept_probs, opacity, click_state):
        if display_image is None:
            return None, None, render_overlay_legend([])
        base, legend = _compose_onehot_base(
            display_image, class_probs, concept_probs, "classes", opacity
        )
        click_xy, _, _ = _unpack_click(click_state)
        return draw_click_marker(base, click_xy), base, legend

    def recompose_tax(display_image, class_probs, concept_probs, rank, opacity, click_state):
        if display_image is None:
            return None, None, render_overlay_legend([])
        base, legend = _compose_onehot_base(
            display_image, class_probs, concept_probs, rank, opacity
        )
        click_xy, _, _ = _unpack_click(click_state)
        return draw_click_marker(base, click_xy), base, legend

    def _growth_update(display_image, concept_probs, trait, opacity, click_state):
        if display_image is None or trait is None:
            return None, None, _multihot_legend(trait)
        base = _compose_multihot_base(display_image, concept_probs, trait, opacity)
        click_xy, x_src, y_src = _unpack_click(click_state)
        channel_idx = multihot_channel_by_name.get(trait) if trait else None
        pixel_prob = (
            float(concept_probs[channel_idx, y_src, x_src])
            if concept_probs is not None
            and channel_idx is not None
            and x_src is not None
            and y_src is not None
            else None
        )
        return (
            draw_click_marker(base, click_xy),
            base,
            _multihot_legend(trait, concept_probs, pixel_prob),
        )

    def refresh_taxonomy_highlight(class_probs, concept_probs, click_state, tax_rank):
        """Re-accent only the Taxonomy-tab tree when the overlay rank changes."""
        _, x_src, y_src = _unpack_click(click_state)
        if x_src is None or concept_probs is None:
            return _empty_taxonomy()
        return _taxonomy_at(class_probs, concept_probs, x_src, y_src, highlight_rank=tax_rank)

    def select_growth_form(display_image, concept_probs, trait, opacity, click_state):
        # Picking a growth form clears the "other groups" selection (single active trait).
        img, base, legend = _growth_update(
            display_image, concept_probs, trait, opacity, click_state
        )
        return img, base, legend, gr.update(value=None)

    def select_other_group(display_image, concept_probs, trait, opacity, click_state):
        img, base, legend = _growth_update(
            display_image, concept_probs, trait, opacity, click_state
        )
        return img, base, legend, gr.update(value=None)

    def recompose_growth(
        display_image, concept_probs, growth_form, other_group, opacity, click_state
    ):
        return _growth_update(
            display_image, concept_probs, growth_form or other_group, opacity, click_state
        )

    def pick_sample(evt: gr.SelectData):
        if not static_examples:
            return None
        idx = int(evt.index) if evt.index is not None else 0
        if 0 <= idx < len(static_examples):
            return np.array(Image.open(static_examples[idx]).convert("RGB"))
        return None

    with gr.Blocks(title="Concept Bottleneck Demo") as ui:
        gr.HTML(_header_html(), elem_id="mermaid-header")

        display_state = gr.State(None)
        class_probs_state = gr.State(None)
        concept_probs_state = gr.State(None)
        click_state = gr.State(None)
        # Cached pre-marker composites (720² uint8) so a click only redraws the marker.
        cls_base_state = gr.State(None)
        tax_base_state = gr.State(None)
        growth_base_state = gr.State(None)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260):
                input_img = gr.Image(
                    type="numpy", image_mode="RGB", label="Upload image", height=300
                )
            with gr.Column(scale=2, min_width=320):
                if static_examples:
                    sample_gallery = gr.Gallery(
                        value=[(p, Path(p).name) for p in static_examples],
                        label="Select sample image",
                        columns=4,
                        height=300,
                        allow_preview=False,
                    )
                else:
                    sample_gallery = None

        predict_btn = gr.Button(
            PRIMARY_BTN_LABEL, variant="primary", size="lg", elem_id="mermaid-segment-btn"
        )

        # One tab per view: MERMAID classes · Taxonomy · Concept Bottleneck.
        with gr.Tabs():
            with gr.Tab("MERMAID classes"), gr.Row():
                with gr.Column(scale=2, min_width=340):
                    cls_opacity = gr.Slider(0, 1, value=0.5, step=0.05, label="Overlay opacity")
                    cls_img = gr.Image(
                        type="numpy",
                        label="Predicted class — click a pixel",
                        interactive=False,
                        elem_id="mermaid-cls-img",
                    )
                    cls_legend = gr.HTML(render_overlay_legend([]))
                with gr.Column(scale=1, min_width=240):
                    cls_top = gr.HTML(render_top_classes_html([]))
                    cls_tree = gr.HTML(_empty_taxonomy())

            with gr.Tab("Taxonomy"), gr.Row():
                with gr.Column(scale=2, min_width=340):
                    tax_rank = gr.Radio(
                        choices=rank_choices, value=default_rank, label="Color by rank"
                    )
                    tax_opacity = gr.Slider(0, 1, value=0.5, step=0.05, label="Overlay opacity")
                    tax_img = gr.Image(
                        type="numpy",
                        label="Taxonomic rank — click a pixel",
                        interactive=False,
                        elem_id="mermaid-tax-img",
                    )
                    tax_legend = gr.HTML(render_overlay_legend([]))
                with gr.Column(scale=1, min_width=240):
                    tax_tree = gr.HTML(_empty_taxonomy(), label="Taxonomy at clicked pixel")

            with gr.Tab("Concept Bottleneck"), gr.Row():
                with gr.Column(scale=2, min_width=340):
                    gf_sel = gr.Radio(
                        choices=growth_form_choices,
                        value=default_trait if default_trait in growth_form_choices else None,
                        label="Growth forms",
                    )
                    other_sel = gr.Radio(
                        choices=other_group_choices,
                        value=default_trait if default_trait in other_group_choices else None,
                        label="Other groups",
                    )
                    growth_opacity = gr.Slider(0, 1, value=0.5, step=0.05, label="Heatmap opacity")
                    growth_img = gr.Image(
                        type="numpy",
                        label="Concept heatmap — click a pixel",
                        interactive=False,
                        elem_id="mermaid-growth-img",
                    )
                    growth_legend = gr.HTML(_multihot_legend(default_trait))
                with gr.Column(scale=1, min_width=240):
                    growth_other = gr.HTML(_empty_other())
                    growth_tree = gr.HTML(_empty_taxonomy())

        gr.HTML(_footer_html(), elem_id="mermaid-footer")

        predict_inputs = [
            input_img,
            tax_rank,
            gf_sel,
            other_sel,
            cls_opacity,
            tax_opacity,
            growth_opacity,
        ]
        predict_outputs = [
            cls_img,
            cls_legend,
            cls_top,
            cls_tree,
            tax_img,
            tax_legend,
            tax_tree,
            growth_img,
            growth_legend,
            growth_other,
            growth_tree,
            display_state,
            class_probs_state,
            concept_probs_state,
            click_state,
            cls_base_state,
            tax_base_state,
            growth_base_state,
        ]

        predict_btn.click(run_predict, inputs=predict_inputs, outputs=predict_outputs)

        # Opacity sliders fire on release (not every drag tick); the rank/trait radios
        # fire on user input only, so programmatically clearing the sibling radio does
        # not re-trigger and loop.
        cls_opacity.release(
            recompose_cls,
            [display_state, class_probs_state, concept_probs_state, cls_opacity, click_state],
            [cls_img, cls_base_state, cls_legend],
        )

        tax_inputs = [
            display_state,
            class_probs_state,
            concept_probs_state,
            tax_rank,
            tax_opacity,
            click_state,
        ]
        tax_outputs = [tax_img, tax_base_state, tax_legend]
        tax_rank.input(recompose_tax, tax_inputs, tax_outputs).then(
            refresh_taxonomy_highlight,
            [class_probs_state, concept_probs_state, click_state, tax_rank],
            [tax_tree],
        )
        tax_opacity.release(recompose_tax, tax_inputs, tax_outputs)

        gf_sel.input(
            select_growth_form,
            [display_state, concept_probs_state, gf_sel, growth_opacity, click_state],
            [growth_img, growth_base_state, growth_legend, other_sel],
        )
        other_sel.input(
            select_other_group,
            [display_state, concept_probs_state, other_sel, growth_opacity, click_state],
            [growth_img, growth_base_state, growth_legend, gf_sel],
        )
        growth_opacity.release(
            recompose_growth,
            [display_state, concept_probs_state, gf_sel, other_sel, growth_opacity, click_state],
            [growth_img, growth_base_state, growth_legend],
        )

        click_inputs = [
            display_state,
            class_probs_state,
            concept_probs_state,
            cls_base_state,
            tax_base_state,
            growth_base_state,
            gf_sel,
            other_sel,
            click_state,
            tax_rank,
        ]
        click_outputs = [
            cls_img,
            cls_top,
            cls_tree,
            tax_img,
            tax_tree,
            growth_img,
            growth_other,
            growth_tree,
            growth_legend,
            click_state,
        ]
        cls_img.select(pixel_click, click_inputs, click_outputs)
        tax_img.select(pixel_click, click_inputs, click_outputs)
        growth_img.select(pixel_click, click_inputs, click_outputs)
        if sample_gallery is not None:
            sample_gallery.select(pick_sample, inputs=None, outputs=input_img).then(
                run_predict,
                inputs=predict_inputs,
                outputs=predict_outputs,
            )

    return ui


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    artifacts = _load(args)
    taxonomy_csv = _resolve_taxonomy_csv(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Loading model on %s", device)
    model = build_model(artifacts, device)
    logger.info(
        "Loaded %s with %d classes and %d concepts",
        artifacts.model_cfg.get("name"),
        max(artifacts.id2label) + 1,
        model.concept_classifier.in_channels,
    )
    build_ui(artifacts, model, device, taxonomy_csv).launch(
        server_port=args.port, share=args.share, css=CSS, js=RESPONSIVE_JS
    )


if __name__ == "__main__":
    main(sys.argv[1:])
