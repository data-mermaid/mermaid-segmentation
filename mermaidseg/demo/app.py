"""Gradio demo for a trained concept-bottleneck segmentation model.

Usage::

    # New four-file config layout (data + model + training [+ logger])
    uv run python -m mermaidseg.demo.app \\
        --checkpoint model_checkpoints/<run>/model_epoch9 \\
        --data-config configs/data_config.yaml \\
        --model-config configs/model_config_cbm.yaml \\
        --training-config configs/training_config_cbm.yaml \\
        --logger-config configs/logger_config.yaml \\
        --id2label /tmp/id2label.json \\
        --concept-id2name /tmp/concept_id2name.json  # optional

    # Single merged config (e.g. ``config/config.json`` downloaded from MLflow)
    uv run python -m mermaidseg.demo.app \\
        --checkpoint model_checkpoints/<run>/model_epoch9 \\
        --config /tmp/config.json \\
        --id2label /tmp/id2label.json

    # MLflow run (auto-downloads best-model + config + metadata artifacts)
    uv run python -m mermaidseg.demo.app --mlflow-run-id <run_id>

Requires the optional ``demo`` extra: ``uv sync --extra demo`` (installs gradio).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from mermaidseg.demo.inference import (
    DISPLAY_SIZE,
    RANK_ORDER,
    DemoArtifacts,
    build_model,
    build_rank_index,
    build_transforms,
    compose_concept_overlay,
    draw_click_marker,
    find_concept_channel,
    load_artifacts_local,
    load_artifacts_mlflow,
    load_taxonomy_parents,
    make_color_palette,
    make_rank_palette,
    parse_concept_rank,
    predict,
    preprocess,
    render_taxonomy_tree,
    render_top_bottom_other_html,
    render_top_classes_html,
    resize_for_display,
)

TOP_K_CLASSES = 3
TOP_K_TREE = 3
# Number of "Other" (non-rank) concepts to show in each of the top / bottom
# rows of the "Predicted Concepts: Other" panel.
TOP_K_OTHER = 5
BOTTOM_K_OTHER = 5

# Compact CSS scoped to the demo Blocks (``elem_id="mermaid-demo"``). Kills
# most of Gradio 6's default block padding, shrinks the gap between rows,
# and tightens the header markdown. Passed to ``launch(css=...)`` because
# Gradio 6 moved ``css`` off the Blocks constructor.
TIGHT_CSS = """
#mermaid-demo .gap { gap: 6px !important; }
#mermaid-demo .form { gap: 6px !important; }
#mermaid-demo .block { padding: 6px !important; border-radius: 6px; }
#mermaid-demo .prose h1 { margin: 0 0 2px 0 !important; font-size: 18px; }
#mermaid-demo .prose p { margin: 0 !important; font-size: 12px; }
#mermaid-demo button { min-height: 32px !important; padding: 4px 10px !important; }
#mermaid-demo label span { font-size: 12px !important; }
#mermaid-demo .panel-section { padding: 0 !important; }
#mermaid-demo .plot-container { padding: 0 !important; }
/* Force the prediction image to be square (1:1) while still filling
   the available column width. ``object-fit: contain`` keeps the
   720x720 composite from being squashed if the container is narrower.
   ``max-height`` keeps the image from blowing past the viewport on tall
   monitors so the right-hand result panels stay visible at 100% zoom. */
#mermaid-output-img { aspect-ratio: 1 / 1 !important; max-height: 70vh !important; }
#mermaid-output-img > div,
#mermaid-output-img .image-container,
#mermaid-output-img .image-frame { aspect-ratio: 1 / 1 !important; height: 100% !important; max-height: 70vh !important; }
#mermaid-output-img img { aspect-ratio: 1 / 1 !important; width: 100% !important; height: 100% !important; max-height: 70vh !important; object-fit: contain !important; }

/* Shared section heading for the three result panels
   ("Predicted MERMAID Classes" / "Predicted Concepts: Taxonomy" /
   "Predicted Concepts: Other"). 1.5x larger than the prior 48px. */
#mermaid-demo .mermaid-section-title {
    font-size: 72px !important;
    font-weight: 700 !important;
    color: #222 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: block !important;
    line-height: 1.05 !important;
    letter-spacing: 0.01em !important;
}
#mermaid-demo .mermaid-section-title-block { margin: 0 !important; padding: 0 !important; }
#mermaid-demo .mermaid-section-title-block p,
#mermaid-demo .mermaid-section-title-block h1,
#mermaid-demo .mermaid-section-title-block h2,
#mermaid-demo .mermaid-section-title-block h3 {
    font-size: 72px !important;
    font-weight: 700 !important;
    color: #222 !important;
    line-height: 1.05 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Scorched-earth: kill all vertical air inside the right results
   column so "Predicted MERMAID Classes" and "Predicted Concepts: Other"
   stack with zero gap. ``.mermaid-results-col`` may carry ``.gap`` or
   ``.form`` classes itself; the .mermaid-result-panel HTML elements
   already use container=False, but Gradio still wraps each in a
   small <div>. ``justify-content: flex-start`` prevents the column
   from spreading its children vertically when the row (sized by the
   center image at 70vh) leaves slack height. */
#mermaid-demo .mermaid-results-col,
#mermaid-demo .mermaid-results-col.gap,
#mermaid-demo .mermaid-results-col.form {
    gap: 0 !important;
    row-gap: 0 !important;
    justify-content: flex-start !important;
    align-content: flex-start !important;
}
#mermaid-demo .mermaid-results-col > * {
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    flex: 0 0 auto !important;
    min-height: 0 !important;
}
#mermaid-demo .mermaid-result-panel,
#mermaid-demo .mermaid-result-panel > div,
#mermaid-demo .mermaid-result-panel > div > div,
#mermaid-demo .mermaid-result-panel .html-container,
#mermaid-demo .mermaid-result-panel .prose {
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    min-height: 0 !important;
}

/* Overlay-mode dropdown: prominent label, NO wasted vertical space
   below the input. Gradio adds bottom padding on .block plus
   .container/.wrap, and the wrapping flex column adds a `gap` between
   children -- we zero them all so the dropdown sits flush with the
   bottom of the box. */
#mermaid-overlay-mode {
    padding: 4px 6px 0 6px !important;
    margin: 0 0 4px 0 !important;
}
#mermaid-overlay-mode,
#mermaid-overlay-mode > div,
#mermaid-overlay-mode > div > div,
#mermaid-overlay-mode label,
#mermaid-overlay-mode .container,
#mermaid-overlay-mode .wrap,
#mermaid-overlay-mode .wrap-inner,
#mermaid-overlay-mode .form,
#mermaid-overlay-mode .block {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    gap: 0 !important;
    row-gap: 0 !important;
    min-height: 0 !important;
}
#mermaid-overlay-mode label > span,
#mermaid-overlay-mode .label-wrap > span:first-child {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #222 !important;
    margin-bottom: 4px !important;
}
#mermaid-overlay-mode input { font-size: 15px !important; }

/* Sample-image gallery: keep filename captions small but legible. */
#mermaid-sample-gallery .grid-wrap { max-height: 220px; }
#mermaid-sample-gallery .caption,
#mermaid-sample-gallery .caption-label,
#mermaid-sample-gallery figcaption {
    font-size: 11px !important;
    opacity: 0.85 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
"""

OVERLAY_MODES: tuple[str, ...] = ("classes", *RANK_ORDER, "algae", "bleached")

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mlflow-run-id", help="MLflow run id to download artifacts from")
    p.add_argument("--checkpoint", help="path to model checkpoint (torch.save'd dict)")

    cfg_group = p.add_argument_group(
        "config (choose one form)",
        "Either pass a single --config pointing at a merged YAML/JSON, OR pass the "
        "split files used by setup_config (data + model + training [+ logger]).",
    )
    cfg_group.add_argument(
        "--config",
        help="path to a single merged YAML/JSON (e.g. config.json downloaded from MLflow)",
    )
    cfg_group.add_argument("--data-config", help="path to data_config.yaml")
    cfg_group.add_argument("--model-config", help="path to model_config[_cbm].yaml")
    cfg_group.add_argument("--training-config", help="path to training_config[_cbm].yaml")
    cfg_group.add_argument("--logger-config", help="path to logger_config.yaml (optional)")

    p.add_argument("--id2label", help="path to id2label.json")
    p.add_argument(
        "--concept-id2name",
        help=(
            "path to concept_id2name.json (or id2concept.json) — optional; "
            "missing => 'concept_<i>' placeholders"
        ),
    )
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--port", type=int, default=7860, help="Gradio server port")
    p.add_argument("--share", action="store_true", help="enable Gradio public share link")
    return p


def _load_artifacts(args: argparse.Namespace) -> DemoArtifacts:
    split_provided = any(
        getattr(args, attr) is not None
        for attr in ("data_config", "model_config", "training_config", "logger_config")
    )
    local_provided = args.checkpoint is not None or args.id2label is not None or split_provided or args.config is not None

    if args.mlflow_run_id and local_provided:
        raise SystemExit("Pass either --mlflow-run-id OR the local config/checkpoint args, not both.")
    if args.mlflow_run_id:
        logger.info("Downloading artifacts from MLflow run %s", args.mlflow_run_id)
        return load_artifacts_mlflow(args.mlflow_run_id)

    if args.checkpoint is None or args.id2label is None:
        raise SystemExit(
            "Local mode requires --checkpoint and --id2label (plus either --config or "
            "--data-config/--model-config/--training-config[/--logger-config])."
        )
    try:
        return load_artifacts_local(
            checkpoint=args.checkpoint,
            id2label=args.id2label,
            config=args.config,
            data_config=args.data_config,
            model_config=args.model_config,
            training_config=args.training_config,
            logger_config=args.logger_config,
            concept_id2name=args.concept_id2name,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e


def _build_ui(
    artifacts: DemoArtifacts,
    model: Any,
    device: torch.device,
) -> gr.Blocks:
    model_transform, display_transform = build_transforms(artifacts.cfg)
    num_classes = max(artifacts.id2label.keys()) + 1
    class_palette = make_color_palette(num_classes)
    # Map model concept-channel index → display name. ``concept_id2name`` may
    # be 1-indexed (as logged by ``SourceLabelRegistry``); sort by key to align
    # with model output channels regardless of base. When the artifact is
    # missing (e.g. the run didn't call ``logger.log_reconciliation``), fall
    # back to ``concept_<i>`` placeholders sized to the model's concept-head
    # channel count.
    num_concepts = model.concept_classifier.in_channels
    if artifacts.concept_id2name:
        concept_names = [
            name for _, name in sorted(artifacts.concept_id2name.items(), key=lambda kv: int(kv[0]))
        ]
        if len(concept_names) < num_concepts:
            concept_names.extend(
                f"concept_{i}" for i in range(len(concept_names), num_concepts)
            )
    else:
        concept_names = [f"concept_{i}" for i in range(num_concepts)]

    rank_index = build_rank_index(concept_names)
    rank_palettes = {
        rank: make_rank_palette([value for _, value in entries], rank)
        for rank, entries in rank_index.items()
    }
    algae_idx = find_concept_channel(concept_names, "algae")
    bleached_idx = find_concept_channel(concept_names, "bleached")

    # "Other" concepts: every channel whose name has no ``<rank>__`` prefix
    # (so morphology/health, substrate, algae, bleached, etc.). These are the
    # candidates for the top-5 / bottom-5 rows in the "Predicted Concepts:
    # Other" panel. We pre-compute parallel ``(channel_idx, name)`` arrays so
    # the per-click ranking is just a slice + argsort.
    other_concept_channels: list[tuple[int, str]] = [
        (idx, name)
        for idx, name in enumerate(concept_names)
        if parse_concept_rank(name)[0] is None
    ]
    if not other_concept_channels:
        logger.warning(
            "No non-rank concepts found; 'Predicted Concepts: Other' panel will be empty."
        )

    try:
        parents = load_taxonomy_parents()
    except FileNotFoundError:
        logger.warning(
            "class_to_concepts.csv not found; taxonomy tree will draw nodes without edges."
        )
        parents = {}

    # Sample image gallery sourced from ``mermaidseg/demo/static/``. Empty list
    # if the folder is missing or contains no supported images, in which case
    # the Examples panel is simply not rendered.
    static_dir = Path(__file__).resolve().parent / "static"
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    static_examples: list[str] = []
    if static_dir.is_dir():
        static_examples = sorted(
            str(p) for p in static_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )

    def _ranked_classes_at(
        class_probs: NDArray[np.float32], x: int, y: int
    ) -> tuple[list[tuple[str, float]], list[tuple[int, int, int]]]:
        """Return parallel ``(items, colors)`` lists for the top-K classes at a pixel."""
        class_at_pixel = class_probs[:, y, x]
        top_indices = np.argsort(class_at_pixel)[::-1][:TOP_K_CLASSES]
        items: list[tuple[str, float]] = []
        colors: list[tuple[int, int, int]] = []
        for i in top_indices:
            idx = int(i)
            items.append(
                (
                    artifacts.id2label.get(idx, f"class_{idx}"),
                    float(class_at_pixel[idx]),
                )
            )
            r, g, b = class_palette[idx]
            colors.append((int(r), int(g), int(b)))
        return items, colors

    def _build_other_panel(
        concept_at_pixel: NDArray[np.float32],
    ) -> str:
        """Render top-K and bottom-K non-rank ("other") concepts at the click.

        Bottom-K is rendered in red as a "model strongly says NOT this"
        signal — useful for spotting suppressed concepts at a glance.
        Both rows are always shown (no gating on rank-concept activations).
        """
        if not other_concept_channels:
            return render_top_bottom_other_html(
                [], [], title="Predicted Concepts: Other",
            )
        idxs = np.asarray([idx for idx, _ in other_concept_channels], dtype=np.int64)
        names = [name for _, name in other_concept_channels]
        probs = concept_at_pixel[idxs]
        order = np.argsort(probs)  # ascending: [0] = lowest, [-1] = highest

        top_k = min(TOP_K_OTHER, len(order))
        bot_k = min(BOTTOM_K_OTHER, len(order) - top_k)
        top_idx = order[::-1][:top_k]
        bot_idx = order[:bot_k]
        top_items = [(names[int(j)], float(probs[int(j)])) for j in top_idx]
        bottom_items = [(names[int(j)], float(probs[int(j)])) for j in bot_idx]
        return render_top_bottom_other_html(
            top_items, bottom_items, title="Predicted Concepts: Other",
        )

    def _render_overlay(
        display_image: NDArray[np.uint8],
        concept_probs: NDArray[np.float32] | None,
        pred_mask: NDArray[np.int64] | None,
        mode: str,
        opacity_value: float,
        click_xy: tuple[int, int] | None = None,
    ) -> NDArray[np.uint8]:
        composite = compose_concept_overlay(
            display_image,
            concept_probs,
            pred_mask,
            class_palette=class_palette,
            mode=mode,
            rank_index=rank_index,
            rank_palettes=rank_palettes,
            algae_idx=algae_idx,
            bleached_idx=bleached_idx,
            opacity=opacity_value,
        )
        resized = resize_for_display(composite)
        return draw_click_marker(resized, click_xy)

    def _empty_other_panel() -> str:
        return render_top_bottom_other_html(
            [], [], title="Predicted Concepts: Other",
            empty_hint="Click a pixel to see other predicted concepts.",
        )

    def run_predict(
        image: NDArray[np.uint8] | None,
        mode: str,
        opacity_value: float,
    ) -> tuple[
        NDArray[np.uint8] | None,
        NDArray[np.uint8] | None,
        NDArray[np.float32] | None,
        NDArray[np.float32] | None,
        NDArray[np.int64] | None,
        str,
        Any,
        str,
        tuple[int, int] | None,
    ]:
        if image is None:
            return (
                None, None, None, None, None,
                render_top_classes_html([], title=top_classes_title),
                render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
                _empty_other_panel(),
                None,
            )
        image_tensor, display_image = preprocess(image, model_transform, display_transform)
        image_tensor = image_tensor.to(device)
        class_probs, concept_probs, pred_mask = predict(model, image_tensor)
        composite = _render_overlay(
            display_image, concept_probs, pred_mask, mode, opacity_value, click_xy=None
        )
        return (
            composite,
            display_image,
            class_probs,
            concept_probs,
            pred_mask,
            render_top_classes_html([], title=top_classes_title),
            render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
            _empty_other_panel(),
            None,
        )

    def recompose(
        display_image: NDArray[np.uint8] | None,
        concept_probs: NDArray[np.float32] | None,
        pred_mask: NDArray[np.int64] | None,
        mode: str,
        opacity_value: float,
        click_xy: tuple[int, int] | None,
    ) -> NDArray[np.uint8] | None:
        if display_image is None:
            return None
        return _render_overlay(
            display_image, concept_probs, pred_mask, mode, opacity_value, click_xy=click_xy
        )

    def on_click(
        display_image: NDArray[np.uint8] | None,
        class_probs: NDArray[np.float32] | None,
        concept_probs: NDArray[np.float32] | None,
        pred_mask: NDArray[np.int64] | None,
        mode: str,
        opacity_value: float,
        evt: gr.SelectData,
    ) -> tuple[NDArray[np.uint8] | None, str, Any, str, tuple[int, int] | None]:
        if display_image is None or class_probs is None or concept_probs is None:
            return (
                None,
                render_top_classes_html([], title=top_classes_title),
                render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
                _empty_other_panel(),
                None,
            )
        x_disp, y_disp = int(evt.index[0]), int(evt.index[1])
        if not (0 <= x_disp < DISPLAY_SIZE and 0 <= y_disp < DISPLAY_SIZE):
            composite = _render_overlay(
                display_image, concept_probs, pred_mask, mode, opacity_value, click_xy=None
            )
            return (
                composite,
                render_top_classes_html([], title=top_classes_title),
                render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
                _empty_other_panel(),
                None,
            )

        # Map display-space click to source-image-space for the probability lookup.
        src_h, src_w = class_probs.shape[1], class_probs.shape[2]
        x_src = min(src_w - 1, max(0, x_disp * src_w // DISPLAY_SIZE))
        y_src = min(src_h - 1, max(0, y_disp * src_h // DISPLAY_SIZE))

        top_items, top_colors = _ranked_classes_at(class_probs, x_src, y_src)
        concept_at_pixel = concept_probs[:, y_src, x_src]
        tree_fig = render_taxonomy_tree(
            concept_at_pixel, rank_index, parents, top_k=TOP_K_TREE
        )
        other_html_value = _build_other_panel(concept_at_pixel)
        composite = _render_overlay(
            display_image, concept_probs, pred_mask, mode, opacity_value,
            click_xy=(x_disp, y_disp),
        )
        return (
            composite,
            render_top_classes_html(top_items, title=top_classes_title, colors=top_colors),
            tree_fig,
            other_html_value,
            (x_disp, y_disp),
        )

    overlay_legend = (
        "`classes` colors each pixel by its predicted segmentation class. The "
        "taxonomy modes (`kingdom`, `phylum`, `class`, `order`, `family`, `genus`) "
        "color each pixel by the rank's argmax concept with per-pixel alpha = that "
        "concept's sigmoid. `algae` (green) and `bleached` (white) are single-concept "
        "heatmaps."
    )

    top_classes_title = "Predicted MERMAID Classes"

    def _pick_sample(evt: gr.SelectData) -> NDArray[np.uint8] | None:
        """Map a thumbnail click to the underlying file → input numpy image.

        ``evt.index`` is the gallery index; ``static_examples`` is the
        parallel list of paths used to populate the gallery.
        """
        if not static_examples:
            return None
        idx = int(evt.index) if evt.index is not None else 0
        if idx < 0 or idx >= len(static_examples):
            return None
        return np.array(Image.open(static_examples[idx]).convert("RGB"))

    with gr.Blocks(
        title="MERMAID Concept Bottleneck Demo",
        elem_id="mermaid-demo",
    ) as ui:
        gr.Markdown(
            "# MERMAID Concept Bottleneck Demo\n"
            "Upload an image, pick an overlay mode, click **Predict**, then click any "
            "pixel to see its top classes and taxonomy below."
        )

        display_state = gr.State(None)
        class_probs_state = gr.State(None)
        concept_probs_state = gr.State(None)
        pred_mask_state = gr.State(None)
        click_state = gr.State(None)

        with gr.Row(equal_height=True):
            # ---- Left: upload + controls + sample gallery + legend ---------
            with gr.Column(scale=1, min_width=240):
                input_img = gr.Image(
                    type="numpy",
                    image_mode="RGB",
                    label="Upload image",
                    height=200,
                )
                predict_btn = gr.Button("Predict", variant="primary", size="sm")
                opacity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Opacity",
                )
                if static_examples:
                    sample_gallery = gr.Gallery(
                        value=[(p, Path(p).name) for p in static_examples],
                        label="Sample images (click to load)",
                        show_label=True,
                        columns=3,
                        height="auto",
                        object_fit="cover",
                        allow_preview=False,
                        elem_id="mermaid-sample-gallery",
                    )
                else:
                    sample_gallery = None
                with gr.Accordion("Overlay legend", open=False):
                    gr.Markdown(overlay_legend)

            # ---- Middle: overlay selector + output image -------------------
            with gr.Column(scale=2):
                mode = gr.Dropdown(
                    choices=list(OVERLAY_MODES),
                    value="classes",
                    label="Overlay Mode",
                    interactive=True,
                    elem_id="mermaid-overlay-mode",
                )
                output_img = gr.Image(
                    type="numpy",
                    label="Prediction (click a pixel)",
                    interactive=False,
                    elem_id="mermaid-output-img",
                )

            # ---- Right: predicted classes + other concepts -----------------
            # ``container=False`` strips Gradio's .block wrapper around each
            # gr.HTML so the two panels stack without padding/margin air
            # between them.
            with gr.Column(scale=2, elem_classes="mermaid-results-col"):
                top_classes_html = gr.HTML(
                    value=render_top_classes_html([], title=top_classes_title),
                    container=False,
                    show_label=False,
                    elem_classes="mermaid-result-panel",
                )
                other_html = gr.HTML(
                    value=_empty_other_panel(),
                    container=False,
                    show_label=False,
                    elem_classes="mermaid-result-panel",
                )

        # ---- Full-width row: taxonomy heading + plot -----------------------
        gr.Markdown(
            "Predicted Concepts: Taxonomy",
            elem_classes="mermaid-section-title-block",
        )
        taxonomy_plot = gr.Plot(
            value=render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
            label=None,
            show_label=False,
            container=False,
        )

        predict_btn.click(
            run_predict,
            inputs=[input_img, mode, opacity],
            outputs=[
                output_img,
                display_state,
                class_probs_state,
                concept_probs_state,
                pred_mask_state,
                top_classes_html,
                taxonomy_plot,
                other_html,
                click_state,
            ],
        )
        mode.change(
            recompose,
            inputs=[
                display_state, concept_probs_state, pred_mask_state,
                mode, opacity, click_state,
            ],
            outputs=output_img,
        )
        opacity.change(
            recompose,
            inputs=[
                display_state, concept_probs_state, pred_mask_state,
                mode, opacity, click_state,
            ],
            outputs=output_img,
        )
        output_img.select(
            on_click,
            inputs=[
                display_state,
                class_probs_state,
                concept_probs_state,
                pred_mask_state,
                mode,
                opacity,
            ],
            outputs=[output_img, top_classes_html, taxonomy_plot, other_html, click_state],
        )
        if sample_gallery is not None:
            sample_gallery.select(
                _pick_sample,
                inputs=None,
                outputs=input_img,
            )

    return ui


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    artifacts = _load_artifacts(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Loading model on %s", device)
    model = build_model(artifacts, device)
    logger.info(
        "Loaded model with %d classes and %d concepts",
        max(artifacts.id2label.keys()) + 1,
        model.concept_classifier.in_channels,
    )

    ui = _build_ui(artifacts, model, device)
    ui.launch(server_port=args.port, share=args.share, css=TIGHT_CSS)


if __name__ == "__main__":
    main(sys.argv[1:])
