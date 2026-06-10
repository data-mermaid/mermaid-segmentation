"""Gradio demo v2: dual one-hot / multi-hot overlay panels for a trained CBM.

Same CLI and artifact loading as :mod:`mermaidseg.demo.app`, but renders two
side-by-side overlays:

- **one-hot**: Classification or a taxonomic rank (argmax + softmax-weighted alpha)
- **multi-hot**: a morphologic or non-coral concept heatmap (colormap + sigmoid-weighted alpha)

Usage::

    uv run python -m mermaidseg.demo.demo_v2_app \\
        --checkpoint model_checkpoints/<run>/model_epoch9 \\
        --data-config configs/data_config.yaml \\
        --model-config configs/model_config_cbm.yaml \\
        --training-config configs/training_config_cbm.yaml \\
        --id2label /tmp/id2label.json \\
        --concept-id2name /tmp/concept_id2name.json

Requires the optional ``demo`` extra: ``uv sync --extra demo`` (installs gradio).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from mermaidseg.demo.app import (
    BOTTOM_K_OTHER,
    TOP_K_CLASSES,
    TOP_K_OTHER,
    TOP_K_TREE,
    TIGHT_CSS,
    _build_parser,
    _load_artifacts,
)
from mermaidseg.demo.inference import (
    DISPLAY_SIZE,
    DemoArtifacts,
    ONEHOT_MODE_LABELS,
    build_model,
    build_rank_index,
    build_transforms,
    compose_multihot_overlay,
    compose_onehot_overlay,
    draw_click_marker,
    find_concept_channel,
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

logger = logging.getLogger(__name__)

# One-hot dropdown choices (edit freely). Keys must be in ONEHOT_MODE_LABELS
# (``classes`` for MERMAID classification, or a taxonomic rank).
ONEHOT_DROPDOWN_MODES: tuple[str, ...] = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "classes",
)

# Multi-hot dropdown choices (edit freely). Only names present in the loaded
# model's concept head appear in the UI; missing names are logged at startup.
MULTIHOT_CONCEPTS: tuple[str, ...] = (#'oval',
 #'arborescent',
 #'encrusting',
 #'digitate',
 #'meandroid',
 'massive',
 'columnar',
 'free_living',
 #'fleshy',
 #'submassive',
 #'round',
 'plating',
 #'tubular',
 #'bushy',
 #'external_polyps',
 #'foliose',
 #'solitary',
 'brain',
 #'phaceloid',
 'branching',
 'tabular',
 #'corymbose',
 #'lobed_brain',
 #'cup_coral',
 #'dead',
 'bleached',
 'algae',
 'background',
 'anthropogenic',
 #'trash',
 'transect',
# 'macroalgae',
 'dark',
 'human',
# 'sand',
 #'hard_substrate'
 )

_OVERLAY_IMG_CSS = """
#mermaid-onehot-img, #mermaid-multihot-img {
    aspect-ratio: 1 / 1 !important;
    max-height: 70vh !important;
}
#mermaid-onehot-img > div,
#mermaid-onehot-img .image-container,
#mermaid-onehot-img .image-frame,
#mermaid-multihot-img > div,
#mermaid-multihot-img .image-container,
#mermaid-multihot-img .image-frame {
    aspect-ratio: 1 / 1 !important;
    height: 100% !important;
    max-height: 70vh !important;
}
#mermaid-onehot-img img,
#mermaid-multihot-img img {
    aspect-ratio: 1 / 1 !important;
    width: 100% !important;
    height: 100% !important;
    max-height: 70vh !important;
    object-fit: contain !important;
}
#mermaid-onehot-mode, #mermaid-multihot-mode {
    padding: 4px 6px 0 6px !important;
    margin: 0 0 4px 0 !important;
}
#mermaid-onehot-mode,
#mermaid-onehot-mode > div,
#mermaid-onehot-mode > div > div,
#mermaid-onehot-mode label,
#mermaid-onehot-mode .container,
#mermaid-onehot-mode .wrap,
#mermaid-onehot-mode .wrap-inner,
#mermaid-onehot-mode .form,
#mermaid-onehot-mode .block,
#mermaid-multihot-mode,
#mermaid-multihot-mode > div,
#mermaid-multihot-mode > div > div,
#mermaid-multihot-mode label,
#mermaid-multihot-mode .container,
#mermaid-multihot-mode .wrap,
#mermaid-multihot-mode .wrap-inner,
#mermaid-multihot-mode .form,
#mermaid-multihot-mode .block {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    gap: 0 !important;
    row-gap: 0 !important;
    min-height: 0 !important;
}
#mermaid-onehot-mode label > span,
#mermaid-onehot-mode .label-wrap > span:first-child,
#mermaid-multihot-mode label > span,
#mermaid-multihot-mode .label-wrap > span:first-child {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #222 !important;
    margin-bottom: 4px !important;
}
#mermaid-onehot-mode input,
#mermaid-multihot-mode input {
    font-size: 15px !important;
}
"""

V2_CSS = TIGHT_CSS + _OVERLAY_IMG_CSS


def _build_ui(
    artifacts: DemoArtifacts,
    model: Any,
    device: torch.device,
) -> gr.Blocks:
    model_transform, display_transform = build_transforms(artifacts.cfg)
    num_classes = max(artifacts.id2label.keys()) + 1
    class_palette = make_color_palette(num_classes)

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

    concept_name_set = set(concept_names)
    multihot_choices = [name for name in MULTIHOT_CONCEPTS if name in concept_name_set]
    missing_multihot = [name for name in MULTIHOT_CONCEPTS if name not in concept_name_set]
    if missing_multihot:
        logger.warning(
            "Multi-hot concepts not in model (skipped in dropdown): %s",
            ", ".join(missing_multihot),
        )
    if not multihot_choices:
        logger.warning("No multi-hot concepts found; multi-hot panel will be empty.")
        multihot_choices = ["(none)"]

    multihot_channel_by_name = {
        name: find_concept_channel(concept_names, name)
        for name in multihot_choices
        if name != "(none)"
    }

    other_concept_channels: list[tuple[int, str]] = [
        (idx, name)
        for idx, name in enumerate(concept_names)
        if parse_concept_rank(name)[0] is None
    ]

    try:
        parents = load_taxonomy_parents()
    except FileNotFoundError:
        logger.warning(
            "class_to_concepts.csv not found; taxonomy tree will draw nodes without edges."
        )
        parents = {}

    static_dir = Path(__file__).resolve().parent / "static"
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    static_examples: list[str] = []
    if static_dir.is_dir():
        static_examples = sorted(
            str(p) for p in static_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )

    onehot_dropdown_choices = [
        (ONEHOT_MODE_LABELS[mode], mode)
        for mode in ONEHOT_DROPDOWN_MODES
        if mode in ONEHOT_MODE_LABELS
    ]
    missing_onehot = [
        mode for mode in ONEHOT_DROPDOWN_MODES if mode not in ONEHOT_MODE_LABELS
    ]
    if missing_onehot:
        logger.warning(
            "One-hot modes not recognized (skipped in dropdown): %s",
            ", ".join(missing_onehot),
        )
    if not onehot_dropdown_choices:
        logger.warning("No one-hot modes configured; falling back to genus.")
        onehot_dropdown_choices = [(ONEHOT_MODE_LABELS["genus"], "genus")]
    onehot_mode_values = [mode for _, mode in onehot_dropdown_choices]
    default_onehot_mode = (
        "genus" if "genus" in onehot_mode_values else onehot_mode_values[0]
    )

    def _ranked_classes_at(
        class_probs: NDArray[np.float32], x: int, y: int
    ) -> tuple[list[tuple[str, float]], list[tuple[int, int, int]]]:
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

    def _build_other_panel(concept_at_pixel: NDArray[np.float32]) -> str:
        if not other_concept_channels:
            return render_top_bottom_other_html(
                [], [], title="Predicted Concepts: Other",
            )
        idxs = np.asarray([idx for idx, _ in other_concept_channels], dtype=np.int64)
        names = [name for _, name in other_concept_channels]
        probs = concept_at_pixel[idxs]
        order = np.argsort(probs)

        top_k = min(TOP_K_OTHER, len(order))
        bot_k = min(BOTTOM_K_OTHER, len(order) - top_k)
        top_idx = order[::-1][:top_k]
        bot_idx = order[:bot_k]
        top_items = [(names[int(j)], float(probs[int(j)])) for j in top_idx]
        bottom_items = [(names[int(j)], float(probs[int(j)])) for j in bot_idx]
        return render_top_bottom_other_html(
            top_items, bottom_items, title="Predicted Concepts: Other",
        )

    def _render_onehot(
        display_image: NDArray[np.uint8],
        class_probs: NDArray[np.float32] | None,
        concept_probs: NDArray[np.float32] | None,
        mode: str,
        opacity_value: float,
        click_xy: tuple[int, int] | None = None,
    ) -> NDArray[np.uint8]:
        composite = compose_onehot_overlay(
            display_image,
            class_probs,
            concept_probs,
            class_palette=class_palette,
            mode=mode,
            rank_index=rank_index,
            rank_palettes=rank_palettes,
            opacity=opacity_value,
        )
        resized = resize_for_display(composite)
        return draw_click_marker(resized, click_xy)

    def _render_multihot(
        display_image: NDArray[np.uint8],
        concept_probs: NDArray[np.float32] | None,
        multihot_name: str,
        opacity_value: float,
        click_xy: tuple[int, int] | None = None,
    ) -> NDArray[np.uint8]:
        channel_idx = multihot_channel_by_name.get(multihot_name)
        if channel_idx is None or concept_probs is None:
            composite = display_image
        else:
            composite = compose_multihot_overlay(
                display_image,
                concept_probs,
                channel_idx,
                opacity=opacity_value,
            )
        resized = resize_for_display(composite)
        return draw_click_marker(resized, click_xy)

    def _empty_other_panel() -> str:
        return render_top_bottom_other_html(
            [], [], title="Predicted Concepts: Other",
            empty_hint="Click a pixel to see other predicted concepts.",
        )

    top_classes_title = "Predicted MERMAID Classes"

    def run_predict(
        image: NDArray[np.uint8] | None,
        onehot_mode: str,
        onehot_opacity: float,
        multihot_name: str,
        multihot_opacity: float,
    ) -> tuple[
        NDArray[np.uint8] | None,
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
                None, None, None, None, None, None,
                render_top_classes_html([], title=top_classes_title),
                render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
                _empty_other_panel(),
                None,
            )
        image_tensor, display_image = preprocess(image, model_transform, display_transform)
        image_tensor = image_tensor.to(device)
        class_probs, concept_probs, pred_mask = predict(model, image_tensor)
        onehot_img = _render_onehot(
            display_image, class_probs, concept_probs, onehot_mode, onehot_opacity
        )
        multihot_img = _render_multihot(
            display_image, concept_probs, multihot_name, multihot_opacity
        )
        return (
            onehot_img,
            multihot_img,
            display_image,
            class_probs,
            concept_probs,
            pred_mask,
            render_top_classes_html([], title=top_classes_title),
            render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
            _empty_other_panel(),
            None,
        )

    def recompose_onehot(
        display_image: NDArray[np.uint8] | None,
        class_probs: NDArray[np.float32] | None,
        concept_probs: NDArray[np.float32] | None,
        mode: str,
        opacity_value: float,
        click_xy: tuple[int, int] | None,
    ) -> NDArray[np.uint8] | None:
        if display_image is None:
            return None
        return _render_onehot(
            display_image, class_probs, concept_probs, mode, opacity_value, click_xy=click_xy
        )

    def recompose_multihot(
        display_image: NDArray[np.uint8] | None,
        concept_probs: NDArray[np.float32] | None,
        multihot_name: str,
        opacity_value: float,
        click_xy: tuple[int, int] | None,
    ) -> NDArray[np.uint8] | None:
        if display_image is None:
            return None
        return _render_multihot(
            display_image, concept_probs, multihot_name, opacity_value, click_xy=click_xy
        )

    def on_click(
        display_image: NDArray[np.uint8] | None,
        class_probs: NDArray[np.float32] | None,
        concept_probs: NDArray[np.float32] | None,
        onehot_mode: str,
        onehot_opacity: float,
        multihot_name: str,
        multihot_opacity: float,
        evt: gr.SelectData,
    ) -> tuple[
        NDArray[np.uint8] | None,
        NDArray[np.uint8] | None,
        str,
        Any,
        str,
        tuple[int, int] | None,
    ]:
        if display_image is None or class_probs is None or concept_probs is None:
            return (
                None,
                None,
                render_top_classes_html([], title=top_classes_title),
                render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
                _empty_other_panel(),
                None,
            )
        x_disp, y_disp = int(evt.index[0]), int(evt.index[1])
        if not (0 <= x_disp < DISPLAY_SIZE and 0 <= y_disp < DISPLAY_SIZE):
            click_xy = None
            top_html = render_top_classes_html([], title=top_classes_title)
            tree_fig = render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE)
            other_html_value = _empty_other_panel()
        else:
            src_h, src_w = class_probs.shape[1], class_probs.shape[2]
            x_src = min(src_w - 1, max(0, x_disp * src_w // DISPLAY_SIZE))
            y_src = min(src_h - 1, max(0, y_disp * src_h // DISPLAY_SIZE))

            top_items, top_colors = _ranked_classes_at(class_probs, x_src, y_src)
            concept_at_pixel = concept_probs[:, y_src, x_src]
            tree_fig = render_taxonomy_tree(
                concept_at_pixel, rank_index, parents, top_k=TOP_K_TREE
            )
            other_html_value = _build_other_panel(concept_at_pixel)
            top_html = render_top_classes_html(
                top_items, title=top_classes_title, colors=top_colors
            )
            click_xy = (x_disp, y_disp)

        onehot_img = _render_onehot(
            display_image, class_probs, concept_probs,
            onehot_mode, onehot_opacity, click_xy=click_xy,
        )
        multihot_img = _render_multihot(
            display_image, concept_probs, multihot_name, multihot_opacity, click_xy=click_xy,
        )
        return onehot_img, multihot_img, top_html, tree_fig, other_html_value, click_xy

    def _pick_sample(evt: gr.SelectData) -> NDArray[np.uint8] | None:
        if not static_examples:
            return None
        idx = int(evt.index) if evt.index is not None else 0
        if idx < 0 or idx >= len(static_examples):
            return None
        return np.array(Image.open(static_examples[idx]).convert("RGB"))

    overlay_legend = (
        "**one-hot**: argmax class or taxonomic concept per pixel; overlay strength "
        "is the argmax softmax probability × the one-hot opacity slider. Taxonomic "
        "`none` is always black.\n\n"
        "**multi-hot**: sigmoid activation for one morphologic or non-coral concept, colored with "
        "a viridis colormap; strength = sigmoid × multi-hot opacity slider."
    )

    with gr.Blocks(
        title="MERMAID Concept Bottleneck Demo v2",
        elem_id="mermaid-demo",
    ) as ui:
        gr.Markdown(
            "# MERMAID Concept Bottleneck Demo v2\n"
            "Upload an image, click **Predict**, then click any overlay pixel to "
            "inspect top classes and taxonomy below."
        )

        display_state = gr.State(None)
        class_probs_state = gr.State(None)
        concept_probs_state = gr.State(None)
        pred_mask_state = gr.State(None)
        click_state = gr.State(None)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=220):
                input_img = gr.Image(
                    type="numpy",
                    image_mode="RGB",
                    label="Upload image",
                    height=200,
                )
                predict_btn = gr.Button("Predict", variant="primary", size="sm")
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

            with gr.Column(scale=2):
                onehot_mode = gr.Dropdown(
                    choices=onehot_dropdown_choices,
                    value=default_onehot_mode,
                    label="one-hot",
                    interactive=True,
                    elem_id="mermaid-onehot-mode",
                )
                onehot_opacity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="One-hot opacity",
                )
                onehot_img = gr.Image(
                    type="numpy",
                    label="one-hot (click a pixel)",
                    interactive=False,
                    elem_id="mermaid-onehot-img",
                )

            with gr.Column(scale=2):
                multihot_mode = gr.Dropdown(
                    choices=multihot_choices,
                    value=multihot_choices[0],
                    label="multi-hot",
                    interactive=True,
                    elem_id="mermaid-multihot-mode",
                )
                multihot_opacity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Multi-hot opacity",
                )
                multihot_img = gr.Image(
                    type="numpy",
                    label="multi-hot (click a pixel)",
                    interactive=False,
                    elem_id="mermaid-multihot-img",
                )

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

        recompose_onehot_inputs = [
            display_state, class_probs_state, concept_probs_state,
            onehot_mode, onehot_opacity, click_state,
        ]
        recompose_multihot_inputs = [
            display_state, concept_probs_state,
            multihot_mode, multihot_opacity, click_state,
        ]
        click_inputs = [
            display_state,
            class_probs_state,
            concept_probs_state,
            onehot_mode,
            onehot_opacity,
            multihot_mode,
            multihot_opacity,
        ]

        predict_btn.click(
            run_predict,
            inputs=[input_img, onehot_mode, onehot_opacity, multihot_mode, multihot_opacity],
            outputs=[
                onehot_img,
                multihot_img,
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
        onehot_mode.change(recompose_onehot, inputs=recompose_onehot_inputs, outputs=onehot_img)
        onehot_opacity.change(recompose_onehot, inputs=recompose_onehot_inputs, outputs=onehot_img)
        multihot_mode.change(recompose_multihot, inputs=recompose_multihot_inputs, outputs=multihot_img)
        multihot_opacity.change(
            recompose_multihot, inputs=recompose_multihot_inputs, outputs=multihot_img
        )
        onehot_img.select(
            on_click,
            inputs=click_inputs,
            outputs=[onehot_img, multihot_img, top_classes_html, taxonomy_plot, other_html, click_state],
        )
        multihot_img.select(
            on_click,
            inputs=click_inputs,
            outputs=[onehot_img, multihot_img, top_classes_html, taxonomy_plot, other_html, click_state],
        )
        if sample_gallery is not None:
            sample_gallery.select(_pick_sample, inputs=None, outputs=input_img)

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
    ui.launch(server_port=args.port, share=args.share, css=V2_CSS)


if __name__ == "__main__":
    main(sys.argv[1:])
