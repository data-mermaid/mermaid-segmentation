"""MERMAID Concept Bottleneck demo — Gradio app entrypoint for `gradio deploy`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
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
from rendering import (
    DISPLAY_SIZE,
    ONEHOT_MODE_LABELS,
    build_morph_concept_choices,
    build_rank_index,
    compose_multihot_overlay,
    compose_onehot_overlay,
    draw_click_marker,
    find_concept_channel,
    load_taxonomy_parents,
    make_color_palette,
    make_rank_palette,
    render_taxonomy_tree,
    render_top_bottom_other_html,
    render_top_classes_html,
    resize_for_display,
)

from mermaidseg.dataset_reconciliation.concepts import parse_concept_rank

TOP_K_CLASSES = 3
TOP_K_TREE = 3
TOP_K_OTHER = 5
BOTTOM_K_OTHER = 5

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

CSS = """
#mermaid-demo .gap, #mermaid-demo .form { gap: 8px !important; }
#mermaid-demo .block { padding: 8px !important; }
#mermaid-demo .section-title { font-size: 1.1rem; font-weight: 700; margin: 0 0 6px 0; }
#mermaid-demo .hint { color: #888; font-style: italic; font-size: 12px; }
#mermaid-demo .panel { padding: 2px; }
#mermaid-onehot-img img, #mermaid-multihot-img img {
    aspect-ratio: 1 / 1 !important;
    max-height: 70vh !important;
    object-fit: contain !important;
}
"""

logger = logging.getLogger(__name__)


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
    class_palette = make_color_palette(num_classes)

    num_concepts = model.concept_classifier.in_channels
    concept_names = [
        name for _, name in sorted(artifacts.concept_id2name.items(), key=lambda kv: int(kv[0]))
    ]
    if len(concept_names) < num_concepts:
        concept_names.extend(f"concept_{i}" for i in range(len(concept_names), num_concepts))

    rank_index = build_rank_index(concept_names)
    rank_palettes = {
        rank: make_rank_palette([value for _, value in entries], rank)
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

    onehot_choices = [
        (ONEHOT_MODE_LABELS[m], m) for m in ONEHOT_DROPDOWN_MODES if m in ONEHOT_MODE_LABELS
    ]
    default_onehot = "genus" if "genus" in ONEHOT_MODE_LABELS else onehot_choices[0][1]

    def _render_onehot(display_image, class_probs, concept_probs, mode, opacity, click_xy=None):
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
        return draw_click_marker(resize_for_display(composite), click_xy)

    def _render_multihot(display_image, concept_probs, multihot_name, opacity, click_xy=None):
        channel_idx = multihot_channel_by_name.get(multihot_name)
        composite = (
            compose_multihot_overlay(display_image, concept_probs, channel_idx, opacity)
            if channel_idx is not None and concept_probs is not None
            else display_image
        )
        return draw_click_marker(resize_for_display(composite), click_xy)

    def _empty_other():
        return render_top_bottom_other_html([], [], title="Predicted Concepts: Other")

    def run_predict(image, onehot_mode, onehot_opacity, multihot_name, multihot_opacity):
        empty_tree = render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE)
        if image is None:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                render_top_classes_html([]),
                empty_tree,
                _empty_other(),
                None,
            )
        image_tensor, display_image = preprocess(image, model_transform, display_transform)
        class_probs, concept_probs, pred_mask = predict(model, image_tensor.to(device))
        return (
            _render_onehot(display_image, class_probs, concept_probs, onehot_mode, onehot_opacity),
            _render_multihot(display_image, concept_probs, multihot_name, multihot_opacity),
            display_image,
            class_probs,
            concept_probs,
            pred_mask,
            render_top_classes_html([]),
            empty_tree,
            _empty_other(),
            None,
        )

    def on_click(
        display_image,
        class_probs,
        concept_probs,
        onehot_mode,
        onehot_opacity,
        multihot_name,
        multihot_opacity,
        evt: gr.SelectData,
    ):
        if display_image is None or class_probs is None or concept_probs is None:
            return (
                None,
                None,
                render_top_classes_html([]),
                render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
                _empty_other(),
                None,
            )
        x_disp, y_disp = int(evt.index[0]), int(evt.index[1])
        if not (0 <= x_disp < DISPLAY_SIZE and 0 <= y_disp < DISPLAY_SIZE):
            click_xy = None
            top_html = render_top_classes_html([])
            tree_fig = render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE)
            other_html = _empty_other()
        else:
            src_h, src_w = class_probs.shape[1], class_probs.shape[2]
            x_src = min(src_w - 1, max(0, x_disp * src_w // DISPLAY_SIZE))
            y_src = min(src_h - 1, max(0, y_disp * src_h // DISPLAY_SIZE))
            class_at_pixel = class_probs[:, y_src, x_src]
            top_indices = np.argsort(class_at_pixel)[::-1][:TOP_K_CLASSES]
            top_items, top_colors = [], []
            for i in top_indices:
                idx = int(i)
                top_items.append(
                    (artifacts.id2label.get(idx, f"class_{idx}"), float(class_at_pixel[idx]))
                )
                r, g, b = class_palette[idx]
                top_colors.append((int(r), int(g), int(b)))
            concept_at_pixel = concept_probs[:, y_src, x_src]
            tree_fig = render_taxonomy_tree(concept_at_pixel, rank_index, parents, top_k=TOP_K_TREE)
            if other_concept_channels:
                idxs = np.asarray([idx for idx, _ in other_concept_channels], dtype=np.int64)
                names = [name for _, name in other_concept_channels]
                probs = concept_at_pixel[idxs]
                order = np.argsort(probs)
                top_k = min(TOP_K_OTHER, len(order))
                bot_k = min(BOTTOM_K_OTHER, len(order) - top_k)
                top_idx = order[::-1][:top_k]
                bot_idx = order[:bot_k]
                other_html = render_top_bottom_other_html(
                    [(names[int(j)], float(probs[int(j)])) for j in top_idx],
                    [(names[int(j)], float(probs[int(j)])) for j in bot_idx],
                )
            else:
                other_html = _empty_other()
            top_html = render_top_classes_html(top_items, colors=top_colors)
            click_xy = (x_disp, y_disp)

        return (
            _render_onehot(
                display_image, class_probs, concept_probs, onehot_mode, onehot_opacity, click_xy
            ),
            _render_multihot(
                display_image, concept_probs, multihot_name, multihot_opacity, click_xy
            ),
            top_html,
            tree_fig,
            other_html,
            click_xy,
        )

    def recompose_onehot(display_image, class_probs, concept_probs, mode, opacity, click_xy):
        if display_image is None:
            return None
        return _render_onehot(display_image, class_probs, concept_probs, mode, opacity, click_xy)

    def recompose_multihot(display_image, concept_probs, multihot_name, opacity, click_xy):
        if display_image is None:
            return None
        return _render_multihot(display_image, concept_probs, multihot_name, opacity, click_xy)

    def pick_sample(evt: gr.SelectData):
        if not static_examples:
            return None
        idx = int(evt.index) if evt.index is not None else 0
        if 0 <= idx < len(static_examples):
            from PIL import Image

            return np.array(Image.open(static_examples[idx]).convert("RGB"))
        return None

    with gr.Blocks(title="MERMAID Concept Bottleneck Demo", elem_id="mermaid-demo") as ui:
        gr.Markdown(
            "# MERMAID Concept Bottleneck Demo\n"
            "Upload an image, click **Predict**, then click any overlay pixel to inspect classes and taxonomy."
        )

        display_state = gr.State(None)
        class_probs_state = gr.State(None)
        concept_probs_state = gr.State(None)
        pred_mask_state = gr.State(None)
        click_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    type="numpy", image_mode="RGB", label="Upload image", height=200
                )
                predict_btn = gr.Button("Predict", variant="primary")
                if static_examples:
                    sample_gallery = gr.Gallery(
                        value=[(p, Path(p).name) for p in static_examples],
                        label="Sample images",
                        columns=3,
                        height=180,
                        allow_preview=False,
                    )
                else:
                    sample_gallery = None
                with gr.Accordion("Overlay legend", open=False):
                    gr.Markdown(
                        "**one-hot**: argmax class or taxonomic concept; alpha = softmax × opacity.\n\n"
                        "**multi-hot**: sigmoid heatmap for one concept; viridis colormap."
                    )

            with gr.Column(scale=2):
                onehot_mode = gr.Dropdown(
                    choices=onehot_choices, value=default_onehot, label="one-hot"
                )
                onehot_opacity = gr.Slider(0, 1, value=0.5, step=0.05, label="One-hot opacity")
                onehot_img = gr.Image(
                    type="numpy",
                    label="one-hot (click a pixel)",
                    interactive=False,
                    elem_id="mermaid-onehot-img",
                )

            with gr.Column(scale=2):
                multihot_mode = gr.Dropdown(
                    choices=multihot_choices, value=multihot_choices[0], label="multi-hot"
                )
                multihot_opacity = gr.Slider(0, 1, value=0.5, step=0.05, label="Multi-hot opacity")
                multihot_img = gr.Image(
                    type="numpy",
                    label="multi-hot (click a pixel)",
                    interactive=False,
                    elem_id="mermaid-multihot-img",
                )

            with gr.Column(scale=2):
                top_classes_html = gr.HTML(render_top_classes_html([]))
                other_html = gr.HTML(_empty_other())

        taxonomy_plot = gr.Plot(
            render_taxonomy_tree(None, rank_index, parents, top_k=TOP_K_TREE),
            label="Predicted Concepts: Taxonomy",
        )

        predict_outputs = [
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
        ]
        predict_btn.click(
            run_predict,
            inputs=[input_img, onehot_mode, onehot_opacity, multihot_mode, multihot_opacity],
            outputs=predict_outputs,
        )

        recompose_onehot_inputs = [
            display_state,
            class_probs_state,
            concept_probs_state,
            onehot_mode,
            onehot_opacity,
            click_state,
        ]
        recompose_multihot_inputs = [
            display_state,
            concept_probs_state,
            multihot_mode,
            multihot_opacity,
            click_state,
        ]
        onehot_mode.change(recompose_onehot, recompose_onehot_inputs, onehot_img)
        onehot_opacity.change(recompose_onehot, recompose_onehot_inputs, onehot_img)
        multihot_mode.change(recompose_multihot, recompose_multihot_inputs, multihot_img)
        multihot_opacity.change(recompose_multihot, recompose_multihot_inputs, multihot_img)

        click_inputs = [
            display_state,
            class_probs_state,
            concept_probs_state,
            onehot_mode,
            onehot_opacity,
            multihot_mode,
            multihot_opacity,
        ]
        click_outputs = [
            onehot_img,
            multihot_img,
            top_classes_html,
            taxonomy_plot,
            other_html,
            click_state,
        ]
        onehot_img.select(on_click, click_inputs, click_outputs)
        multihot_img.select(on_click, click_inputs, click_outputs)
        if sample_gallery is not None:
            sample_gallery.select(pick_sample, inputs=None, outputs=input_img)

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
        server_port=args.port, share=args.share, css=CSS
    )


if __name__ == "__main__":
    main(sys.argv[1:])
