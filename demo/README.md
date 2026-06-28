---
title: MERMAID Concept Bottleneck Demo
emoji: 🪸
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🪸 MERMAID Concept Bottleneck Demo

Interactive demo for **concept-bottleneck coral reef benthic segmentation** (DINOv3 ViT-L + DPT head + LoRA). Upload a reef image, hit **Predict**, then click any pixel to inspect its top classes, taxonomy, and concept activations.

Live Space: **https://huggingface.co/spaces/datamermaid/mermaid-segmentation**

## Using it

- **one-hot panel** — argmax class or taxonomic rank (kingdom → genus); overlay alpha = softmax × opacity slider.
- **multi-hot panel** — sigmoid heatmap for a single morphologic / non-coral concept (viridis).
- **click a pixel** — top-3 classes, taxonomy tree, and the top/bottom "other" concept activations at that point.
- Sample reef images are in the gallery.

## Model & artifacts

- **Checkpoint** (ViT-L LoRA DPT concept-bottleneck, 78 classes / 650 concepts) is pulled at startup from the HF model repo [`datamermaid/mermaid-segmentation-cbm`](https://huggingface.co/datamermaid/mermaid-segmentation-cbm) — override with `DEMO_CHECKPOINT_REPO` / `DEMO_CHECKPOINT_FILE`, or point `DEMO_CHECKPOINT` at a local file.
- **Bundled here:** `id2label.json`, `concept_id2name.json`, `model_config_cbm_dpt_lora_vitl.yaml`, `class_to_concepts.csv` (taxonomy). The label/concept JSONs come from the training run's MLflow `metadata/` artifacts and must match the checkpoint.
- **DINOv3 backbone is gated.** The runtime `HF_TOKEN` must have accepted the [DINOv3 license](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and (for the private checkpoint repo) read access to it.

## Run locally

From the repo root:

```bash
uv sync --extra demo
export HF_TOKEN=...                 # DINOv3 access (+ checkpoint repo if private)
# optional: export DEMO_CHECKPOINT=/path/to/local_checkpoint   # else pulled from the HF repo
uv run python demo/app.py --port 7860
```

Model definitions and concept helpers are imported from the `mermaidseg` package (not vendored). Supported `model.name` values: `ConceptBottleneckDINOv3`, `ConceptBottleneckDPTDINOv3`, `ConceptBottleneckDPTLoRADINOv3`.

## Deploy (Hugging Face Space)

Only this `demo/` folder is uploaded; `mermaidseg` installs from the pinned ref in [`requirements.txt`](requirements.txt):

```bash
hf upload datamermaid/mermaid-segmentation ./demo . --repo-type=space
```

Set the Space secret **`HF_TOKEN`** (DINOv3 + checkpoint-repo read). The Space is pinned to Python 3.12 + Gradio 6.x via the frontmatter above. Keep the `mermaidseg` ref in `requirements.txt` aligned with the checkpoint's training code.
