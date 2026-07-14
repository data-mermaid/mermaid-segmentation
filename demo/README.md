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

Interactive demo for **concept-bottleneck coral reef benthic segmentation** (DINOv3 ViT-L + DPT head + LoRA). Upload a reef image, hit **Run segmentation**, then click any pixel to inspect its top classes, taxonomy, and concept activations.

Live Space: **https://huggingface.co/spaces/datamermaid/mermaid-segmentation**

## Using it

- **one-hot panel** — argmax class or taxonomic rank (kingdom → genus); overlay alpha = softmax × opacity slider.
- **multi-hot panel** — sigmoid heatmap for a single morphologic / non-coral concept (viridis).
- **click a pixel** — top-3 classes, taxonomy ladder (on every tab), and the top/bottom "other" concept activations at that point.
- Sample reef images are in the gallery.

## Model & artifacts

- **Checkpoint** (ViT-L LoRA DPT concept-bottleneck, 79 classes / 704 concepts) is pulled at startup from the private HF model repo [`datamermaid/mermaid-segmentation-cbm`](https://huggingface.co/datamermaid/mermaid-segmentation-cbm) — override with `DEMO_CHECKPOINT_REPO` / `DEMO_CHECKPOINT_FILE`, or point `DEMO_CHECKPOINT` at a local file.
- **Bundled here:** `id2label.json`, `concept_id2name.json`, `model_config_cbm_dpt_lora_vitl.yaml`, `class_to_concepts.csv` (taxonomy). The label/concept JSONs come from the training run's MLflow `metadata/` artifacts and must match the checkpoint.
- **DINOv3 backbone is gated.** The runtime `HF_TOKEN` must have accepted the [DINOv3 license](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and (for the private checkpoint repo) read access to it.

Provenance of the bundled metadata (pull from the training run's MLflow `metadata/` artifacts when swapping checkpoints):

| File | MLflow artifact path | How it is produced |
|------|---------------------|-------------------|
| `id2label.json` | `metadata/id2label.json` | Built in `scripts/train.py` as `{0: "ignore", **registry.target_id2label}` and logged by `Logger` |
| `concept_id2name.json` | `metadata/concept_id2name.json` | Built by `SourceLabelRegistry._build_concepts()` and logged via `Logger.log_reconciliation()` |

## Run locally

From the repo root:

```bash
uv sync --extra demo
export HF_TOKEN=...                 # DINOv3 access (+ checkpoint repo if private)
# optional: export DEMO_CHECKPOINT=/path/to/local_checkpoint   # else pulled from the HF repo
uv run python demo/app.py --port 7860
```

Model definitions and concept helpers are imported from the `mermaidseg` package (not vendored). Supported `model.name` values: `ConceptBottleneckDINOv3`, `ConceptBottleneckDPTDINOv3`, `ConceptBottleneckDPTLoRADINOv3`.

See [LOCAL.md](LOCAL.md) for the full local-hosting guide: token caveats for the private checkpoint repo, artifact-resolution env vars, and failure signatures.

## Deploy (Hugging Face Space)

Only this `demo/` folder is uploaded; `mermaidseg` installs from the pinned ref in [`requirements.txt`](requirements.txt):

```bash
hf upload datamermaid/mermaid-segmentation ./demo . --repo-type=space
```

Set the Space secret **`HF_TOKEN`** (DINOv3 + checkpoint-repo read). The Space is pinned to Python 3.12 + Gradio 6.x via the frontmatter above. Keep the `mermaidseg` ref in `requirements.txt` aligned with the checkpoint's training code.

The Space runs on **ZeroGPU**: `import spaces` precedes torch in `app.py`, the click-bound `run_predict` carries `@spaces.GPU`, and the model is moved to `cuda` at startup so ZeroGPU can pack and stream the weights per request. Do not add `gradio`/`spaces`/`torch` pins to `requirements.txt` — the platform manages them (torch must stay on the ZeroGPU-supported build).
