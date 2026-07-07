# Running the demo locally

Developer guide for hosting the MERMAID Concept Bottleneck demo on your own machine. For what the demo *does*, see [README.md](README.md).

## TL;DR

```bash
# from the repo root
uv sync --extra demo
export HF_TOKEN=hf_...            # see "Access requirements" below
uv run python demo/app.py --port 7860
```

Open <http://localhost:7860>. First prediction on CPU takes a while (ViT-L backbone) — expect tens of seconds per image; use `--device cuda` on a GPU machine.

## How the demo finds its artifacts

At startup the app resolves, in order (CLI flag → env var → default):

| Artifact | CLI flag | Env var | Default |
|---|---|---|---|
| Checkpoint | `--checkpoint` | `DEMO_CHECKPOINT` | downloaded from `DEMO_CHECKPOINT_REPO`/`DEMO_CHECKPOINT_FILE` (`datamermaid/mermaid-segmentation-cbm` / `checkpoint.pt`) |
| Model config | `--model-config` | `DEMO_MODEL_CONFIG` | bundled `demo/model_config_cbm_dpt_lora_vitl.yaml` |
| Label map | `--id2label` | — | bundled `demo/id2label.json` |
| Concept names | `--concept-id2name` | — | bundled `demo/concept_id2name.json` |
| Taxonomy edges | `--taxonomy-csv` | `DEMO_TAXONOMY_CSV` | bundled `demo/class_to_concepts.csv` |

Downloads land in the standard Hugging Face cache (`~/.cache/huggingface/hub`, override with `HF_HOME`), so the checkpoint is only fetched once.

## Access requirements (two separate gates)

The demo needs to fetch weights from **two** Hugging Face repos, and **one token/account must satisfy both**:

1. **DINOv3 backbone (gated license).** The model config points at
   [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
   Visit the model page while logged in as the account that owns your token and accept the license. Approval is per-account and can take time (Meta reviews requests).
2. **Checkpoint repo (private).** `datamermaid/mermaid-segmentation-cbm` is a private repo under the
   `datamermaid` org. Your token needs read access to it.

Either export the token (`export HF_TOKEN=hf_...`) or log in once with `hf auth login` — the app passes `HF_TOKEN` through when set and otherwise falls back to the cached login.

### Private-checkpoint caveats

- **Fine-grained tokens scoped to "your repositories" cannot read the org's private repo** — being an org member or repo collaborator is not enough. When creating a fine-grained token, explicitly grant it read access to `datamermaid/mermaid-segmentation-cbm` (or use a token from an account with org-wide read).
- **Both gates, same account.** A token that can read the private checkpoint but hasn't accepted the DINOv3 license (or vice versa) fails halfway through startup.
- **Failure signatures:**
  - `GatedRepoError` / `403` mentioning `facebook/dinov3-...` → the token's account hasn't accepted (or been granted) the DINOv3 license.
  - `RepositoryNotFoundError` / `401` on `datamermaid/mermaid-segmentation-cbm` → the token can't see the private repo (Hugging Face reports private repos as *not found* to unauthorized tokens, so a "404" here usually means a permissions problem, not a wrong name).
- **Bypassing the private repo entirely:** if you already have the checkpoint (e.g. from the training run's artifacts), skip the download:

  ```bash
  export DEMO_CHECKPOINT=/path/to/checkpoint.pt
  uv run python demo/app.py --port 7860
  ```

  You still need the DINOv3 gate — the backbone weights are fetched separately from the checkpoint.

## Matching artifacts matter

The bundled `id2label.json` / `concept_id2name.json` / model config correspond to the hosted checkpoint (78 classes / 650 concepts) and come from the training run's MLflow metadata. If you point `DEMO_CHECKPOINT` at a different checkpoint, bring its matching label/concept JSONs (`--id2label`, `--concept-id2name`) — mismatched maps produce garbage class names or a shape error at load time.

`requirements.txt` pins `mermaidseg` to the training-code ref the checkpoint expects; local runs via `uv sync` use your checkout instead, which must be on a compatible branch (concept `hidden_states` activated to [0, 1]).

## Useful flags

| Flag | Effect |
|---|---|
| `--port 7860` | serve port (default 7860) |
| `--share` | public Gradio share link |
| `--device cuda` | force device (default: cuda if available, else cpu) |

## Version notes

- Python **3.12** and Gradio **6.17.3** — the same pins as the deployed Space (`uv.lock` resolves gradio to exactly the Space's version, so local rendering matches production).
- Gradio 6 quirk to keep in mind when editing the UI: `css=` is passed to `launch()`, not `Blocks()`, and page-level CSS is scoped to `.gradio-container` (a `Blocks(elem_id=...)` never reaches the DOM).
