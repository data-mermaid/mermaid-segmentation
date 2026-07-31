# LoRA baseline: per-segment readout

What the flat (standard-mode) LoRA segmentation model can report per point, and how to run it.

## What the model outputs

The LoRA baseline (`LinearLoRADINOv3`, `training_mode=standard`, best run `dinov3-lora-qv-r8`, epoch 69) is a
**flat classifier**: DINOv3 tokens → a 1×1-conv head → **one softmax over 72 benthic-attribute classes** per
pixel. The 72 classes are a heterogeneous mix of levels — genus (acropora, porites), functional groups (hard
coral, macroalgae), and substrate (sand, rubble). It does **not** natively emit taxonomic ranks or growth form.

Metrics at the best epoch (source-disjoint holdout): accuracy **0.669**, mIoU **0.290**, mIoU-weighted **0.524**.

## What the readout adds (no retraining)

`scripts/diagnostics/posthoc_readout.py` keeps the full distribution instead of rolling up, and derives a
confidence at every taxonomic level by **aggregating the 72-class softmax up the benthic-attribute hierarchy**:
`P(node) = Σ probabilities of the leaf classes under that node`. Because a parent aggregates its children, the
confidence is non-decreasing toward the root — so a reviewer sees the leaf guess *and* how confident the model
is at each coarser level, and can confirm/correct at whichever level they trust.

Real example (point on an acropora-table quadrat):

```
top-3:  acropora 51%  ·  hard coral 46%  ·  galaxea 1%
ladder: acropora 51%  →  acroporidae 52%  →  hard coral 99%
```

The current UI would roll this up and show only "hard coral" (because the leaf is below the confirm threshold),
discarding that the model's specific guess is acropora. The readout keeps both.

## Growth form — follow-up (needs a trained head)

Growth form is **not** in this readout, and cannot be. It is a separate axis the flat model never predicts, it
is not part of the taxonomic hierarchy we aggregate over, and it is **not derivable from a class** — one genus
maps to many forms (e.g. `porites` → encrusting / massive / branching / foliose). A genuine per-point
growth-form confidence requires a model with a growth-form head. The design for that (a multi-task hierarchical
head on the same LoRA trunk that adds per-rank taxonomy softmaxes + 24 growth-form sigmoids) is in
[`docs/multitask-head-plan.md`](multitask-head-plan.md).

## How to run

1. Download the checkpoint dir from MLflow (run `5e59e544…`, artifact `best-model/`) — you need `model.pt`,
   `config.json`, and `target_id2label.json` in one directory. The frozen DINOv3 backbone is reloaded from the
   HF cache (not stored in the checkpoint), so no HF token is needed if it is already cached locally.
2. Run the readout + render the figure:

```bash
python scripts/diagnostics/posthoc_readout.py <ckpt_dir> demo/static/nadir_acropora_table.jpg --out out_readout
python scripts/diagnostics/render_readout.py out_readout
```

Outputs: `out_readout/readout.json` (per-point top-K + hierarchy ladder for a 5×5 grid) and
`out_readout/readout_figure.png` (a reviewer-facing figure). Both scripts are read-only inference; the ladder
needs the MERMAID benthic-attribute API (`initialize_benthic_hierarchy`) — without it the readout degrades to
top-K only.
