# Local loss smoke test

Exercise the new loss functions (`TaxonomicalLoss`, `DualTaxonomicalLoss`) end-to-end on your
laptop before spending a SageMaker job. Catches the class of wiring bug unit tests miss — model
output → loss shapes, buffer/device placement, the new loss components, and the hierarchy eval
metrics.

## Fully offline — no S3, no creds

`make train-local*` runs `scripts/diagnostics/local_loss_smoke.py`, which builds the **real** model
(by name, cached DINOv3 backbone) and the **real** loss (from a training config's `loss:` block)
over a **small synthetic label space + hierarchy**, feeds synthetic batches, and runs a few train
steps + one eval step. It touches no data — just the model + loss + eval code paths — so it runs in
a few seconds on CPU with nothing but the cached backbone.

```bash
make train-local-taxonomical   # TaxonomicalLoss: CE + tree-distance + level CE (LoRA q/v model)
make train-local-dual          # DualTaxonomicalLoss: + masked morphology BCE (dual head)

# generic — smoke any model + loss combo:
make train-local \
  LOCAL_MODEL_CONFIG=configs/model_config.yaml \
  LOCAL_TRAINING_CONFIG=configs/training_config_dinov3_linear_taxonomical.yaml
```

Prereq: the DINOv3 backbone cached locally (`facebook/dinov3-vitb16-pretrain-lvd1689m`). If it isn't,
run once with network + `HF_TOKEN` to populate the cache; after that `HF_HUB_OFFLINE=1` (set by the
target) keeps it offline.

### What a good run looks like

```
model=LinearLoRADINOv3  loss=TaxonomicalLoss  classes=7  dual=False  device=cpu
--- 3 train steps (synthetic batches) ---
  step 0: total=2.2453  |grad|=1.908  classification=1.9469  tree_distance=2.3468  level/hard coral=0.6370 ...
  ...
--- eval step (hierarchy metrics) ---
  mean_tree_distance = 1.9320
  ancestor_accuracy/hard coral = 0.9428
PASS — loss forward/backward, optimizer step, and hierarchy eval all ran offline.
```

Check that: the loss is **finite** and its total trends down; every configured component is
present and **non-zero** — `classification`, `tree_distance` (`alpha>0`), `level/<name>` (`beta>0`,
must be > 0, not a flat zero), `morphology` (dual, `gamma>0`); and the eval prints
`mean_tree_distance` + `ancestor_accuracy/<level>`. A crash indexing the distance matrix means the
loss buffers weren't co-located with the batch; a shape error means `num_classes` ≠ the head width.

## Optional: real-data smoke (needs S3)

To also exercise the data pipeline (dataset → registry → source-label mapping) on real MERMAID data,
run the training entrypoint against the MERMAID-only local config with `--dry-run` (a real loop
capped to 3 epochs × 1 batch). This needs `mermaid-core` AWS creds because MERMAID images live on S3:

```bash
MLFLOW_TRACKING_URI= AWS_PROFILE=mermaid-core uv run python scripts/train.py \
  --config-data configs/data_config_local.yaml \
  --config-model configs/model_config_dinov3_lora_qv_r8.yaml \
  --config-training configs/training_config_dinov3_lora_taxonomical.yaml \
  --config-logger configs/logger_config.yaml \
  --experiment-name local-loss-smoke --num-workers 0 --dry-run
```

`--num-workers 0` (local macOS uses `spawn`; the dataset S3 clients aren't fork/pickle-safe).
`MLFLOW_TRACKING_URI=` forces a local `./mlruns` store so the smoke never touches the shared
SageMaker MLflow.

## Comparing the losses offline (`make compare-local`)

`scripts/diagnostics/local_loss_compare.py` trains the **same** model (fresh, same seed) under
`CrossEntropyLoss`, `TaxonomicalLoss`, and `DualTaxonomicalLoss` on **identical** synthetic data,
then tabulates leaf metrics (`accuracy`, `miou`) against taxonomic-severity metrics
(`mean_tree_distance`, `ancestor_accuracy/<level>`). The synthetic task is *structured* so the
hierarchy is real — sibling classes get near-identical colors, distant classes distinct ones, with
noise so they're confusable.

```bash
make compare-local                       # default: 80 steps, noise 1.2
make compare-local COMPARE_ARGS="--steps 120 --noise 1.5"
```

**What it can and can't tell you.** It confirms all three losses train end-to-end on the same data
and surfaces the hierarchy metrics, and in a harder regime you can see the taxonomical loss's
intended effect (higher `ancestor_accuracy` — getting the coarse level right more often — at a small
leaf-accuracy cost). But it is a **controlled mechanism demo on synthetic data, not a coral-quality
verdict**: the effect is small and noisy, and whether the loss actually helps on reef imagery needs
a real training run. Use the SageMaker ablation matrix in
[`taxonomical_loss_experiments.md`](taxonomical_loss_experiments.md) for that.

## Comparing the losses on real CoralNet data (`make compare-local-coralnet`)

`scripts/diagnostics/local_loss_compare_coralnet.py` runs the same CE-vs-Taxonomical-vs-Dual
comparison on the **actual downloaded CoralNet subset** in `data/coralnet_local_subset` — real reef
photos, the real MERMAID benthic hierarchy, and the real confusion structure between coral genera —
instead of synthetic patches. It builds a compact label space from the most frequent `mermaid_label`
values, scatters the sparse point annotations into ignore-masked target masks (dilated by
`--padding`), derives morphology targets for the dual head from `growth_form_name`, and reads images
from the manifest's local `local_path` (no S3, no AWS creds).

```bash
make compare-local-coralnet                                   # 20/10 imgs, 60 steps
make compare-local-coralnet COMPARE_CORALNET_ARGS="--n-train 40 --steps 120 --top-k 24"
```

Loss weights default to the taxonomical training config (`alpha=beta=gamma=0.1`,
`damping_denominator=100`, `lr=1e-3`) so the run previews what the SageMaker job will do. The benthic
hierarchy is fetched once from the public MERMAID API (no creds) and cached to
`data/coralnet_local_subset/benthic_hierarchy.json`; after that the script is fully offline.

A representative run (20 train / 10 val images, 60 steps):

```
loss                    accuracy    miou  mean_tree_dist      anc_acc/hard coral
CrossEntropyLoss           0.550   0.246           1.785                   0.822
TaxonomicalLoss            0.558   0.255           1.734                   0.824
DualTaxonomicalLoss        0.584   0.275           1.627                   0.783
```

Here the taxonomical losses lower `mean_tree_distance` (less severe errors) while holding or nudging
leaf accuracy — the intended trade — and the `hard coral` ancestor accuracy (~0.82) confirms the
coral genera roll up correctly through the hierarchy.

**Still a smoke-scale run, not a verdict.** A handful of images and a few dozen CPU steps: the gaps
are small and seed-sensitive, and a genuine quality comparison needs the SageMaker ablation
(`docs/taxonomical_loss_experiments.md`). One real caveat this surfaced: with **sparse point labels**
(most pixels are `ignore`), a large `--alpha` over very few steps can collapse the model onto the
background channel, because the tree-distance term charges nothing for predicting `ignore`. The
config default (`alpha=0.1`) trains stably; keep `alpha` modest when steps are few.

## Notes

- These are **wiring/smoke checks, not convergence runs** — synthetic (or 1-batch) data.
- For real ablations (`alpha`/`beta`/`gamma`, `level_names`), use the SageMaker matrix in
  [`docs/taxonomical_loss_experiments.md`](taxonomical_loss_experiments.md).
