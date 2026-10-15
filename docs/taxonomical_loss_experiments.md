# Taxonomical loss + dual morphology head — experiment notes

Standard-mode only (CBM unchanged). Use these configs against the usual data/logger split YAMLs.

## Phase 1 — flat Linear / LoRA + `TaxonomicalLoss`

| Run | Model config | Training config | Loss knobs |
|-----|--------------|-----------------|------------|
| CE baseline (linear) | `configs/model_config` LinearDINOv3 | `training_config_dinov3_linear.yaml` | `CrossEntropyLoss` |
| Taxonomical linear | same | `training_config_dinov3_linear_taxonomical.yaml` | `alpha=0.1`, `beta=0.1` |
| CE baseline (LoRA) | `model_config_dinov3_lora_qv_r{8,16}.yaml` | `training_config_dinov3_lora.yaml` | `CrossEntropyLoss` |
| Taxonomical LoRA | same | `training_config_dinov3_lora_taxonomical.yaml` | `alpha=0.1`, `beta=0.1` |

Suggested ablations (edit YAML or override):

1. `alpha=0.1`, `beta=0` — tree-distance only
2. `alpha=0`, `beta=0.1` — multi-level CE only
3. `alpha=0.1`, `beta=0.1` — full taxonomical loss

Compare leaf `f1` / `miou` **and** hierarchy diagnostics:

- `mean_tree_distance` (lower is better taxonomic severity)
- `ancestor_accuracy/<level>` (e.g. Hard coral)
- logged loss components: `classification`, `tree_distance`, `level/*`, `taxonomical_total`

## Phase 2 — dual head + `DualTaxonomicalLoss`

| Run | Model config | Training config |
|-----|--------------|-----------------|
| Dual LoRA | `model_config_dinov3_dual_lora_qv_r8.yaml` | `training_config_dinov3_dual_taxonomical.yaml` |

Morphology channels default to `plating`, `branching`, `massive`, `encrusting`, `tabular`. Targets come from `class_to_concepts.csv` (unknown → 0, masked out of morph BCE). Tune `gamma` independently of `alpha` / `beta`.

Report class metrics plus morph quality on **labeled** morph pixels only; confirm leaf taxonomy metrics do not regress vs Phase 1.

## LoRA ablation — launch-ready run configs

Three sibling run configs under `sagemaker/configs/`, identical except the loss (same data, image,
instance, `seed=42`, `epochs=200`, early stopping), all grouped under the MLflow experiment
`taxonomical-loss-ablation`:

| Arm | Config dir | Loss |
|-----|------------|------|
| CE baseline | `sagemaker/configs/lora_tax_ce/` | `CrossEntropyLoss` |
| Taxonomical | `sagemaker/configs/lora_tax_taxonomical/` | `TaxonomicalLoss` (`alpha=0.1`, `beta=0.1`) |
| Dual | `sagemaker/configs/lora_tax_dual/` | `DualTaxonomicalLoss` (`+gamma=0.1`, morphology head) |

**Prerequisite — rebuild the training image.** The taxonomical loss, `hierarchy_loss.py`, the dual
model, and the taxonomical/dual config YAMLs must be **committed and baked into
`mermaid-segmentation-jobs:training-latest`** before launching — the current image predates this
work. Commit branch `172-lora-qv-r8-r16`, then rebuild + push the image (needs `aws sso login`).

Launch each arm (after SSO + image rebuild):

```bash
make sm-launch SM_RUN_CONFIG=sagemaker/configs/lora_tax_ce/run.yaml          SM_CONFIG_DIR=sagemaker/configs/lora_tax_ce
make sm-launch SM_RUN_CONFIG=sagemaker/configs/lora_tax_taxonomical/run.yaml SM_CONFIG_DIR=sagemaker/configs/lora_tax_taxonomical
make sm-launch SM_RUN_CONFIG=sagemaker/configs/lora_tax_dual/run.yaml        SM_CONFIG_DIR=sagemaker/configs/lora_tax_dual
```

Swap `sm-launch` → `sm-dry-run` to preview without submitting. Each is a separate `ml.g5.2xlarge`
TrainingJob (up to 48h). For a single-variable read on the loss, one seed is enough to start;
re-launch with `seed=43,44` (override in the run YAML) if the per-level metrics look seed-sensitive
— which the local smoke suggested they are (`docs/local-loss-testing.md`).

## Non-goals (still)

- No CBM config changes
- Growth form is not in the taxonomic tree
- No taxon×growth-form compound class IDs
