# Plan: multi-task hierarchical head on the LoRA trunk ("plan head")

## Context & goal

The extracted LoRA baseline (`LinearLoRADINOv3`, `training_mode=standard`) emits a **single softmax over 72
flat benthic classes**. To give reviewers *every taxonomic level + a growth form, each with its own
confidence* (the coworker's ask) without paying the concept-bottleneck's accuracy tax, add a **multi-task
head**: keep the frozen DINOv3 + LoRA q/v trunk and predict, in parallel from the shared features:

1. the **flat class** (direct, unchanged — protects mIoU),
2. **each taxonomic rank** (kingdom→genus) as its own softmax,
3. **each growth form / health / functional concept** as an independent sigmoid.

**Key insight — this is the concept bottleneck *minus* the bottleneck.** The concept encoding, targets,
activation, loss helpers, and metrics already exist; the only genuinely new part is a model `forward` whose
**class head reads features, not concepts**. So this is a small, well-scoped change that reuses ~80% of the
concept-bottleneck plumbing.

```
                          ┌─ class head  (Conv2d 768→72)        → class logits      (direct; mIoU)
DINOv3+LoRA tokens (768) ─┤
                          └─ concept head (Conv2d 768→C)  → activate → concept outputs (per-rank softmax
                                                                        + growth-form sigmoid)
```
Contrast: concept-bottleneck routes `class = Conv2d(C→72)(concepts)` ([models.py:322-326](mermaidseg/model/models.py)),
which caps class accuracy at what the concepts encode. Multi-task removes that constraint.

## Output contract

`forward → ConceptBottleneckOutput(logits, concept_outputs, concept_logits)` — the **same dataclass the CBM
already returns** ([models.py:18-28](mermaidseg/model/models.py)), so every downstream consumer (batch_predict,
Evaluator, the demo) works unchanged:
- `.logits` `(B, 72, H, W)` — flat class, direct from features.
- `.concept_outputs` `(B, C, H, W)` — activated: per-rank softmax (kingdom→genus) + sigmoid (growth forms /
  health / functional), via the existing `concept_outputs_activation` ([models.py:362-374](mermaidseg/model/models.py)).
- `.concept_logits` `(B, C, H, W)` — raw, for the loss.

## Implementation — files & changes

### 1. New model class — `mermaidseg/model/models.py`
Add `MultiTaskDINOv3` (and LoRA / DPT variants mirroring the CBM ones). It's `ConceptBottleneckDINOv3`
with the class head rewired to features:
- reuse: encoder build + LoRA wrap ([models.py:159-186](mermaidseg/model/models.py)), the concept head
  (`ConceptHead`/`concept_proj`), and `concept_outputs_activation`.
- change: `self.class_head = LinearClassifier(hidden, tw, th, num_classes)` on the patch tokens; drop the
  `concept_classifier` (Conv2d C→num_classes). `forward`: `class_logits = self.class_head(tokens)`;
  `concept_logits = concept_head(tokens)`; `concept_outputs = concept_outputs_activation(concept_logits)`;
  return all three.
- Variants to add (match the LoRA run + the CBM's DPT variants): `MultiTaskLoRADINOv3` (q/v default),
  optionally `MultiTaskDPTLoRADINOv3`. Model is selected **by name** via
  `getattr(mermaidseg.model.models, model_name)` ([meta.py:205](mermaidseg/model/meta.py)) — no registry edit.

### 2. Training mode — `mermaidseg/model/meta.py`
- add `"multitask"` to the validated set ([meta.py:158-163](mermaidseg/model/meta.py)).
- construction branch: like concept-bottleneck, pass `num_concepts` + `concept_value2id`
  ([meta.py:207-209](mermaidseg/model/meta.py)); do **not** overwrite `num_classes` (that's concept mode).
- `batch_predict` / `batch_predict_loss`: reuse the **concept-bottleneck branch verbatim** — it already
  returns `(logits, concept_outputs)` and feeds `concept_logits + logits` to the loss
  ([meta.py:372-374, 426-431](mermaidseg/model/meta.py)). `has_concepts` already true for concept modes
  ([meta.py:450-454](mermaidseg/model/meta.py)) so `_to_concept_labels` targets are built.

### 3. Loss — `mermaidseg/model/loss.py`
Reuse the concept helpers: `calculate_taxonomic_rank_loss` ([concept_metrics.py:51](mermaidseg/model/concept_metrics.py))
and `calculate_multi_hot_concept_loss` ([concept_metrics.py:93](mermaidseg/model/concept_metrics.py), BCE with
0=invalid/1=False/2=True + background masking). The CBM already has a combined loss (selected by
`getattr(mermaidseg.model.loss, type)`, [meta.py:229](mermaidseg/model/meta.py)) — model the multi-task loss on
it:
```
L = w_class · CE/Focal(class_logits, target_labels)          # the flat-class term (keeps mIoU honest)
  + w_tax   · Σ_rank taxonomic_rank_loss(concept_logits[rank], concept_target[rank])
  + w_gf    · multi_hot_concept_loss(concept_logits[binary], concept_target[binary])
```
Expose `w_class / w_tax / w_gf` in the loss config. Start `1.0 / 1.0 / 1.0`; if class mIoU regresses vs the
flat baseline, raise `w_class`.

### 4. Configs
- `configs/model_config_dinov3_multitask_lora.yaml` — `name: MultiTaskLoRADINOv3`, same encoder/LoRA block as
  `model_config_dinov3_lora_qv_r8.yaml`.
- `configs/training_config_dinov3_multitask.yaml` — `training_mode: multitask`, the combined loss + weights,
  `concept_mapping_path: configs/class_to_concepts.csv`, `label_roll_up: True`, same `class_subset`.
- `sagemaker/runs/dinov3_multitask_lora.yaml` — wires the above; `num-workers: 7`, `persistent-workers` default.

## "Specify which to report" — the reporting knob
The reportable set = the concept channels the head emits = the columns present in
`configs/class_to_concepts.csv` → `ConceptSchema.from_csv` ([concept_schema.py:32-71](mermaidseg/dataset_reconciliation/concept_schema.py)).
Want only genus + family + growth form? Ship a schema CSV with just those columns; `num_concepts` and the
head width follow automatically. At **inference** you can also surface any subset by slicing
`concept_outputs` channels (the demo already does this via `parse_concept_rank`).

## Metrics / eval
`Evaluator.evaluate_concepts` ([eval.py:266-303](mermaidseg/model/eval.py)) already scores per-rank accuracy
(`accuracy/{rank}`) + multi-hot (`accuracy/multi_hot`), and `accumulate` scores the flat class (mIoU/F1). It
fires whenever `concept_outputs` is present — so multitask gets both the flat-class metrics **and** the
per-level/growth-form metrics for free. This is what makes the head-to-head vs the flat baseline fair (same
split, same mIoU + new concept metrics).

## Test plan
- `tests/model/` shape contract: `MultiTaskLoRADINOv3(...)` returns `ConceptBottleneckOutput` with
  `logits (B,72,H,W)` and `concept_outputs (B,C,H,W)`; class head gradients flow independently of concept head
  (perturbing concept channels doesn't change class logits — proves no bottleneck).
- `meta.py` mode wiring: `training_mode=multitask` builds the model with `num_concepts` set and `num_classes`
  **unchanged** (regression guard against the concept-mode overwrite).
- loss: combined loss returns finite scalar; weights scale the terms; background-masked BCE ignores ignore-index.
- offline smoke: `ml_pipeline_integration_tests`-style dataset→model→train-step with a tiny synthetic concept
  schema, one step, asserts all three loss terms non-zero.

## Verification (end to end)
1. Unit + smoke green; ruff/docformatter clean.
2. Short local/1-epoch run on the CoralNet+MERMAID config: confirm MLflow logs `miou` **and**
   `accuracy/genus`, `accuracy/family`, `accuracy/multi_hot`.
3. Full run on the **same source-disjoint split as the flat baseline**, then compare: does mIoU hold near the
   flat 0.290 while the per-level + growth-form metrics come online? That answers "what do we lose/gain vs the
   baseline."

## Risks & decisions
- **Consistency across levels is not enforced.** Independent per-rank heads can disagree (genus vs family). If
  hierarchical coherence is required, that's a follow-up (e.g. hierarchical softmax / conditioning children on
  parents) — out of scope for v1; note it.
- **Class↔concept agreement.** The direct class head and the concept heads can disagree on a pixel. That's the
  price of not bottlenecking; surface both in the UI and let the reviewer arbitrate.
- **Loss weighting** is the main tuning knob; keep the flat-class term dominant until mIoU parity is confirmed.
- **Decision vs Path 2 (CBM):** if interpretability ("the class is *explained by* its concepts") matters more
  than raw mIoU, prefer the bottleneck. If protecting the flat-class metric matters more, prefer this head.

## Effort
Small–medium: one new model class + variants, ~10-line mode branch, one combined loss (mostly assembling
existing helpers), 3 configs, ~4 tests. No changes to the data pipeline, registry, or DataLoader.
