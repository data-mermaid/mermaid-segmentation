# MLflow Dataset Statistics — Design

**Issue:** [#72 — Add coral labels and statistics per run to mlflow](https://github.com/data-mermaid/mermaid-segmentation/issues/72)
**Milestone:** 0.1 Engineering Readiness
**Status:** Design
**Author:** Lauren Yee
**Date:** 2026-05-01

## Background

Stakeholders (Emily, Kim) want per-run dataset statistics tracked in MLflow for traceability, drift detection, and run-to-run QA. The sister `mermaid-classifier` repo already implements this pattern; this design mirrors that precedent in `mermaidseg`.

The classifier logs `ba_counts.csv` (per benthic-attribute), `project_stats_train_data.csv` (per source/project, with split breakdown), and `train_summary.yaml`. We mirror this scoped to MermaidSeg's data shapes (no growth-form, segmentation classes instead of classifier labels).

Pixel-level mask statistics are explicitly **out of scope** for this spec; they require materializing all masks once per run and will be tracked in a follow-up issue once we benchmark the cost.

## Goals

- One MLflow artifact bundle per training run that captures class and source distribution across train/val/test splits.
- Drop-in alignment with PR #88's `log_dataloader_params` pattern (call site, error handling, init order).
- Read-only against datasets — no mutation, no caching, no side effects.

## Non-goals

- Pixel-level mask statistics (follow-up issue).
- Cross-run aggregation or drift dashboards (consumed via MLflow UI / downstream tooling).
- Backfilling stats onto historical runs.

## API

Add one public method to `Logger` in `mermaidseg/logger.py`:

```python
def log_dataset_statistics(
    self,
    splits: dict[str, Dataset | Subset],
    *,
    artifact_dir: str = "dataset_stats",
) -> None
```

- `splits` keys are arbitrary labels (e.g. `"train"`, `"val"`, `"test"`) and become column suffixes in the CSVs.
- Independent of existing `log_dataset` / `log_datasets` (which log MLflow `Dataset` *inputs* — different MLflow concept).
- Defensive: gated on `_ensure_active_run()`, wrapped in `try/except`, never raises into training.
- Uses `mlflow.log_text(csv_str, "<name>.csv")` rather than `mlflow.log_table` (the latter writes JSON, mirroring classifier choice).

## Artifact schema

Four artifacts under `dataset_stats/`:

### `class_counts.csv`
Per benthic-attribute distribution across splits. Headline artifact for class-imbalance / drift QA.

| Column | Type | Notes |
|---|---|---|
| `class_id` | int | from `id2label` |
| `class_name` | str | from `id2label` (incl. background) |
| `class_kind` | str | `target` \| `background` \| `unclassified` (see policy below) |
| `train_annotations` | int | annotation rows in train split |
| `val_annotations` | int | annotation rows in val split |
| `test_annotations` | int | annotation rows in test split |
| `train_images` | int | unique images in train containing this class |
| `val_images` | int | unique images in val containing this class |
| `test_images` | int | unique images in test containing this class |
| `train_fraction` | float | `train_annotations / sum(train_annotations across all classes)` |
| `val_fraction` | float | analogous |
| `test_fraction` | float | analogous |

One row per class in the parent dataset's `id2label`. This `id2label` is already post-`class_subset` filtering (constructed in `BaseCoralDataset.__init__`), so unwanted classes do not appear here. Zero-fill for classes absent from a split.

**`class_kind` policy** (case-insensitive match on `class_name`):
- `background` → the synthetic background class added by `BaseCoralDataset` (id 0).
- `unclassified` → names whose lowercased form is exactly `other`, `unknown`, or `unclassified`. Qualified names like `"Other Invertebrates"` remain `target` (they are real, distinct classes the model is trained on). These rows are kept as their own classes (not folded into background) and are visible in counts so reviewers can see how much of the dataset is unlabeled noise.
- `target` → all remaining named classes.

### `source_stats.csv`
Per source/region distribution.

| Column | Type | Notes |
|---|---|---|
| `source_key` | str | `region_name` (Mermaid) or `source_id` (CoralNet, cast to str) |
| `source_type` | str | `"region"` or `"source"` |
| `train_images` | int | |
| `val_images` | int | |
| `test_images` | int | |
| `train_annotations` | int | |
| `val_annotations` | int | |
| `test_annotations` | int | |

`source_type` is inferred per child dataset from which column is present (`region_id` vs `source_id`). Schemas are not unified — Mermaid rows and CoralNet rows coexist in the same CSV, distinguished by `source_type`.

### `class_by_source.csv`
Long-format class × source × split matrix. Highest-leverage artifact for geographic / source-level drift QA — without this, "Indonesia is Acropora-dominated, Caribbean is Macroalgae-dominated" is invisible.

| Column | Type | Notes |
|---|---|---|
| `source_key` | str | matches `source_stats.csv` |
| `source_type` | str | `"region"` or `"source"` |
| `class_id` | int | |
| `class_name` | str | |
| `split` | str | `"train"` / `"val"` / `"test"` |
| `annotations` | int | annotation rows for this (source, class, split) |
| `images` | int | unique images for this (source, class, split) |

Long format (rather than a wide pivot) keeps the file pivot-friendly in pandas/Excel and avoids 50+ columns when there are many sources. Zero-count rows are omitted to keep size sane.

### `train_summary.yaml`

```yaml
total_images: int
total_annotations: int
splits:
  train: { images: int, annotations: int }
  val:   { images: int, annotations: int }
  test:  { images: int, annotations: int }
class_subset: [str, ...] | null   # from config.data.class_subset
num_classes: int
# Class-balance summaries — training split only, excludes background and unclassified.
top1_share: float                       # share of training annotations going to the top class
top3_share: float                       # share going to the top-3 classes
top5_share: float                       # share going to the top-5 classes
                                        # If fewer than K eligible classes exist, the share is computed over the available classes (effectively the full eligible distribution).
effective_num_classes: float            # exp(entropy) over the training class distribution
# Annotations-per-image distribution per split.
annotations_per_image:
  train: { mean: float, median: float, p10: float, p90: float, min: int, max: int }
  val:   { mean: float, median: float, p10: float, p90: float, min: int, max: int }
  test:  { mean: float, median: float, p10: float, p90: float, min: int, max: int }
```

**Why these summaries (vs. `max_count / min_count` ratio):** with severe coral-reef class imbalance, `max/min` is dominated by rare classes (often 0, giving null) and is identical run-to-run — it does not distinguish a healthy run from a broken one. The replacements:
- `topK_share` flags long-tail collapse (suddenly top-3 = 95% means most classes were filtered out).
- `effective_num_classes` is a single scalar that drops cleanly when the long tail collapses.
- The annotations-per-image distribution catches misjoined data (e.g. images that lost their points during the join — `min: 0` immediately).

A "minimum-annotations-per-class" trainability threshold was considered but rejected — once we train on all sources, the count loses meaning as a per-run QA signal, and `topK_share` + `effective_num_classes` already cover long-tail collapse.

## Subset & wrapper resolution

Single private helper:

```python
def _resolve_annotations(split) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]] | None
```

Handles four input shapes:

1. **Plain dataset** (`MermaidDataset` / `CoralNetDataset`):
   - Returns `(dataset.df_annotations, dataset.df_images, dataset.id2label)` directly.

2. **PyTorch `Subset`** (from `random_split` — what `scripts/train.py` and `Base_Pipeline.ipynb` produce):
   - `parent = subset.dataset; indices = subset.indices`
   - `image_ids = parent.df_images.iloc[indices]["image_id"]`
   - `df_annotations = parent.df_annotations[parent.df_annotations["image_id"].isin(image_ids)]`
   - `df_images = parent.df_images.iloc[indices].reset_index(drop=True)`
   - Pass through `parent.id2label`.

3. **Combined / `ConcatDataset` wrapper** (`Combined_Pipeline.ipynb`):
   - Detect via `_datasets` or `datasets` attribute (matches existing `log_datasets`).
   - Recursively resolve each child, concat resulting frames.
   - For `id2label`, take the union; if children differ, log a warning and use the first child's mapping.

4. **Unknown shape** → return `None`, log a warning naming the split, do not raise.

This helper is the only place that touches dataset internals — keeping the artifact-generation code dataset-agnostic.

## Error handling

Mirrors existing logger conventions (`log_dataloader_params`, `log_dataset`):

| Failure | Behavior |
|---|---|
| MLflow not active | Early return via `_ensure_active_run()`. Info-log if disabled. |
| Per-split resolution fails | Catch in `_resolve_annotations`, log warning naming split, skip that split's contribution, continue. |
| Per-artifact serialization fails | Catch around each of the four artifacts independently; one artifact failure does not drop the others. |
| Empty result for a split | Emit zero-row CSV with headers (empty data is meaningful signal — never skip silently). |
| Outer guard | Whole method wrapped in `try/except Exception`; logs `"Failed to log dataset statistics: %s"` and returns. Training never aborts because of stats logging. |

The principle: **strictly additive observability**. If it breaks, training continues, the rest of MLflow logging continues, and a warning tells you exactly which split/artifact failed.

## Call sites

### `scripts/train.py`
After PR #88's `log_dataloader_params` block:

```python
logger.log_dataloader_params(train_loader, prefix="train_loader")
logger.log_dataloader_params(val_loader, prefix="val_loader")
logger.log_dataloader_params(test_loader, prefix="test_loader")
logger.log_dataset_statistics({"train": train_ds, "val": val_ds, "test": test_ds})
```

### `nbs/Base_Pipeline.ipynb`
Same pattern, in the cell that follows `Logger(...)` initialization, alongside the existing `logger.log_dataset(...)` call.

### `nbs/Combined_Pipeline.ipynb`
Same pattern — combined wrappers handled by `_resolve_annotations` recursion. The concept-bottleneck notebook is intentionally not wired in this iteration; CBM stats live with the broader CBM workstream.

## Testing

Add to `tests/test_logger.py` following existing patterns (`tmp_mlflow_uri` fixture, `unittest.mock.patch` against `mermaidseg.logger.mlflow.*`).

### Unit tests — `_resolve_annotations`
- Plain dataset: returns inputs unchanged.
- Subset: filters parent annotations to subset image_ids; row count matches.
- ConcatDataset-style wrapper: concatenated frames; warning logged on conflicting `id2label`.
- Unknown shape: returns `None`, logs warning, does not raise.

### Integration tests — `log_dataset_statistics`
- Happy path: 3 splits → exactly 4 artifacts present (`class_counts.csv`, `source_stats.csv`, `class_by_source.csv`, `train_summary.yaml`), schemas match.
- Class absent from a split → CSV has zero counts (not missing rows).
- MermaidDataset (region) vs CoralNetDataset (source) → `source_type` column tagged correctly across both `source_stats.csv` and `class_by_source.csv`.
- `class_kind` tagging: a class named `"Other"` → `unclassified`; the synthetic background → `background`; named coral classes → `target`.
- Summary metrics: `top1_share` + `top3_share` + `top5_share` are non-decreasing; `effective_num_classes` falls within `[1, num_classes]`.
- Annotations-per-image distribution: median/p10/p90 match a hand-computed fixture; `min: 0` surfaces correctly when an image has no annotations.
- One split unresolvable → other splits still logged, warning emitted.
- `mlflow.log_text` raises on one artifact → other three still written.
- MLflow disabled → early return, no error.

### Fixtures
Build minimal `df_annotations` / `df_images` from in-memory `pd.DataFrame` literals (same approach as `test_datasets.py`). No S3, no network.

### Out of scope
- Notebook smoke tests — verified manually.
- End-to-end SageMaker MLflow round-trip — covered by existing integration tests.

## Alignment with PR #88

PR #88 (`Add num_workers logging to mlflow`) introduces `random_split` → Subset usage and the `log_dataloader_params` call pattern. This design:

- Uses the same call-site placement (post-`Logger` init, before `train_model`).
- Reuses the same `_ensure_active_run()` + `try/except` defensive pattern.
- Handles the `Subset` shape that PR #88 produces.
- Lands after PR #88 to avoid merge churn against `scripts/train.py`.

## Out of scope (for follow-up)

- **Pixel-level mask statistics** — open follow-up issue once #72 lands. Captures actual segmentation pixel distribution per class per split (more truthful than annotation-point counts for segmentation models).
- **Temporal drift columns** (survey year, season) — `df_annotations` does not currently carry survey date; would require expanding the parquet schemas. Open as separate Data ticket.
- **`id2label_hash` in `train_summary.yaml`** — would let the MLflow UI distinguish runs that silently changed `class_subset`. Nice-to-have; defer.
- **Cross-run drift visualization** — handled via MLflow UI / downstream tooling.
