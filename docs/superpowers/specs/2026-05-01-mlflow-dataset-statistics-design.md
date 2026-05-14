# MLflow Dataset Statistics — Design

**Issue:** #72
**Status:** Design
**Date:** 2026-05-01

## Background

Stakeholders want per-run dataset statistics in MLflow for traceability, drift detection, and run-to-run QA. Pattern mirrors the `mermaid-classifier` repo's `ba_counts.csv` / `project_stats.csv` / `train_summary.yaml` precedent, scoped to the post-PR-#96 dataset architecture.

Pixel-level mask statistics are explicitly **out of scope**; defer to a follow-up once materialization cost is benchmarked.

## Architecture context (post PR #96)

- Datasets emit `(image, source_labels)` in **source-label space** (per-dataset vocab).
- `SourceLabelRegistry` assigns global offsets and exposes `target_id2label`, `source_to_target` (LongTensor of shape `(num_global_source + 1,)`), `dataset_offsets`, `concept_id2name`.
- `CombinedCoralDataset` wraps multiple registered datasets via `ConcatDataset` and exposes `_datasets`.
- Common columns:
  - `df_annotations`: `image_id`, `row`, `col`, `source_label_name` (always); plus `region_id`/`region_name` (mermaid) or `source_id` (coralnet).
  - `df_images`: `image_id` plus the same source-key column.

## API

```python
def log_dataset_statistics(
    self,
    splits: dict[str, "Dataset | Subset"],
    registry: "SourceLabelRegistry",
    *,
    artifact_dir: str = "dataset_stats",
) -> None
```

The registry is **required** because target-space class identities (the headline `class_counts.csv` axis) live there, not on individual datasets.

## Artifacts

All artifacts live under `dataset_stats/`.

### `class_counts.csv` — target-space per-class × split counts

Columns:

| Column | Type | Notes |
|---|---|---|
| `target_id` | int | from `registry.target_id2label` plus implicit `0` (background) |
| `target_name` | str | `"background"` for id 0; otherwise `registry.target_id2label[id]` |
| `class_kind` | str | `target` / `background` / `unclassified` (see policy below) |
| `train_annotations` | int | annotation rows mapped to this target id |
| `val_annotations` | int | |
| `test_annotations` | int | |
| `train_images` | int | unique images containing annotations of this target id |
| `val_images` | int | |
| `test_images` | int | |
| `train_fraction` | float | `train_annotations / sum across all rows for that split` |
| `val_fraction` | float | |
| `test_fraction` | float | |

Mapping from `source_label_name` → `target_id` goes through:
1. Look up local source id via `dataset.source_name2id[name]`.
2. Apply `dataset.global_offset` to get the global source id.
3. Look up target id via `registry.source_to_target[global_id]`.

`class_kind` policy (case-insensitive `target_name`):
- `id 0` → `background`
- name lowercased is exactly `"other"`, `"unknown"`, `"unclassified"` → `unclassified`
  (qualified names like `"Other Invertebrates"` remain `target`)
- otherwise → `target`

### `source_stats.csv` — per region/source × split image and annotation counts

| Column | Type | Notes |
|---|---|---|
| `source_key` | str | `region_name` (mermaid) or `source_id` cast to str (coralnet) |
| `source_type` | str | `region` or `source` |
| `train_images` | int | |
| `val_images` | int | |
| `test_images` | int | |
| `train_annotations` | int | |
| `val_annotations` | int | |
| `test_annotations` | int | |

### `class_by_source.csv` — long-format source × class × split

| Column | Type |
|---|---|
| `source_key` | str |
| `source_type` | str |
| `target_id` | int |
| `target_name` | str |
| `split` | str |
| `annotations` | int |
| `images` | int |

Zero-count rows are omitted.

### `train_summary.yaml`

```yaml
total_images: int
total_annotations: int
splits:
  train: { images: int, annotations: int }
  val:   { images: int, annotations: int }
  test:  { images: int, annotations: int }
num_target_classes: int          # registry.num_target_classes (includes background)
eligible_num_classes: int        # target_id2label entries with class_kind == target
top1_share: float                # share over the training split, eligible classes only
top3_share: float                # clamps to total share if < K eligible classes
top5_share: float
effective_num_classes: float     # exp(entropy) over the eligible training class distribution
annotations_per_image:
  train: { mean: float, median: float, p10: float, p90: float, min: int, max: int }
  val:   { mean: float, median: float, p10: float, p90: float, min: int, max: int }
  test:  { mean: float, median: float, p10: float, p90: float, min: int, max: int }
```

## Resolution helper

`resolve_split_annotations(split, registry)` returns `(df_annotations, df_images, source_to_target_dict)` or `None`. It handles:

- Plain dataset with `df_annotations`, `df_images`, `source_name2id`, `global_offset` → returns the dataset's frames plus a per-name → target_id dict computed from registry.
- PyTorch `Subset` → filters parent frames by subset image ids, then resolves through parent.
- `CombinedCoralDataset` (exposes `_datasets`) or anything else with `_datasets`/`datasets` → resolves each child against the registry, concatenates frames, **adds a column** `_target_id` per child so each row carries its mapped target id (necessary because different children have different `global_offset`).
- Unknown shape → returns `None`, logs a warning.

In all cases, the resolved annotation frame has a `target_id` column added (mapped via registry), so downstream helpers do not need to know about offsets.

## Error handling

- MLflow not active → early return.
- Per-split resolution fails → log warning, skip that split, continue.
- Per-artifact write fails → log warning, continue with remaining artifacts.
- No outer try/except — per-split and per-artifact isolation are sufficient.

## Out of scope (follow-ups)

- Pixel-level mask statistics.
- Concept-level QA (separate CBM workstream).
- Temporal columns (require parquet schema expansion).
- `id2label_hash` for run-comparison.
