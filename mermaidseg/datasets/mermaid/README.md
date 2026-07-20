# MERMAID dataset

PyTorch dataset and supporting code for the MERMAID source dataset.

The dataset class ([`MermaidDataset`](mermaid_dataset.py)) emits annotation
masks in the **MERMAID benthic attribute label space**. The MERMAID label
space is the canonical target space used by
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py); the
`source -> target` mapping for MERMAID is therefore the identity (minus the
global offset assigned by the registry).

The dataset class reads the annotations from the a parquet file (s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet), from which the images are also derived.

## Layout

- `mermaid_dataset.py` — `MermaidDataset` class.
- `nbs/` — exploration notebooks for MERMAID-specific data analysis.

## Description

The MERMAID dataset is a continuously growing dataset with images uploaded by users of the MERMAID platform (https://datamermaid.org/).
- The annotations of each image are done in a systematic approach, such that 25 points are taken in a 5x5 grid across the image. The annotations dataframe, contains the label for a specific row & column of a specific image, as well as the MEOW region it belongs to.
- We apply padding to the annotations, with the assumption that for a specific point (pixel being) assigned to a class, the neighbouring pixels are very likely to also be in that class as these labels come either from a image classification approach that makes a prediction based on a image crop around the point, or a manual annotation, both of which are most likely not precise to a pixel level.

## Train / val split

By default, `MermaidDataset` loads **all regions** and applies the rare-aware holdout with `holdout_fraction=0.1`, `holdout_seed=42`, and `holdout_role="train"`. Construct a second instance with `holdout_role="val"` (same fraction and seed) for the validation complement. Pass `holdout_fraction=None` and `holdout_role=None` to load the full corpus without splitting.

When holdout is enabled, whole images are assigned to train or val using a **two-pass rare-aware region-blocked holdout** (`select_stratified_holdout_image_ids`).

### Per-image label sets

For each image, holdout uses the full set of labels across its 25 annotation points (not the mode):

- `region_name` — `region_name` with `region_id` fallback
- `benthic_attribute_names` — distinct `benthic_attribute_name` values on the image
- `growth_form_names` — distinct non-null / non-sentinel `growth_form_name` values (nulls and `"None"` are excluded from growth-form coverage)

### Pass 1 — rare coverage

A `(region_name, label)` pair is **rare** when it appears on **2–9 images** (`RARE_IMAGE_THRESHOLD = 10`). Pairs with only one image stay train-only.

Pass 1 greedily assigns val images so each rare `(region_name, benthic_attribute_name)` and each rare `(region_name, growth_form_name)` has at least one val image containing that label in that region. An image counts if the label appears on any of its 25 points.

### Pass 2 — regional quota

Per region, the target val count is `max(1, round(n_images * holdout_fraction))`, capped at `n - 1`. Remaining slots after Pass 1 are filled by proportional sampling over the image’s **dominant** `benthic_attribute_name` (highest point count on the image).

Train and val splits are complements when they share the same `holdout_fraction` and `holdout_seed`. `select_composite_holdout_image_ids` remains available for before/after comparisons (legacy mode-based composite strata). See `nbs/datasets/Mermaid_Snapshot.ipynb` for corpus and split diagnostics.

## Usage

Three invocation patterns:

**Training (via `scripts/train.py` or config-driven notebooks)** — two instances, complementary roles:

```python
train_ds = MermaidDataset(holdout_role="train", transform=..., padding=...)
val_ds = MermaidDataset(holdout_role="val", transform=..., padding=...)
```

Or rely on class defaults for train and pass only `holdout_role="val"` for validation. `holdout_fraction` and `holdout_seed` default to `0.1` and `42`.

**Exploration / custom splits** — disable holdout so the class loads the full corpus:

```python
MermaidDataset(holdout_fraction=None, holdout_role=None, transform=..., padding=...)
```

Use this before ad-hoc splits (e.g. `random_split`) or when analyzing the full parquet in a notebook.

**Split diagnostics only** — call `select_stratified_holdout_image_ids` on a pandas frame from parquet, without constructing `MermaidDataset`. See `nbs/datasets/Mermaid_Snapshot.ipynb`.

When holdout is enabled, `MermaidDataset.split` is set to the holdout role (`"train"` or `"val"`).

## Note

As the mermaid_confirmed_annotations.parquet is continuously being updated, each run currently might have slightly different results due to changes in the (number of) images. As a solution to this, we can potentially save occasional copies of the file (e.g. at the end of every month).
