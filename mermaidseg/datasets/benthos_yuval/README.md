# Benthos Yuval

PyTorch dataset and ingestion script for the **Benthos** photomosaic
segmentation dataset
([`doi:10.5061/dryad.8cz8w9gm3`](https://datadryad.org/dataset/doi:10.5061/dryad.8cz8w9gm3)).

> Yuval, M., Alonso, I., Eyal, G., Tchernov, D., Loya, Y., Murillo, A. C.,
> Treibitz, T. (2021). *Repeatable Semantic Reef-Mapping through
> Photogrammetry and Label-Augmentation.* Remote Sensing 13(4):659.
> [doi:10.3390/rs13040659](https://doi.org/10.3390/rs13040659)

## Dataset

The paper reconstructs an orthorectified photomosaic of a reef from a
Structure-from-Motion (SfM) survey, then propagates sparse manual labels
via super-pixel-based *label-augmentation* into dense, full-coverage
semantic masks. The released **Benthos** dataset ships the resulting
fully-labelled photomosaics for three sites:

| Code | Region                 | Site                                                              | Depth (m) | Approx. surveyed area |
| ---- | ---------------------- | ----------------------------------------------------------------- | --------- | --------------------- |
| RS   | Northern Red Sea       | IUI Eilat reef, Gulf of Aqaba, Israel                             | 24-28     | 5x5 m                 |
| CR   | Eastern Caribbean      | Double-Wreck reef, St. Eustatius                                  | 20        | 12x2 m                |
| MD   | Eastern Mediterranean  | Spartan Reef offshore Haifa, Israel                               | 20        | 5x4 m                 |

Manual full-segmentation labels were drawn in Labelbox over **eight
terrain-type classes** (`Algae`, `Calcareous Algae`, `Coral`, `Other`,
`Rock`, `Sand`, `Soft Coral`, `Sponge`); pixels outside the labelled ROI
are `unlabeled`. We **drop MD/Spartan** (rocky Mediterranean, no
scleractinian coral community) and keep only `RS24` and `CR_DoubleWreck`.

### Files we use

We use 6 of the ~500 files in the Dryad bundle:

| File                       | Used for                                                |
| -------------------------- | ------------------------------------------------------- |
| `RS24.png`                 | RGB photomosaic of the Red Sea site (`RS24`).           |
| `ManualRS24.png`           | Dense uint8 label mask aligned with `RS24.png`.         |
| `CR_DoubleWreck.png`       | RGB photomosaic of the Caribbean site (`CR_DoubleWreck`). |
| `ManualDW.png`             | Dense uint8 label mask aligned with `CR_DoubleWreck.png`. |
| `dictionary_labels.txt`    | `{class_name: raw_mask_id}` for the `RS24` mask.        |
| `dictionary_labelsDW.txt`  | `{class_name: raw_mask_id}` for the `CR_DoubleWreck` mask. |

### Photomosaic dimensions

| Site             | Image + mask size                  |
| ---------------- | ---------------------------------- |
| `RS24`           | `11070x11786px` (image and mask)   |
| `CR_DoubleWreck` | `24192x3982px`  (image and mask)   |

### Per-site pixel composition (after class remap)

`unlabeled` is reported separately so the percentages of labelled
classes sum to `100 - unlabeled%`.

**RS24:**

| Class            | %      |
| ---------------- | ------ |
| Sand             | 40.07  |
| Rock             | 35.72  |
| Coral            | 13.13  |
| Sponge           |  3.22  |
| _unlabeled_      |  2.52  |
| Other            |  2.18  |
| Soft Coral       |  1.78  |
| Algae            |  0.86  |
| Calcareous Algae |  0.51  |

**CR_DoubleWreck:**

| Class            | %      |
| ---------------- | ------ |
| Algae            | 43.71  |
| Sand             | 11.87  |
| Sponge           | 10.92  |
| Rock             |  8.62  |
| Coral            |  8.47  |
| _unlabeled_      |  7.55  |
| Soft Coral       |  4.32  |
| Calcareous Algae |  2.90  |
| Other            |  1.64  |

## Dense vs. point annotations

Unlike the other source datasets in this repo (CoralNet / MERMAID / Catlin
Seaview / Moorea Labeled Corals), Benthos Yuval has **dense annotations**.
The S3-side parquet therefore has no `(row, col)` columns; instead it
carries one row per `(site, image_id, source_label_name)` tile-class pair,
and `BenthosYuvalCoralsDataset.__getitem__` reads the dense PNG label tile
and remaps its `classes.json` IDs through a small lookup vector into the
local source-id space expected by `BaseCoralDataset`.

## Processing performed by [`verify_raw_data_and_add_to_s3.py`](verify_raw_data_and_add_to_s3.py)

The Dryad files are behind an OAuth-protected download endpoint, so this
script does **not** auto-download anything. Manually download the 6 files
listed above into one flat directory, then run:

```bash
python -m mermaidseg.datasets.benthos_yuval.verify_raw_data_and_add_to_s3 \
    --data-dir /path/to/benthos_yuval_downloaded \
    --bucket dev-datamermaid-sm-sources \
    --prefix external_validation_datasets/benthos_yuval
```

### 1. Integrity check

- Confirm all 6 required files are present under `--data-dir`.
- Parse the `dictionary_labels{,DW}.txt` files via `ast.literal_eval`.
- Each photomosaic + mask is fully PIL-decoded once during tiling (step 3);
  any decode error aborts before any S3 write.

### 2. Build `classes.json` / `colors.json`

- Alphabetical union of class names across both per-site dictionaries.
- `classes.json` is `{"unlabeled": 0, "<class_1>": 1, ..., "<class_N>": N}`.
- `colors.json` carries a deterministic RGB palette
  (`np.random.default_rng(seed=1337)`) with `"unlabeled" -> [0, 0, 0]`.

### 3. Tile each photomosaic into 2048x2048 PNG image+label pairs

**Tile geometry: stride == tile size (no overlap, no padding).** Each
photomosaic is iterated in row-major order with step `TILE = 2048`:

```
for y0 in range(0, H, 2048):
    for x0 in range(0, W, 2048):
        y1 = min(y0 + 2048, H)
        x1 = min(x0 + 2048, W)
        # write image[y0:y1, x0:x1] and mask[y0:y1, x0:x1] as <y0>_<x0>.png
```

Two consequences worth knowing:

1. **The last tile in each row and column may be smaller than 2048 along
   one or both axes.** They are *not* padded out to 2048 — they are saved
   as-is at their natural (`y1 - y0`, `x1 - x0`) shape. Downstream training
   code therefore needs to either resize tiles to a fixed shape inside the
   Albumentations transform, or accept variable-size tiles.

2. **Tile filenames encode the top-left pixel of the tile in the source
   mosaic**: `images/<site>/<y0>_<x0>.png` and `labels/<site>/<y0>_<x0>.png`.
   That is also the `image_id` used by `BenthosYuvalCoralsDataset.df_images`.

While tiling, the dense uint8 mask is remapped from the **per-site raw-id
space** (as given by `dictionary_labels{,DW}.txt`) into the **global
`classes.json` ID space** (alphabetical, shared across both sites), so a
single shared `classes.json` describes every mask tile written to S3.

### 4. Build the annotations Parquet

`benthos_yuval_annotations.parquet` has one row per
`(site, image_id, source_label_name)`, with `tile_height` / `tile_width`
columns recording the actual saved tile shape (useful for spotting which
tiles are the partial-edge ones):

| Column              | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `site`              | `"RS24"` or `"CR_DoubleWreck"`.                            |
| `image_id`          | `<y0>_<x0>` of the tile within the source mosaic.          |
| `source_label_name` | One of the alphabetical Benthos class names.               |
| `tile_height`       | Saved tile height in pixels (<=2048; may be smaller at the bottom edge). |
| `tile_width`        | Saved tile width in pixels (<=2048; may be smaller at the right edge).  |

### 5. Upload to S3

Layout under `s3://<bucket>/<prefix>/`:

```
benthos_yuval_annotations.parquet
classes.json
colors.json
manifest.json
images/<site>/<image_id>.png
labels/<site>/<image_id>.png
```

A `manifest.json` records bucket layout, per-site tile counts, and the
`excluded_sites: ["MD_spartan"]` decision so dropping the Mediterranean
site is auditable.

## Using the dataset class

```python
from mermaidseg.datasets.benthos_yuval import BenthosYuvalCoralsDataset

ds = BenthosYuvalCoralsDataset(
    source_bucket="dev-datamermaid-sm-sources",
    source_s3_prefix="external_validation_datasets/benthos_yuval",
    annotations_path="external_validation_datasets/benthos_yuval/benthos_yuval_annotations.parquet",
)
image, source_mask = ds[0]   # (H, W, 3) uint8 image, (H, W) int64 mask in local source-id space
```

`SOURCE_NAME = "benthos_yuval"` so the dataset plugs straight into
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py); the
provider-side source-to-MERMAID mapping is fetched from
`https://api.datamermaid.org/v1/classification/labelmappings/?provider=Benthos%20Yuval`
via
[`fetch_benthos_yuval_to_mermaid()`](../../dataset_reconciliation/label_mapping.py).
Until that provider is populated on the MERMAID side, the mapping is empty
and every Benthos label collapses to background at training time (same
fallback as `moorea_labeled_corals`).

## Layout

```
benthos_yuval/
├── README.md                              # this file
├── __init__.py                            # re-exports BenthosYuvalCoralsDataset
├── benthos_yuval_corals_dataset.py        # PyTorch dataset (dense PNG mask reader)
├── verify_raw_data_and_add_to_s3.py       # one-shot ingestion script (manual download required)
└── nbs/
    └── benthos_yuval_EDA.ipynb            # post-ingestion EDA against the S3-hosted dataset
```
