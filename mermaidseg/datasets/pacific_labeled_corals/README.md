# Pacific Labeled Corals

PyTorch dataset and ingestion script for the **Pacific Labeled Corals**
random-point coral-reef survey dataset
([`doi:10.5061/dryad.m5pr3`](https://datadryad.org/dataset/doi:10.5061/dryad.m5pr3)).

> Beijbom, O., Edmunds, P. J., Roelfsema, C., Smith, J., Kline, D. I.,
> Neal, B. P., Dunlap, M. J., Moriarty, V., Fan, T.-Y., Tan, C.-J.,
> Chan, S., Treibitz, T., Gamst, A., Mitchell, B. G., Kriegman, D.
> (2015). *Towards Automated Annotation of Benthic Survey Images:
> Variability of Human Experts and Operational Modes of Automation.*
> PLOS ONE 10(7): e0130312.
> [doi:10.1371/journal.pone.0130312](https://doi.org/10.1371/journal.pone.0130312)

## Dataset

The Beijbom et al. (2015) release bundles random-point photoquadrat
surveys from four Pacific coral-reef monitoring projects, all annotated
against a single **consensus 20-class label-set**:

| Code           | Region                          | Reef type                                      | Original local annotator |
| -------------- | ------------------------------- | ---------------------------------------------- | ------------------------ |
| `heron_reef`   | Heron Reef, Great Barrier Reef  | Platform reef, fore reef (5 m) / reef flat (1 m) | C. Roelfsema             |
| `line_islands` | Northern Line Islands           | Fore reef (10 / 17 m), reef flat (2-5 m)       | J. Smith                 |
| `nanwan_bay`   | Nanwan Bay, Taiwan              | Fringing reef, fore reef (2-5 m)               | T.-Y. Fan                |
| `moorea`       | Moorea, French Polynesia (MCR-LTER) | Fringing reef (depth 2-5 m)                | P. J. Edmunds            |

> **We drop the Moorea split** — the same images live in the larger
> MCR-LTER release (`knb-lter-mcr.5006`) processed independently in
> [`mermaidseg/datasets/moorea_labeled_corals`](../moorea_labeled_corals).
> Only `heron_reef`, `line_islands`, and `nanwan_bay` are kept here.

### Subsets and annotators

Each kept site is split into two subsets:

* **`reference`** — the bulk of the survey images (471 - 2,597 images per
  site, 24 - 200 random points per image). Carries only the **`archived`**
  annotation: the label assigned by the local coral expert during the
  original ecological survey.
* **`evaluation`** — exactly 200 images per site at 10 random points per
  image. Re-annotated by 6 coral experts during the Beijbom et al. study,
  so each point carries 7 label columns:

  | Column     | Annotator                                                                                    |
  | ---------- | -------------------------------------------------------------------------------------------- |
  | `archived` | Original ecological-survey label (same person as `host`, but typically 1-6 years earlier).   |
  | `host`     | The local coral expert re-annotating their own images for this study.                        |
  | `visitor1`-`visitor5` | Five visiting coral experts not familiar with the local ecology.                  |

### Per-site stats

| Place          | Images | Annotations (points) | Distinct label classes present |
| -------------- | ------ | -------------------- | ------------------------------ |
| `heron_reef`   | 2,797  | 64,328               | 5                              |
| `line_islands` | 732    | 55,200               | 20                             |
| `nanwan_bay`   | 890    | 36,260               | 20                             |
| **TOTAL**      | 4,419  | 155,788              | 20                             |

The 5-vs.-20 asymmetry on Heron Reef is intentional (Beijbom et al.):
*"Corals were not resolved to genus level in the original Heron Reef
label-set and all `Archived` coral annotations were therefore mapped to
the generic 'other scleractinians' label for this location."*

The full Dryad release decodes cleanly — there are **no damaged or
missing images** under our integrity check.

## Ingestion ([`verify_raw_data_and_add_to_s3.py`](verify_raw_data_and_add_to_s3.py))

The Dryad files are gated behind a portal with no stable download API,
so this script does **not** auto-download. Manually download the package
from
[`doi:10.5061/dryad.m5pr3`](https://datadryad.org/dataset/doi:10.5061/dryad.m5pr3)
and lay it out as below, then run:

```bash
python -m mermaidseg.datasets.pacific_labeled_corals.verify_raw_data_and_add_to_s3 \
    --data-dir /path/to/pacific_labeled_corals_downloaded \
    --bucket dev-datamermaid-sm-sources \
    --prefix external_validation_datasets/pacific_labeled_corals
```

Expected `--data-dir` layout:

```
/path/to/pacific_labeled_corals_downloaded/
    heron_reef/
        labelmap.txt                                 # labelid,name
        reference/
            imagemap.txt                             # imageid,imagefile
            annotations.txt                          # imageid,row,col,archived
            imgs/<file>.JPG
        evaluation/
            imagemap.txt
            annotations.txt                          # imageid,row,col,archived,host,visitor1,...,visitor5
            imgs/<file>.JPG
    line_islands/...
    nanwan_bay/...
```

### 1. Integrity check

- All required `<site>/labelmap.txt`, `<site>/<subset>/imagemap.txt`,
  `<site>/<subset>/annotations.txt`, and `<site>/<subset>/imgs/`
  paths must exist.
- Every site's `labelmap.txt` must be byte-equivalent (consensus
  label-set is shared across sites).
- Every image referenced by an `imagemap.txt` is fully PIL-decoded
  in parallel before any S3 write; **any decode failure aborts the run.**

### 2. Build `classes.json` / `colors.json`

- `classes.json` is `{"unlabeled": 0, "<label_1>": 1, ..., "<label_N>": N}`
  with the per-site `labelmap.txt` names sorted alphabetically.
- `colors.json` carries a deterministic RGB palette
  (`np.random.default_rng(seed=1337)`) keyed by label name, with
  `"unlabeled" -> [0, 0, 0]`.

### 3. Build the annotations Parquet

`pacific_labeled_corals_annotations.parquet` is a single PyArrow file
with one row per `(site, subset, raw_imageid, row, col)`:

| Column                                    | Description                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| `site`                                    | `"heron_reef"` / `"line_islands"` / `"nanwan_bay"`.                          |
| `subset`                                  | `"reference"` or `"evaluation"`.                                             |
| `raw_imageid`                             | Original integer `imageid` from `imagemap.txt` (kept for traceability).      |
| `image_id`                                | Filename stem of the source jpg (string).                                    |
| `image_ext`                               | Filename suffix incl. dot, e.g. `".JPG"`.                                    |
| `row`, `col`                              | Annotation pixel coordinates (integer).                                      |
| `archived`                                | Consensus label from the original survey. **Always populated.**              |
| `host`                                    | Same expert re-annotating; populated only for `evaluation`, else `pd.NA`.    |
| `visitor1` ... `visitor5`                 | Five visiting experts; populated only for `evaluation`, else `pd.NA`.        |

Per-annotator label IDs are mapped through the shared `labelmap.txt` to
their string names. Rows whose `archived` ID is not in `labelmap.txt`
are dropped (the original notebook's behaviour for `archived`).

### 4. Upload to S3

Layout under `s3://<bucket>/<prefix>/`:

```
pacific_labeled_corals_annotations.parquet
classes.json
colors.json
manifest.json
images/<site>/<subset>/<image_id><image_ext>
```

`manifest.json` records the bucket layout, parquet schema, source
citation, and the `excluded_sites: ["moorea"]` decision so dropping the
Moorea split is auditable.

## Using the dataset class

```python
from mermaidseg.datasets.pacific_labeled_corals import PacificLabeledCoralsDataset

ds = PacificLabeledCoralsDataset(
    source_bucket="dev-datamermaid-sm-sources",
    source_s3_prefix="external_validation_datasets/pacific_labeled_corals",
    annotations_path="external_validation_datasets/pacific_labeled_corals/pacific_labeled_corals_annotations.parquet",
)
image, source_mask = ds[0]   # (H, W, 3) uint8 image, (H, W) int64 mask in local source-id space
```

### Annotator selection

Each evaluation point carries 7 alternative labels; the dataset class
picks one to expose as `source_label_name`:

- `annotator_column: str = "host"` — re-annotation by the same coral
  expert; typically the cleanest single-annotator stream. Allowed:
  `archived | host | visitor1 | ... | visitor5`.
- `fallback_annotator: str | None = "archived"` — used when
  `annotator_column` is null. The default keeps the reference subsets
  loaded under the chosen annotator's spelling. Pass
  `fallback_annotator=None` to drop those rows (reducing the dataset to
  the 600 evaluation images).

The actually-used column is recorded per row in
`ds.df_annotations["annotator_used"]`.

You can also restrict to a subset of sites or subsets via
`whitelist_sites` / `blacklist_sites` and `whitelist_subsets` /
`blacklist_subsets`.

### Reconciliation with MERMAID

`SOURCE_NAME = "pacific_labeled_corals"` so the dataset plugs into
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py); the
provider-side source-to-MERMAID mapping is fetched from
`https://api.datamermaid.org/v1/classification/labelmappings/?provider=Pacific%20Labeled%20Corals`
via [`fetch_pacific_labeled_corals_to_mermaid()`](../../dataset_reconciliation/label_mapping.py).
Until that provider is populated on the MERMAID side, the mapping is
empty and every Pacific Labeled Corals label collapses to background at
training time.

## Layout

```
pacific_labeled_corals/
├── README.md                              # this file
├── __init__.py                            # re-exports PacificLabeledCoralsDataset
├── pacific_labeled_corals_dataset.py      # PyTorch dataset (sparse-point reader)
├── verify_raw_data_and_add_to_s3.py       # one-shot ingestion script (manual download required)
└── nbs/
    └── pacific_labeled_corals_EDA.ipynb   # post-ingestion EDA against the S3-hosted dataset
```
