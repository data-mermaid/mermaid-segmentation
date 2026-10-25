# CoralNet dataset

PyTorch dataset and supporting code for the CoralNet source dataset.

The dataset class ([`CoralNetDataset`](coralnet_dataset.py)) emits annotation
masks in the **CoralNet provider label space** (i.e. CoralNet `Label ID` /
`provider_id` values). Mapping into the unified target MERMAID benthic
attribute space is done at training time by the
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py) and
[`fetch_coralnet_to_mermaid`](../../dataset_reconciliation/label_mapping.py).

The dataset class reads from a self-contained training manifest parquet
(currently:
`s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/20260623_nogit/coralnet_training_manifest_20260623_nogit.parquet`),
which carries per-annotation rows plus the resolved ``image_s3_key`` for each
image. Annotations and image keys are derived from this single file.

## Layout

- `coralnet_dataset.py` — `CoralNetDataset` class.
- `etl/` — reproducible, tested ETL that builds the annotations + image-metadata
  parquets ([CoralNet-ETL](../../../wiki/CoralNet-ETL.md)).
- `scraper/` — scraping utilities and CSV artifacts originally located under
  the repo-root `scraping/` directory. `scraper/source_metadata.py` fetches
  source names, admins and locations for the dedup split.
- `dedup/` — removes images uploaded to several sources, merges their
  annotations and builds leak-free train/val/test splits (see below).
- `nbs/` — exploration and annotation-aggregation notebooks for the CoralNet
  sources. The `CoralNet_Annotations.ipynb` notebook has been superseded by
  the `etl/` module; treat it as historical reference only.
  `Dedup_Review.ipynb` visualises what the deduplication did to each image.

## Description

CoralNet (https://coralnet.ucsd.edu/) is an online repository of annotated coral reef imagery and expert annotations. It provides a large-scale source of coral reef images with point annotations for benthic attributes structured around sources. A source typically corresponds to a single research project, survey, institution, field site, camera system, or contributor that supplied imagery and metadata to CoralNet. Sources group images that share common collection conditions (location, time period, photographer or instrument, and sampling protocol). Specific sources also contain metadata including the location, datetime and depth.

- Annotations in CoralNet are usually acquired through random sampling or through human labelling across the image (which also result in randomly spaced annotations). The annotations dataframe, contains the label for a specific row & column of a specific image from a specific source. The CoralNet classes are mapped to MERMAID classes using the LabelMapping API endpoint.
- We apply padding to the annotations, with the assumption that for a specific point (pixel being) assigned to a class, the neighbouring pixels are very likely to also be in that class as these labels come either from a image classification approach that makes a prediction based on a image crop around the point, or a manual annotation, both of which are most likely not precise to a pixel level.

## Pointing CoralNetDataset at a training manifest

The authoritative training manifest is a self-contained parquet with
per-annotation rows and resolved image S3 keys. Pin a specific build via
``MERMAID_CORALNET_MANIFEST_VERSION`` (e.g. ``20260623_nogit``) or override
the full S3 key with ``MERMAID_CORALNET_MANIFEST_PATH``. With neither set,
``CoralNetDataset`` falls back to the default published manifest under
``etl-outputs/coralnet/``. See
[CoralNet-ETL](../../../wiki/CoralNet-ETL.md) for ETL usage and runtimes.

## Duplicate images and the dedup splits

CoralNet gives every upload a new `image_id`, so the same photo uploaded to
several sources looks like different images. With the old source-level splits
about half of the val/test images were exact copies of train images. The
`dedup/` package finds copies by content hash, keeps one copy per photo (from
the best-ranked source of its cluster), merges the annotations of all copies
onto it (finest consistent label wins; conflicts go to manual-review files),
and splits the result so val/test only use sources that share no image with
another source. Run the steps in order:

```bash
python -m mermaidseg.datasets.coralnet.dedup.hashing --all-sources   # ~35 min
python -m mermaidseg.datasets.coralnet.dedup.conflicts
python -m mermaidseg.datasets.coralnet.dedup.dedup
python -m mermaidseg.datasets.coralnet.scraper.source_metadata        # once, ~1 h
python -m mermaidseg.datasets.coralnet.dedup.splits                   # writes configs/data_config_dedup.yaml
python -m mermaidseg.datasets.coralnet.dedup.leakage --config configs/data_config_dedup.yaml
```

Versioned results and curated inputs live in `configs/coralnet/`:
`image_blacklist.parquet` and `annotation_overrides.parquet` (read by
`CoralNetDataset` through the data config), `source_splits.csv`,
`low_quality_sources.txt` (sources excluded entirely) and
`mermaid_benthic_hierarchy.json` (a snapshot of the MERMAID API's benthic
attribute hierarchy, used to roll labels up to the training classes when
balancing the split). Intermediate outputs (hashes, pair lists, provenance,
review files) go to `data/coralnet_dedup/`, which is not versioned.

`CoralNetDataset` resolves a relative `image_blacklist` /
`annotation_overrides` reference against the repo root when that file exists,
so `configs/coralnet/...` works from any working directory; otherwise the
reference is an S3 key, as for the manifest.

## Incomplete S3 folders and CoralNet remediation

When the audit parquet shows `not is_complete` for some `source_id` values,
export them (for example via `incomplete_df.parquet` in the EDA notebook), then:

1. `uv run coralnet-remediate probe --audit-parquet … --incomplete-parquet …`
   merges audit columns with a live CoralNet probe and writes
   `reconciliation_report.parquet`.
2. `uv run coralnet-remediate redownload --report reconciliation_report.parquet`
   applies categorized fixes (omit `--dry-run` for real uploads). Sources whose
   overview shows **Confirmed: 0** (parsed from the Image Status table) get
   `skip_no_confirmed_annotations` and are not redownloaded. Optionally use
   `--clean-prefix` plus `CORALNET_*` credentials in the environment
   (`CORALNET_USERNAME` / `CORALNET_PASSWORD`).
3. Re-audit only touched IDs:
   `uv run coralnet-etl audit --source-ids-file incomplete_df.parquet …`.

Bulk rescrape of newly listed public sources:
Sources with **Confirmed: 0** on the overview page are skipped at download time (same rule as remediation).

The annotations parquet built by `coralnet-etl build-annotations` keeps **Confirmed** rows only.

Local image smoke test for a single source (writes JPEGs under `outputs/`; public sources
such as 7238 do not require CoralNet credentials):

```sh
uv run python -c "
from pathlib import Path
from mermaidseg.datasets.coralnet.scraper.downloader import CoralNetDownloader
dest = Path('outputs/coralnet-source-7238-test/s7238')
n_ok, n_total, img_dir = CoralNetDownloader('', '').download_source_images_local(7238, dest)
print(n_ok, n_total, img_dir)
"
```
`python -m mermaidseg.datasets.coralnet.scraper.scrape_coralnet_s3`
(supports `--audit-parquet`, `--force`, `--no-legacy-skip-annotations`; see `-h`).

**Offline tests** (CI default):

```sh
uv run pytest tests/datasets/coralnet/ -q
```

**Live validation** (Playwright against coralnet.ucsd.edu; not run in CI):

```sh
uv run sync --extra test-live
uv run playwright install chromium
uv run --extra test-live pytest tests/datasets/coralnet/test_live_smoke.py -m live -v
```

Optional: refresh static HTML fixtures when CoralNet markup changes:

```sh
uv run --extra test-live pytest tests/datasets/coralnet/test_live_smoke.py -m live --refresh-fixtures
```

The export-form live test requires `CORALNET_USERNAME` / `CORALNET_PASSWORD`
(use single quotes if the password contains shell metacharacters like `&` or `#`):

```sh
export CORALNET_USERNAME='you@example.com'
export CORALNET_PASSWORD='your-password'

# Quick login check (requests, same path as the scraper)
uv run python -c "
import os, sys
from mermaidseg.datasets.coralnet.scraper.downloader import CoralNetDownloader
u, p = os.environ.get('CORALNET_USERNAME'), os.environ.get('CORALNET_PASSWORD')
if not u or not p:
    sys.exit('Set CORALNET_USERNAME and CORALNET_PASSWORD')
sys.exit(0 if CoralNetDownloader(u, p).login() else 1)
"
```

Override the default public source (`23`) with `CORALNET_LIVE_SOURCE_ID` if needed.
