# Get the training datasets

This runbook says where each dataset in `configs/data_config.yaml` comes from, how to load it into the place that training reads, and how to add a new dataset. CoralNet has its own runbook: [Recreate and update the CoralNet data](coralnet-data.md).

- [Where each dataset lives](#where-each-dataset-lives)
- [Get the Hugging Face datasets](#get-the-hugging-face-datasets)
- [Ingest an external dataset into S3](#ingest-an-external-dataset-into-s3)
- [Publish the versioned datasets](#publish-the-versioned-datasets)
- [Update the MERMAID data](#update-the-mermaid-data)
- [Add a new dataset](#add-a-new-dataset)
- [Known gaps](#known-gaps)

## Where each dataset lives

Each dataset class reads its data from one place. The defaults are in the class's constructor in `mermaidseg/datasets/<name>/`.

| Dataset | Read from | Original source |
|---|---|---|
| `coralnet` | `s3://dev-datamermaid-sm-sources/coralnet-public-images/` and the CoralNet build | [CoralNet](https://coralnet.ucsd.edu/). See [coralnet-data.md](coralnet-data.md) |
| `mermaid` | `s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet` | The MERMAID platform. See [Update the MERMAID data](#update-the-mermaid-data) |
| `coralscapes_v2` | Hugging Face `josauder/coralscapesV2` | Same |
| `ucsd_mosaics` | Hugging Face `josauder/UCSD-mosaics-mirror` | Edwards et al. 2017, Alonso et al. 2019 and Raine et al. 2024 |
| `catlin_seaview` | `s3://dev-datamermaid-sm-sources/external_validation_datasets/catlin_seaview/` | [data.qld.edu.au/public/Q1281](http://data.qld.edu.au/public/Q1281/) (González-Rivero et al., 2019) |
| `moorea_labeled_corals` | `s3://dev-datamermaid-sm-sources/external_validation_datasets/moorea_labeled_corals/` | EDI package [knb-lter-mcr.5006](https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006). Needs a login |
| `pacific_labeled_corals` | `s3://dev-datamermaid-sm-sources/external_validation_datasets/pacific_labeled_corals/` | Dryad [doi:10.5061/dryad.m5pr3](https://datadryad.org/dataset/doi:10.5061/dryad.m5pr3) |
| `benthos_yuval` | `s3://dev-datamermaid-sm-sources/external_validation_datasets/benthos_yuval/` | Dryad [doi:10.5061/dryad.8cz8w9gm3](https://datadryad.org/dataset/doi:10.5061/dryad.8cz8w9gm3) |

The data config selects the subset of each dataset that a split uses, for example `whitelist_regions` for Catlin Seaview or `whitelist_sites` for Benthos. A split set to `null` is skipped.

## Get the Hugging Face datasets

`coralscapes_v2` and `ucsd_mosaics` download from Hugging Face the first time a dataset object is built. They go into `HF_HOME`, not into the local S3 cache. You do not load them into S3.

- For a local run, the download happens at the start of the run.
- For an offline Slurm run, `scripts/prefetch_cache.py` downloads them. Run it with `HF_HOME` set to the cache that batch jobs use. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).

To read CoralscapesV2 from a local copy, pass the directory as `hf_repo_id`. `scripts/evaluate.py` takes `--coralscapes-root` for the same purpose.

## Ingest an external dataset into S3

Catlin Seaview, Moorea Labeled Corals, Pacific Labeled Corals and Benthos are already in S3. Ingest one again only if its S3 copy is lost, or the original source publishes a new version.

`segdata ingest` in the `segdata-etl` package checks the raw download and writes the parquet, the images and the metadata to the legacy `external_validation_datasets/<name>/` prefix. Training reads that prefix. Its [README](../../packages/segdata-etl/README.md) gives the output contract and every flag. The README of each dataset lists the raw files to download:

| Dataset | Command | Raw files |
|---|---|---|
| Catlin Seaview | `catlin_seaview` | The run downloads them. See [the README](../../mermaidseg/datasets/catlin_seaview/README.md#how-to-rebuild) |
| Moorea Labeled Corals | `moorea_labeled_corals` | [Download by hand](../../mermaidseg/datasets/moorea_labeled_corals/README.md#how-to-rebuild) |
| Pacific Labeled Corals | `pacific_labeled_corals` | [Download by hand](../../mermaidseg/datasets/pacific_labeled_corals/README.md#how-to-rebuild) |
| Benthos | `benthos_yuval` | [Download by hand](../../mermaidseg/datasets/benthos_yuval/README.md#how-to-rebuild) |

1. Download the raw files that the dataset's README lists. Catlin Seaview needs no download.
2. Do a dry run. It writes nothing to S3, and it stops if the parquet or a JSON file is already there. For Catlin Seaview, leave out `--data-dir`, and the run downloads the raw files:

   ```bash
   uv run --package segdata-etl segdata ingest <dataset> --data-dir <raw dir> --dry-run
   ```

3. Run the same command without `--dry-run`. To replace a dataset that is already in S3, add `--overwrite`.
4. Fill the offline data cache again. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).
5. Publish new versions of the dataset. See [Publish the versioned datasets](#publish-the-versioned-datasets).

## Publish the versioned datasets

`segdata build <name>` stages an external dataset from its legacy prefix as an immutable versioned dataset, and `segdata taxonomy` stages a snapshot of the MERMAID benthic taxonomy that the dataset labels map to. Staging writes to a local directory only. Publishing writes to the `mermaid-ic-datasets` bucket: `segdata taxonomy --tables-from <folder> --publish` for the taxonomy, `segdata promote` for a dataset. A published version is permanent: the bucket denies overwrite and delete, so a version is write-once. Stage and check first, then publish once.

A version string is `YYYYMMDD-<sha7>`, with `-2`, `-3` for a rebuild on the same day. Each stage and publish command prints `version:` and `manifest_sha256:`. Together they are the pin that consumers use. The package README has the [flags](../../packages/segdata-etl/README.md#build-a-versioned-external-dataset), the [taxonomy options](../../packages/segdata-etl/README.md#taxonomy-snapshot) and the [promote steps](../../packages/segdata-etl/README.md#promote-a-version).

The dataset builds read the taxonomy pin from `mermaid-ic-datasets`, so publish the taxonomy before you stage a dataset.

1. Commit and push the code. Staging and publishing both refuse a dirty or unpushed tree.
2. Stage the taxonomy. This writes no S3 object:

   ```bash
   uv run --package segdata-etl segdata taxonomy --stage-dir <stage dir>
   ```

3. Review the staged tables. `<staged taxonomy folder>` is `<stage dir>/mermaid-ic-datasets/taxonomy/mermaid-benthic/<version>/`.
4. Publish the taxonomy to S3 from the staged copy, with no new fetch. The command re-checks the staged tables, publishes them under a new version name (date and git sha) and prints the pin:

   ```bash
   uv run --package segdata-etl segdata taxonomy --tables-from <staged taxonomy folder> --publish
   ```

5. Stage each dataset against the published pin. `<name>` is `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` or `benthos_yuval`. The build copies no image. It stages the tables and `copies.jsonl`, and Benthos also stages its masks and `uploads.jsonl`:

   ```bash
   uv run --package segdata-etl segdata build <name> --stage-dir <stage dir> \
     --taxonomy-version <v> --taxonomy-manifest-sha256 <sha>
   ```

   An image whose header cannot be read is dropped with its points. The count is `stats.dropped["image_unreadable_header"]` in the manifest.

6. Check that each staged version loads the same as the legacy dataset. `--no-sync` reads the new version from `--cache-dir`, the stage dir, with no S3 access for it, and still checks the sha256 of each table. The old side loads its annotations, `classes.json` and masks from S3 through `LocalS3Cache`, so the check needs S3 read access unless you pre-fill a local cache (for example with `scripts/prefetch_cache.py`) and set `MERMAIDSEG_LOCAL_CACHE_DIR=<that cache>` and `MERMAIDSEG_S3_OFFLINE=1`. A cache miss then raises instead of calling S3. The points datasets need no images. Benthos reads its staged masks.

   ```bash
   uv run python scripts/check_dataset_parity.py --no-sync \
     --cache-dir <stage dir> \
     --data-config <old data config> --dataset <name> --split <split> \
     --version <v> --manifest-sha256 <sha>
   ```

   For `pacific_labeled_corals`, add `--annotator host --annotator archived`.

   The script exits 1 on any difference. Run `--help` for the options.
7. Publish each dataset to S3. Get explicit approval first:

   ```bash
   uv run --package segdata-etl segdata promote --stage-dir <stage dir> \
     --dataset <name> --version <version>
   ```

The datasets in S3 are new versions built from the same legacy prefix. Training still reads the legacy prefix.

## Update the MERMAID data

`mermaid_confirmed_annotations.parquet` grows as MERMAID users confirm annotations. This repository does not write it. The dataset reads its images from the keys in that parquet.

`annotation_date_cutoff` in the data config freezes the MERMAID training data. The dataset uses only annotations with `updated_on` on or before that date. `MermaidDataset` has no default for it, so the argument is required. `configs/data_config.yaml` and the generated `configs/data_config_trainval.yaml` both set it to `2026-08-12`, the date when MERMAID annotations were last audited for this project.

To train on newer MERMAID annotations:

1. Set `annotation_date_cutoff` under `mermaid.train` in `configs/data_config.yaml`.
2. Regenerate the derived configs. See [Regenerate the derived configs](derived-configs.md).
3. Fill the offline data cache again.

## Add a new dataset

1. Write a dataset class in `mermaidseg/datasets/<name>/`, on the pattern of an existing class that reads the same kind of data (point annotations or dense masks).
2. Add the class to `DATASET_CLASSES` in `mermaidseg/datasets/factory.py`. If the dataset has dense masks, also add it to `NO_PADDING_DATASETS`.
3. Map its labels into the MERMAID benthic-attribute space. Add a `fetch_<name>_to_mermaid` function to `mermaidseg/dataset_reconciliation/label_mapping.py`, and register it in `_REMOTE_FETCHERS` in `mermaidseg/dataset_reconciliation/registry.py`. If the dataset is scored in evaluation, also add it to `mermaidseg/evaluation/gt_mapping.py`.
4. Add a row for each label to `configs/class_to_concepts.csv`, with the dataset in `source_dataset_source` and the label in `source_label_class_name`.
5. Add the dataset and its splits to `configs/data_config.yaml`.
6. Regenerate the derived configs. See [Regenerate the derived configs](derived-configs.md).
7. Fill the offline data cache again.

## Known gaps

- The MERMAID parquet changes in place. A run that uses the same cutoff reads the same annotations only if no annotation older than the cutoff changes after the run.
