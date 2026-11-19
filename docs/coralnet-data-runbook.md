# Recreate and update the CoralNet data

This runbook takes you from an empty bucket, or an older scrape, to a CoralNet build that training reads. It has two stages:

1. `coralnet-scraper` downloads the public CoralNet sources to S3.
2. `coralnet-dataset` builds the training manifest, the resized images and the label names from that scrape.

The package READMEs are the reference for flags, environment variables, exclusion kinds and exit codes; this page gives the order and links to them.

- [Before you start](#before-you-start)
- [Recreate the data from scratch](#recreate-the-data-from-scratch)
- [Update the data](#update-the-data)
- [Use the new build in training](#use-the-new-build-in-training)
- [When something fails](#when-something-fails)
- [Known gaps](#known-gaps)

## Before you start

- **AWS credentials.** Both packages use the default boto3 credential chain, for example `AWS_PROFILE`.
- **CoralNet login.** The scraper reads `CORALNET_USERNAME` and `CORALNET_PASSWORD` when both are set. Otherwise it prompts on the terminal.
- **Environment.** Copy the CoralNet variables from `.env.example` into `.env`. Nothing loads `.env` for you, so export it in each shell before you run a command:

  ```bash
  set -a; source .env; set +a
  ```

  See the scraper's [environment variables](../packages/coralnet-scraper/README.md#environment-variables) and the build's [environment variables](../packages/coralnet-dataset/README.md#environment-variables).
- **Bucket.** Set `MERMAID_CORALNET_BUCKET` to the `source_bucket` of `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`). See [Use a build in training](../packages/coralnet-dataset/README.md#use-a-build-in-training).

Run every command from the repository root.

## Recreate the data from scratch

### Scrape CoralNet

1. Run the scraper:

   ```bash
   uv run --package coralnet-scraper coralnet-scrape
   ```

2. Read the failure summary at the end of the log. It lists the failed sources by error kind.
3. Check the exit code. 0 means every source downloaded or was skipped. 1 means a login failure, a failed source, or an uncaught error.

See [Run](../packages/coralnet-scraper/README.md#run) for the flags, the failure kinds and the exit codes. If a source failed, go to [When something fails](#when-something-fails).

### Build the dataset

Pick a version tag for the run, for example `20261118_coralnet`. Then run all four steps in one invocation:

```bash
uv run --package coralnet-dataset coralnet-dataset all --version-tag <run> --upload-to-s3
```

To run the steps one at a time, pass the same `--version-tag` to each. Each invocation computes its own default tag, so without it the outputs land in different run directories. The step order and the flags of each step are in [Run](../packages/coralnet-dataset/README.md#run).

Use an unused tag; see [Run](../packages/coralnet-dataset/README.md#run).

### Check the build

1. Read the summary that each step logs: the excluded sources from `build-annotations`, the count per `header_status` from `build-images`, and the resized, corrupted and failed counts from `resize`.
2. Check the exit code. Exit code 1 from `all` means at least one step reported failures. Every step still runs and writes its outputs.
3. Confirm that `etl-outputs/coralnet/<run>/` in the bucket holds four files:
   - `coralnet_annotations_<run>.parquet`
   - `coralnet_id2name_<run>.json`
   - `coralnet_images_<run>.parquet`
   - `coralnet_training_manifest_<run>.parquet`

The resized images are not in that directory. `resize` writes them under `<resized prefix>/resized/`, which every run shares. See [Outputs](../packages/coralnet-dataset/README.md#outputs) and [Check a build for completeness](../packages/coralnet-dataset/README.md#check-a-build-for-completeness).

## Update the data

### Add new sources

Re-run the scraper without flags. It skips each source that already has `annotations.csv` on S3, so it downloads new sources and any source that failed before its annotations upload.

### Pick up new annotations or images in existing sources

The scraper skips existing sources, so it does not see changes to them. To pick them up, run it with `--force`:

```bash
uv run --package coralnet-scraper coralnet-scrape --force
```

`--force` reprocesses every source, not only the changed ones; see [Resume and `--force`](../packages/coralnet-scraper/README.md#resume-and---force) for what it exports again and what it keeps.

### Rebuild under a new version tag

After any scrape, build again with a version tag that you have not used before:

```bash
uv run --package coralnet-dataset coralnet-dataset all --version-tag <new-run> --upload-to-s3
```

`build-annotations` and `build-images` read every source and every annotated image again. `resize` resizes only the images that have no resized object on S3.

Use an unused tag; see [Run](../packages/coralnet-dataset/README.md#run).

## Use the new build in training

### Select the build

Set `MERMAID_CORALNET_MANIFEST_VERSION=<run>` or `MERMAID_CORALNET_MANIFEST_PATH=<key>` in the environment of the training run. See [Use a build in training](../packages/coralnet-dataset/README.md#use-a-build-in-training).

### Make the build the default

With neither variable set, `CoralNetDataset` reads the build named by `_DEFAULT_RUN` in `mermaidseg/datasets/coralnet/coralnet_dataset.py`. To make the new build the default, set `_DEFAULT_RUN` to its tag.

A new default build changes model outputs, so item 7 of the working agreement in [AGENTS.md](../AGENTS.md#2-working-agreement) applies. Run the evaluation, and put the metrics and the baseline they compare against in the PR.

### Add new sources to the splits

Each CoralNet split has a `whitelist_sources` list, and `CoralNetDataset` keeps only the sources on that list. A new source is not in any split until you add its id to a list:

1. Add the id to a split's `whitelist_sources` in `configs/data_config.yaml`.
2. Regenerate `configs/data_config_trainval.yaml`, which is derived from it:

   ```bash
   uv run python scripts/build_trainval_data_config.py
   ```

### Map new labels

A label name in the build's `coralnet_id2name_<run>.json` trains as background when `configs/coralnet_to_mermaid_mapping_temporary.json` has no key for it, or its value is `null` (`mermaidseg/dataset_reconciliation/label_mapping.py`, `mermaidseg/dataset_reconciliation/registry.py`). Names are matched lowercased. Before training, give each new label a MERMAID benthic-attribute name.

### Duplicates in new sources

`configs/coralnet/image_blacklist.parquet` and `configs/coralnet/annotation_overrides.parquet` cover only the sources they were built from. Duplicate images in new sources are not removed. See the build's [Known gaps](../packages/coralnet-dataset/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).

### Warm the offline cache again

An offline run (`MERMAIDSEG_S3_OFFLINE=1`) fails on any cache miss. The new manifest, its id2name JSON and new images are not in a cache that you filled before. Fill it again with the build selected, using the same data config as the training run:

```bash
MERMAID_CORALNET_MANIFEST_VERSION=<run> uv run python scripts/prefetch_cache.py --cache-dir /path/to/cache
```

`--data-config` defaults to `configs/data_config.yaml`. Run the script with `--help` for the other options.

## When something fails

| Failure | What to do |
|---|---|
| Scraper exit 1: login failure | Check `CORALNET_USERNAME` and `CORALNET_PASSWORD`, then run again. |
| Scraper exit 1: a source failed with `metadata_download_failed`, `labelset_failed`, `image_list_failed`, `images_missing_url` or `images_upload_failed` | Unless it also failed with `annotations_failed`, the source has `annotations.csv` on S3, so a plain re-run skips it. Run the scraper with `--force`. |
| Scraper exit 1: a source failed with any other kind, for example `annotations_failed` or an exception class name | If the source has no `annotations.csv` on S3, run the scraper again without flags. If it has one, run with `--force`. |
| `build-annotations` excluded a source with `annotations_missing` | Run the scraper again without flags, then rebuild. |
| `build-annotations` excluded a source with `image_list_missing` | Run the scraper with `--force`, then rebuild. |
| `build-annotations` excluded a source with `labelset_missing` | Check the source on CoralNet. If it has no labels, the scraper writes no `labelset.csv` and the exclusion is expected. If it has labels, the labelset step failed (`labelset_failed` in the scraper summary): delete the source's `annotations.csv` or run the scraper with `--force`, then rebuild. |
| `build-annotations` excluded a source with `csv_read_failed` | Rebuild first. If it repeats, run the scraper with `--force`, then rebuild. |
| `build-annotations` excluded a source with `columns_missing` or `annotations_empty` | The CSV on S3 is bad, or, for `annotations_empty`, the source may have no annotations. Run the scraper with `--force` to export it again, then rebuild. |
| `build-annotations` excluded a source with `image_list_incomplete` or `label_unnamed` | The source CSVs do not agree with each other. Run the scraper with `--force`, then rebuild. |
| `build-images` reports `not_found` | The image object is not on S3. Run the scraper with `--force` to upload it, then rebuild. |
| `build-images` reports `s3_error` or `worker_exception` | Rebuild. |
| `build-images` reports `corrupt` | The object on S3 is bad. The scraper does not replace an existing image object, so `--force` does not repair it. The manifest drops the image. |
| `resize` reports corrupted or failed images | Run the build again. `resize` retries every image that has no resized object. The manifest drops an image that needs a resize and has none. |

`--force` is all-or-nothing: it reprocesses every source, not only the failed one. After a scrape, always rebuild under a new version tag.

A partial `annotations.csv` from an interrupted paginated export is not detectable. Neither the scraper nor the build reports it, so the build includes that source with fewer images than CoralNet has. See [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).

## Known gaps

See the build's [Known gaps](../packages/coralnet-dataset/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).
