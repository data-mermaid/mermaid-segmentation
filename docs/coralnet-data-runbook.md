# Recreate and update the CoralNet data

This runbook takes you from an empty bucket, or an older scrape, to a CoralNet build that training reads. It has two stages:

1. `coralnet-scraper` downloads the public CoralNet sources to S3.
2. `coralnet-dataset` builds the training manifest, the resized images and the label names from that scrape.

The package READMEs are the reference for flags, environment variables, exclusion kinds and exit codes. This page links to them and does not repeat them.

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
- **Bucket.** Set `MERMAID_CORALNET_BUCKET` to the bucket that `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`) reads, its `source_bucket`. The manifest's `image_s3_key` and the id2name JSON are keys in that bucket. A build in any other bucket is not readable in training.

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

Use a tag that is not on S3 yet. An upload to an existing key stops the command with `RuntimeError`, with or without `--version-tag`. Only `--overwrite` lets the upload replace existing files.

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

Re-run the scraper without flags. It skips each source that already has `annotations.csv` on S3, so it downloads only the sources that are new on CoralNet.

### Pick up new annotations or images in existing sources

The scraper skips existing sources, so it does not see changes to them. To pick them up, run it with `--force`:

```bash
uv run --package coralnet-scraper coralnet-scrape --force
```

`--force` reprocesses every source, not only the changed ones. For each source:

- The scraper exports the annotations again and replaces `annotations.csv`.
- It uploads only the images that are not on S3. An image object that exists is not uploaded again.
- A source with more than 5000 images is exported in pages and merged into its existing `annotations.csv`. If the existing file already covers as many images as CoralNet reports, the scraper keeps it and does not export again. New annotations on existing images of such a source are not picked up.

See [Resume and `--force`](../packages/coralnet-scraper/README.md#resume-and---force).

### Rebuild under a new version tag

After any scrape, build again with a version tag that you have not used before:

```bash
uv run --package coralnet-dataset coralnet-dataset all --version-tag <new-run> --upload-to-s3
```

`build-annotations` and `build-images` read every source and every annotated image again. `resize` resizes only the images that have no resized object on S3.

Do not reuse an old tag. A reused tag stops with `RuntimeError` unless you pass `--overwrite`. With `--overwrite`, the upload replaces the files of that tag, and any training run that pins the old build then reads different data.

## Use the new build in training

### Select the build

Set one of these in the environment of the training run. The path wins when both are set:

- `MERMAID_CORALNET_MANIFEST_VERSION=<run>` reads `etl-outputs/coralnet/<run>/coralnet_training_manifest_<run>.parquet`.
- `MERMAID_CORALNET_MANIFEST_PATH=<key>` reads that key.

`CoralNetDataset` reads the label names from `coralnet_id2name_<run>.json` in the same directory as the manifest, where `<run>` is the directory name. A custom path must keep that layout. See [Use a build in training](../packages/coralnet-dataset/README.md#use-a-build-in-training).

### Make the build the default

With neither variable set, `CoralNetDataset` reads the build named by `_DEFAULT_RUN` in `mermaidseg/datasets/coralnet/coralnet_dataset.py`. To make the new build the default, set `_DEFAULT_RUN` to its tag.

A new default build changes model outputs, so item 7 of the working agreement in [AGENTS.md](../AGENTS.md#2-working-agreement) applies. Run the evaluation, and put the metrics and the baseline they compare against in the PR.

### Add new sources to the splits

Each CoralNet split in `configs/data_config.yaml` and `configs/data_config_trainval.yaml` has a `whitelist_sources` list. `CoralNetDataset` keeps only the sources on that list. A new source is not in any split until you add its id to a list.

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
| Scraper exit 1: a source failed with `metadata_download_failed`, `labelset_failed`, `image_list_failed`, `images_missing_url` or `images_upload_failed` | The source has `annotations.csv` on S3, so a plain re-run skips it. Run the scraper with `--force`. |
| Scraper exit 1: a source failed with any other kind, for example `annotations_failed` or an exception class name | If the source has no `annotations.csv` on S3, run the scraper again without flags. If it has one, run with `--force`. |
| `build-annotations` excluded a source with `annotations_missing`, `image_list_missing` or `labelset_missing` | Run the scraper with `--force` to write the missing file, then rebuild. |
| `build-annotations` excluded a source with `csv_read_failed`, `columns_missing` or `annotations_empty` | The CSV on S3 is bad. Run the scraper with `--force` to export it again, then rebuild. |
| `build-annotations` excluded a source with `image_list_incomplete` or `label_unnamed` | The source CSVs do not agree with each other. Run the scraper with `--force`, then rebuild. |
| `build-images` reports `not_found` | The image object is not on S3. Run the scraper with `--force` to upload it, then rebuild. |
| `build-images` reports `s3_error` or `worker_exception` | Rebuild. |
| `build-images` reports `corrupt` | The object on S3 is bad. The scraper does not replace an existing image object, so `--force` does not repair it. The manifest drops the image. |
| `resize` reports corrupted or failed images | Run the build again. `resize` retries every image that has no resized object. The manifest drops an image that needs a resize and has none. |

`--force` is all-or-nothing: it reprocesses every source, not only the failed one. After a scrape, always rebuild under a new version tag.

A partial `annotations.csv` from an interrupted paginated export is not detectable. Neither the scraper nor the build reports it, so the build includes that source with fewer images than CoralNet has. See [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).

## Known gaps

See the build's [Known gaps](../packages/coralnet-dataset/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).
