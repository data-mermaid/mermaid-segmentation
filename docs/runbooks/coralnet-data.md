# Recreate and update the CoralNet data

This runbook takes you from an empty bucket, or an older scrape, to a CoralNet build that training reads. It has two stages:

1. `coralnet-scraper` downloads the public CoralNet sources to S3.
2. `coralnet-dataset` builds the training manifest, the resized images and the label names from that scrape.

The package READMEs are the reference for flags, environment variables, exclusion kinds and exit codes; this page gives the order and links to them.

- [Before you start](#before-you-start)
- [Recreate the data from scratch](#recreate-the-data-from-scratch)
- [Update the data](#update-the-data)
- [Adopt an existing bucket](#adopt-an-existing-bucket)
- [Tune the request rate](#tune-the-request-rate)
- [Use the new build in training](#use-the-new-build-in-training)
- [When something fails](#when-something-fails)
- [Known gaps](#known-gaps)

## Before you start

- **AWS credentials.** Both packages use the default boto3 credential chain, for example `AWS_PROFILE`.
- **CoralNet login.** The scraper reads `CORALNET_USERNAME` and `CORALNET_PASSWORD` when both are set. Otherwise it prompts on the terminal. Only the scraper run logs in; `verify-existing` and `invalidate` read no credentials.
- **Environment.** Copy the CoralNet variables from `.env.example` into `.env`. Nothing loads `.env` for you, so export it in each shell before you run a command:

  ```bash
  set -a; source .env; set +a
  ```

  See the scraper's [environment variables](../../packages/coralnet-scraper/README.md#environment-variables) and the build's [environment variables](../../packages/coralnet-dataset/README.md#environment-variables).
- **Bucket.** Set `MERMAID_CORALNET_BUCKET` to the `source_bucket` of `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`). See [Use a build in training](../../packages/coralnet-dataset/README.md#use-a-build-in-training).

- **Request rate.** The scraper sends requests without a rate limit unless you set one. To choose a limit, see [Tune the request rate](#tune-the-request-rate).

- **Earlier scraper version.** If the bucket was filled by an earlier scraper version, its source folders have no `_complete.json`. Do [Adopt an existing bucket](#adopt-an-existing-bucket) first. A plain run would scrape every source again.

Run every command from the repository root.

## Recreate the data from scratch

### Scrape CoralNet

1. Run the scraper:

   ```bash
   uv run --package coralnet-scraper coralnet-scrape
   ```

2. Read the failure summary at the end of the log. It lists the failed sources by error kind.
3. Check the exit code. 0 means every source downloaded or was skipped. 1 means a login failure, a failed source, or an uncaught error.

See [Run](../../packages/coralnet-scraper/README.md#run) for the flags, the failure kinds and the exit codes. If a source failed, go to [When something fails](#when-something-fails).

### Build the dataset

Pick a version tag for the run, for example `20261118_coralnet`. Then run all four steps in one invocation:

```bash
uv run --package coralnet-dataset coralnet-dataset all --version-tag <run> --upload-to-s3
```

To run the steps one at a time, pass the same `--version-tag` to each. Each invocation computes its own default tag, so without it the outputs land in different run directories. The step order and the flags of each step are in [Run](../../packages/coralnet-dataset/README.md#run).

Use an unused tag; see [Run](../../packages/coralnet-dataset/README.md#run).

### Check the build

1. Read the summary that each step logs: the excluded sources from `build-annotations` (it excludes a source with no `_complete.json` as `not_complete`), the count per `header_status` from `build-images`, and the resized, oversized, corrupted and failed counts from `resize`.
2. Check the exit code. Exit code 1 from `all` means at least one step reported failures. Every step still runs and writes its outputs.
3. Confirm that `etl-outputs/coralnet/<run>/` in the bucket holds four files:
   - `coralnet_annotations_<run>.parquet`
   - `coralnet_id2name_<run>.json`
   - `coralnet_images_<run>.parquet`
   - `coralnet_training_manifest_<run>.parquet`

The resized images are not in that directory. `resize` writes them under `<resized prefix>/resized/`, which every run shares. See [Outputs](../../packages/coralnet-dataset/README.md#outputs) and [Check a build for completeness](../../packages/coralnet-dataset/README.md#check-a-build-for-completeness).

## Update the data

### Add new sources

Run the scraper without flags. It skips each source that has a `_complete.json` marker on S3, so it downloads new sources and finishes any source that is incomplete.

### Pick up new annotations or images in existing sources

The scraper skips complete sources, so it does not see changes to them. To pick up the changes:

1. Delete the markers of the sources whose live image count differs from the count in their marker:

   ```bash
   uv run --package coralnet-scraper coralnet-scrape invalidate --changed
   ```

2. Run the scraper without flags. It redoes the invalidated sources. Image objects already on S3 are skipped.

`--changed` compares image counts only, so it does not see an annotation change that leaves the count the same. To redo specific sources, use `invalidate --source-ids <id>,<id>,...`. To redo every source, use `invalidate --all`, which forces a full re-export of every source. See [Resume](../../packages/coralnet-scraper/README.md#resume).

### Rebuild under a new version tag

After any scrape, build again with a version tag that you have not used before:

```bash
uv run --package coralnet-dataset coralnet-dataset all --version-tag <new-run> --upload-to-s3
```

`build-annotations` and `build-images` read every source and every annotated image again. `resize` resizes only the images that have no resized object on S3.

Use an unused tag; see [Run](../../packages/coralnet-dataset/README.md#run).

## Adopt an existing bucket

Use this once for a bucket that an earlier version of the scraper filled, so it has sources without `_complete.json`. The earlier version inferred "done" from `annotations.csv`, so its sources can be incomplete.

1. Tune the request rate; see [Tune the request rate](#tune-the-request-rate).
2. Verify the sources that have no marker and mark those that pass:

   ```bash
   uv run --package coralnet-scraper coralnet-scrape verify-existing
   ```

   This sends one overview request per source and reads every source's CSVs from S3.
3. Run the scraper without flags. It downloads every source that has no marker again. Image objects already on S3 are skipped.

   > **Warning:** `--prune` deletes cannot be undone unless the bucket has S3 versioning enabled. Before step 4, read the `orphan_image_objects:<n>` count in each source's marker.
4. For each source that has an `orphan_image_objects` warning in its marker, delete the image objects that are not in its image list. Step 2 already marked these sources, and the scraper skips a marked source, so remove the markers first:

   1. Remove the marker of each source:

      ```bash
      uv run --package coralnet-scraper coralnet-scrape invalidate --source-ids <id>,<id>,...
      ```

   2. Run the scraper with `--prune` on the same sources:

      ```bash
      uv run --package coralnet-scraper coralnet-scrape --prune --source-ids <id>,<id>,...
      ```

   `--prune` deletes only for a source that verifies in that run.
5. Rebuild the dataset under a new version tag:

   ```bash
   uv run --package coralnet-dataset coralnet-dataset all --version-tag <new-run> --upload-to-s3
   ```

## Tune the request rate

Find the highest request rate at which CoralNet answers without 429s, retries or transport failures. The scraper has no rate limit by default.

1. Pick about 10 public sources of mid size (a few hundred to a few thousand images each) as a test set. Run the scraper on them into a scratch prefix with the default settings (no rate limit, default workers):

   ```bash
   MERMAID_CORALNET_PREFIX=scratch/rate-tuning uv run --package coralnet-scraper coralnet-scrape --source-ids <id>,<id>,...
   ```

   The run includes image traffic because the scratch prefix is empty. Before each further iteration, delete the scratch prefix from the bucket: a source with a marker is skipped, so deleting the prefix makes each run download the same sources again.
2. Read the [run report](../../packages/coralnet-scraper/README.md#run-report) at `s3://<bucket>/scratch/rate-tuning/_runs/<UTC time>-<short id>.json`. Note the `429` and 5xx counts, the `timeout` and `connection` counts under `transport.retries`, the failed sources, and `transport.mean_rate_per_s`.
3. If there is any 429, if retried requests exceed 1% of `transport.requests`, or if any source failed for a transport reason, set `--requests-per-second` to half the observed `transport.mean_rate_per_s`, which counts retries too. If timeouts dominate, also halve `--scraper-workers`. Repeat from step 1.
4. Stop at the first setting with no 429, retries under 1% of requests and no transport failures. Record it in the table below, and set the values in `.env`: `MERMAID_CORALNET_REQUESTS_PER_SECOND`, `MERMAID_CORALNET_SCRAPER_WORKERS` and, if you changed it, `MERMAID_CORALNET_EXPORT_WORKERS`.
5. Run this procedure again if a later run reports 429s or the circuit breaker stops it.

Results:

| Date | Requests per second | Scraper workers | Export workers | Run report path | Outcome |
|---|---|---|---|---|---|
| | | | | | |

## Use the new build in training

### Select the build

Set `manifest_path` on each CoralNet split (`train`, `val` and `test`) in `configs/data_config.yaml`:

```yaml
manifest_path: etl-outputs/coralnet/<run>/coralnet_training_manifest_<run>.parquet
```

Then regenerate the derived configs, starting with `configs/data_config_trainval.yaml`. See [Regenerate the derived configs](derived-configs.md). For the key layout, see [Use a build in training](../../packages/coralnet-dataset/README.md#use-a-build-in-training).

A new build changes model outputs, so item 7 of the working agreement in [AGENTS.md](../../AGENTS.md#2-working-agreement) applies. Run the evaluation, and put the metrics and the baseline they compare against in the PR.

### Add new sources to the splits

Each CoralNet split has a `whitelist_sources` list, and `CoralNetDataset` keeps only the sources on that list. A new source is not in any split until you add its id to a list:

1. Add the id to a split's `whitelist_sources` in `configs/data_config.yaml`. The id must already be in the manifest chosen under [Select the build](#select-the-build). Otherwise `CoralNetDataset` raises a `ValueError` that names it.
2. Regenerate the derived configs, starting with `configs/data_config_trainval.yaml`. See [Regenerate the derived configs](derived-configs.md).

### Map new labels

A label name in the build's `coralnet_id2name_<run>.json` trains as background when `configs/coralnet_to_mermaid_mapping_temporary.json` has no key for it, or its value is `null` (`mermaidseg/dataset_reconciliation/label_mapping.py`, `mermaidseg/dataset_reconciliation/registry.py`). Names are matched lowercased. Before training, give each new label a MERMAID benthic-attribute name.

### Duplicates in new sources

`configs/coralnet/image_blacklist.parquet` and `configs/coralnet/annotation_overrides.parquet` cover only the sources they were built from. Duplicate images in new sources are not removed. See the build's [Known gaps](../../packages/coralnet-dataset/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).

### Warm the offline cache again

An offline run (`MERMAIDSEG_S3_OFFLINE=1`) fails on any cache miss. The new manifest, its id2name JSON and new images are not in a cache that you filled before. Fill it again, using the same data config as the training run. The data config selects the build:

```bash
uv run python scripts/prefetch_cache.py --cache-dir /path/to/cache
```

`--data-config` defaults to `configs/data_config.yaml`. For the Slurm command, with `HF_HOME` and the final runs' data config, see [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).

## When something fails

| Failure | What to do |
|---|---|
| Scraper exit 1: login failure | Check `CORALNET_USERNAME` and `CORALNET_PASSWORD`, then run again. |
| Scraper exit 1: a source failed with `cannot access source <id>: …` | The source is missing or private on CoralNet. Running again does not help. The failure does not stop the run. |
| Scraper stopped with `Run stopped early: circuit_open` (exit 1) | CoralNet is likely down or rejecting requests. Wait, then check the transport tallies in the [run report](../../packages/coralnet-scraper/README.md#run-report). Run the same command again. If the tallies show 429 retries, lower `--requests-per-second`. |
| Scraper exit 1: a source failed with any other kind, except `verification_failed` | The source has no marker, so run the scraper again without flags. To retry only that source, add `--source-ids <id>`. |
| Scraper exit 1: a source failed with `verification_failed` | The log lists the [verification codes](../../packages/coralnet-scraper/README.md#verification). The source has no marker and its `_partial/` is discarded. Run the scraper again without flags. If the same codes repeat, check the source on CoralNet. |
| Scraper exit 1: a source failed with `overview_total_missing` | The source's overview page showed no "Total images" count. Check the source page on CoralNet, then run again. The source is retried on each run until it verifies. |
| Scraper stopped because CoralNet listed fewer sources than the previous run (exit 1, no run report) | Check that CoralNet lists the public sources. If the drop is real, run the scraper with `--allow-source-drop`. |
| Scraper run was killed or the machine stopped | Run the same command again. Finished stages and pages are kept in `_partial/`. |
| `build-annotations` excluded a source with `annotations_missing` | Run the scraper again without flags, then rebuild. If the source has a marker, run `invalidate --source-ids <id>` first. |
| `build-annotations` excluded a source with `image_list_missing` | Run `invalidate --source-ids <id>`, run the scraper without flags, then rebuild. |
| `build-annotations` excluded a source with `labelset_missing` | Check the source on CoralNet. If it has no labels, the scraper writes no `labelset.csv` and the exclusion is expected. If it has labels, the labelset step failed (`labelset_failed` in the scraper summary): run `invalidate --source-ids <id>`, run the scraper without flags, then rebuild. |
| `build-annotations` excluded a source with `csv_read_failed` | Rebuild first. If it repeats, run `invalidate --source-ids <id>`, run the scraper without flags, then rebuild. |
| `build-annotations` excluded a source with `columns_missing` or `annotations_empty` | The CSV on S3 is bad, or, for `annotations_empty`, the source may have no annotations. Run `invalidate --source-ids <id>`, run the scraper without flags to export it again, then rebuild. |
| `build-annotations` excluded a source with `image_list_incomplete` or `label_unnamed` | The source CSVs do not agree with each other. Run `invalidate --source-ids <id>`, run the scraper without flags, then rebuild. |
| `build-images` reports `not_found` | The image object is not on S3. Run `invalidate --source-ids <id>` for its source and run the scraper without flags to upload it, then rebuild. |
| `build-images` reports `s3_error` or `worker_exception` | Rebuild. |
| `build-images` reports `corrupt` | The object on S3 is bad. Delete the corrupt image objects, run `coralnet-scrape invalidate --source-ids <id>`, then run `coralnet-scrape`. `invalidate` deletes the marker and `_partial/`, so the scraper runs the walk and the export again. Only the deleted images download again, because the image stage skips non-empty objects. The manifest drops the image until then. |
| `resize` reports corrupted or failed images | Run the build again. `resize` retries every image that has no resized object, so a rerun helps only if the cause was transient. The manifest drops an image that needs a resize and has none. |
| `resize` reports oversized images | The original exceeds twice Pillow's `MAX_IMAGE_PIXELS`, the decompression-bomb error threshold. A rerun skips the image again and `resize` exits 1 again. The manifest drops the image. |

After a scrape, always rebuild under a new version tag.

## Known gaps

See the build's [Known gaps](../../packages/coralnet-dataset/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).
