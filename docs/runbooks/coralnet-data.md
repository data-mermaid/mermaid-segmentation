# Recreate and update the CoralNet data

This runbook takes you from an empty bucket, or an older scrape, to published CoralNet dataset versions that training pins. It has three stages:

1. `coralnet-scraper` downloads the public CoralNet sources to S3.
2. `segdata build coralnet` builds the annotations table, the images table and the label names from that scrape.
3. `segdata build coralnet raw`, `segdata taxonomy` and `segdata curate coralnet` stage the `coralnet-raw` dataset version, a `mermaid-benthic` taxonomy version and the `coralnet` dataset version. `segdata promote` publishes each staged version (with `--taxonomy` for the taxonomy).

The package READMEs are the reference for flags, environment variables, exclusion kinds and exit codes; this page gives the order and links to them.

- [Before you start](#before-you-start)
- [Recreate the data from scratch](#recreate-the-data-from-scratch)
- [Update the data](#update-the-data)
- [Adopt an existing bucket](#adopt-an-existing-bucket)
- [Tune the request rate](#tune-the-request-rate)
- [Train on a published version](#train-on-a-published-version)
- [Publish CoralNet dataset versions](#publish-coralnet-dataset-versions)
- [When something fails](#when-something-fails)
- [Known gaps](#known-gaps)

## Before you start

- **AWS credentials.** Both packages use the default boto3 credential chain, for example `AWS_PROFILE`.
- **CoralNet login.** The scraper reads `CORALNET_USERNAME` and `CORALNET_PASSWORD` when both are set. Otherwise it prompts on the terminal. Only the scraper run logs in; `verify-existing` and `invalidate` read no credentials.
- **Environment.** Copy the CoralNet variables from `.env.example` into `.env`. Nothing loads `.env` for you, so export it in each shell before you run a command:

  ```bash
  set -a; source .env; set +a
  ```

  See the scraper's [environment variables](../../packages/coralnet-scraper/README.md#environment-variables) and the build's [environment variables](../../packages/segdata-etl/README.md#environment-variables).
- **Bucket.** Set `MERMAID_CORALNET_BUCKET` to the bucket that holds the scrape.
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

Pick a version tag for the run, for example `20261118_coralnet`. Then run both stages in one invocation:

```bash
uv run --package segdata-etl segdata build coralnet all --version-tag <run> --upload-to-s3
```

To run the stages one at a time, pass the same `--version-tag` to each. Each invocation computes its own default tag, so without it the outputs land in different run directories. The step order and the flags of each step are in [Run](../../packages/segdata-etl/README.md#run).

Use an unused tag; see [Run](../../packages/segdata-etl/README.md#run).

### Check the build

1. Read the summary that each stage logs: the excluded sources from `build-annotations` (it excludes a source with no `_complete.json` as `not_complete`), and the count per `header_status` from `build-images`.
2. Check the exit code. Exit code 1 from `all` means at least one stage reported failures. Every stage still runs and writes its outputs.
3. Confirm that `etl-outputs/coralnet/<run>/` in the bucket holds three files:
   - `coralnet_annotations_<run>.parquet`
   - `coralnet_id2name_<run>.json`
   - `coralnet_images_<run>.parquet`

See [Outputs](../../packages/segdata-etl/README.md#outputs) and [Check a build for completeness](../../packages/segdata-etl/README.md#check-a-build-for-completeness).

## Update the data

### Add new sources

Run the scraper without flags. It skips each source that has a `_complete.json` marker on S3, so it downloads new sources and finishes any source that is incomplete.

### Pick up new annotations or images in existing sources

The scraper skips complete sources, so it does not see changes to them. To pick up the changes:

1. List the sources whose live image count differs from the count in their marker. This deletes nothing:

   ```bash
   uv run --package coralnet-scraper coralnet-scrape invalidate --changed --dry-run
   ```

2. Check the sources in the `Would invalidate <n> sources` line, then delete their markers:

   ```bash
   uv run --package coralnet-scraper coralnet-scrape invalidate --changed
   ```

3. Run the scraper without flags. It redoes the invalidated sources. Image objects already on S3 are skipped.

`--changed` compares image counts only, so it does not see an annotation change that leaves the count the same. To redo specific sources, use `invalidate --source-ids <id>,<id>,...`. To redo every source, use `invalidate --all --yes`, which forces a full re-export of every source. Run each with `--dry-run` first. See [Resume](../../packages/coralnet-scraper/README.md#resume).

### Rebuild under a new version tag

After any scrape, build again with a version tag that you have not used before:

```bash
uv run --package segdata-etl segdata build coralnet all --version-tag <new-run> --upload-to-s3
```

`build-annotations` and `build-images` read every source and every annotated image again. Then [publish new dataset versions](#publish-coralnet-dataset-versions) from the new run.

Use an unused tag; see [Run](../../packages/segdata-etl/README.md#run).

## Adopt an existing bucket

Use this once for a bucket that an earlier version of the scraper filled, so it has sources without `_complete.json`. The earlier version inferred "done" from `annotations.csv`, so its sources can be incomplete.

1. Tune the request rate; see [Tune the request rate](#tune-the-request-rate).
2. Verify the sources that have no marker and mark those that pass:

   ```bash
   uv run --package coralnet-scraper coralnet-scrape verify-existing
   ```

   This sends one overview request per source and reads every source's CSVs from S3. A source with zero Confirmed images or zero images is logged as skipped, gets no marker, and does not fail the command. Exit code 1 means at least one source failed verification.
3. Run the scraper without flags. It downloads every source that has no marker again. Image objects already on S3 are skipped.

   > **Warning:** `--prune` deletes cannot be undone unless the bucket has S3 versioning enabled. Before step 4, read the `orphan_image_objects:<n>` count in each source's marker.
4. For each source that has an `orphan_image_objects` warning in its marker, delete the image objects that are not in its image list. Step 2 already marked these sources, and the scraper skips a marked source, so remove the markers first:

   1. Remove the marker of each source:

      ```bash
      uv run --package coralnet-scraper coralnet-scrape invalidate --source-ids <id>,<id>,...
      ```

   2. Run the scraper with `--prune --dry-run` on the same sources. It logs each orphan key as `would prune orphan image object <key>` and deletes nothing. A source that verifies is left without a marker:

      ```bash
      uv run --package coralnet-scraper coralnet-scrape --prune --dry-run --source-ids <id>,<id>,...
      ```

   3. Check the logged keys, then run the scraper with `--prune` on the same sources:

      ```bash
      uv run --package coralnet-scraper coralnet-scrape --prune --source-ids <id>,<id>,...
      ```

   `--prune` deletes only for a source that verifies in that run.
5. Rebuild the dataset under a new version tag:

   ```bash
   uv run --package segdata-etl segdata build coralnet all --version-tag <new-run> --upload-to-s3
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

## Train on a published version

Training pins a published `coralnet` version in `configs/data_config.yaml`. The sync downloads the version and resizes its images to 2048 px at sync time. See [Get the training datasets](training-datasets.md).

### Pin the version

1. Publish the version. See [Publish CoralNet dataset versions](#publish-coralnet-dataset-versions).
2. Set `version` and `manifest_sha256` of `datasets.coralnet` in `configs/data_config.yaml` to the pin that `segdata curate coralnet` printed. The data config selects the groups of each split from `configs/splits/coralnet.csv`: `train: {use: [train, val]}` trains on the train and val groups, and `test: {use: [test]}` selects the test groups.
3. Regenerate the derived configs. See [Regenerate the derived configs](derived-configs.md).
4. Sync the cache. See [Get the data](training-datasets.md#get-the-data).

A new version changes model outputs, so item 7 of the working agreement in [AGENTS.md](../../AGENTS.md#2-working-agreement) applies. Run the evaluation, and put the metrics and the baseline they compare against in the PR.

### Add new sources to the splits

The split file `configs/splits/coralnet.csv` assigns each CoralNet group (`s<source_id>`) to `train`, `val`, `test` or `excluded`. A sync fails on any group of the pinned version that the file does not list. A group that is `excluded` stays in the version and is not used by a `use:` selection. To add sources:

1. Publish a `coralnet` version that holds them, and pin it.
2. Append rows to `configs/splits/coralnet.csv` by hand. This keeps existing train, val and test membership. Each row has `group` `s<id>`, the `source_id`, its `split`, and a `split_group` that is either a new id not used elsewhere or blank.
3. Run `segdata split leakage` on the file against the new version. The full command is in [CoralNet split](../../packages/segdata-etl/README.md#coralnet-split).
4. Regenerate the derived configs. See [Regenerate the derived configs](derived-configs.md).

`segdata split build` assigns every source again, so existing train, val and test membership changes, the test set included. Use it only to re-split on purpose. That changes model inputs, so it needs review. If you rebuild, copy `<out-dir>/coralnet.csv` and its meta over the committed files.

### Map new labels

The labels of a `coralnet` version are in its `labels.parquet`. `segdata curate coralnet` fills them from the MERMAID LabelMapping API (`/v1/classification/labelmappings/?provider=CoralNet`) at stage time. Rows are keyed by CoralNet label id, and the build's `coralnet_id2name_<run>.json` names each id, so a label renamed in CoralNet still maps. A label that has no API row, or a row with no target, has null MERMAID columns in `labels.parquet`, and training ignores their points (label 0 is the ignore index). Add the missing rows in MERMAID, and fill in the target of rows that have none, before you stage the version. There is no mapping file in this repository to edit.

The version's manifest records the mapping under `label_mapping`: its sha256, fetch time, label count and mapped count. A new version can map differently from an older one, because the API answer changes. Compare the class set of the new run with the baseline run before you compare metrics.

### Duplicates in new sources

`image_blacklist.parquet` and `annotation_overrides.parquet` in `packages/segdata-etl/src/segdata_etl/datasets/coralnet/curate/data/` cover only the sources they were built from. Duplicate images in new sources are not removed. See the build's [Known gaps](../../packages/segdata-etl/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).

## Publish CoralNet dataset versions

This procedure stages a CoralNet build locally, then publishes it as dataset versions in `s3://mermaid-ic-datasets/`: `coralnet-raw`, then a curated `coralnet` version whose labels the MERMAID API maps. The flags, the tables and the resume rules are in [CoralNet dataset versions](../../packages/segdata-etl/README.md#coralnet-dataset-versions).

### Preconditions

- A build run in `etl-outputs/coralnet/<run>/`. See [Build the dataset](#build-the-dataset). `raw` stages `20260623_nogit` unless you pass `--build-run`.
- A git checkout of this repository. Commit and push before you stage: every stage and publish command refuses a tree with uncommitted changes to tracked files, and a HEAD that no remote branch contains. `raw` checks this before it probes any image.
- The environment from [Before you start](#before-you-start). `raw` requires `MERMAID_CORALNET_BUCKET` (it raises `ValueError` when unset) and reads `MERMAID_CORALNET_PREFIX`.
- AWS credentials that read the scrape bucket. `promote` also needs credentials that write `mermaid-ic-datasets`.
- Network access to `api.datamermaid.org`, for the taxonomy and for `curate`, which fetches the CoralNet label mapping.
- A local stage directory, `<stage>` below. Every `build`, `taxonomy` and `curate` command writes there and nothing to S3. Use one stage directory for the whole procedure.

Each stage command prints `version:` and `manifest_sha256:`. Record both: they are the pin that the next step takes. `promote` publishes the version you staged, under the same name, so the pin of a staged version is the pin of the published one.

Publishing is irreversible: the bucket denies overwrite and delete. Get approval before each `promote`.

### Stage the raw version

Run this step on an EC2 host in `us-east-1`, not a laptop. A full corpus reads far more than a laptop link carries. Run it from the repository root of a checkout on that host:

```bash
uv run --package segdata-etl segdata build coralnet raw --stage-dir <stage> --build-run <run>
```

The command HEADs every kept original. It also reads in full each original that is multipart or has no CRC checksum, which is most scraper uploads, to compute the values the copy carries. It writes the tables and `copies.jsonl` under `<stage>`. No image is copied and nothing is written to S3. `--workers N` sets the concurrency. To reuse the probe results of an earlier `coralnet-raw` version, stage that version under the same `<stage>` and add `--parent-version <v>` and `--parent-manifest-sha256 <sha>`. The command reads the parent from `<stage>`, not from S3. With a parent, an image whose scrape original is gone (absent from the scrape listing, or `header_status` `not_found`) keeps the parent's row and gets no copy when the parent holds its `image_id`; `stats.dropped["carried_from_parent"]` counts these images. This lets you delete the scrape originals of already-published images without breaking a later raw build. A carried image that disagrees with the parent's size stops the build with a `ValueError`; see [When a publish fails](#when-a-publish-fails).

To delete scrape originals safely, go in this order:

1. Publish the parent version.
2. Stage the new raw version with the parent pin.
3. Check `carried_from_parent` and `source_missing` in the staged `manifest.json`, under `stats.dropped`.
4. Promote it.
5. Delete originals, only those whose `image_id` the published parent holds.

A raw version staged before the deletion still plans copies for the images it reuses from the parent. Promote it before you delete, or stage it again after.

The probe results are checkpointed under `<stage>/checkpoints/`. If the run stops, run the same command again with the same `--stage-dir`: it probes only the images the checkpoint lacks. See [Resume a raw stage](../../packages/segdata-etl/README.md#resume-a-raw-stage).

### Publish the raw version

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset coralnet-raw --version <raw>
```

`promote` copies the originals into `s3://mermaid-ic-datasets/images/coralnet/`, then writes the tables and `manifest.json`. See [Promote a version](../../packages/segdata-etl/README.md#promote-a-version).

### Publish the taxonomy

`curate` checks each mapped benthic attribute id and growth form id against a `mermaid-benthic` taxonomy version on S3, so publish one before you stage the curated version. A taxonomy that lacks an id the API returns stops `curate`: stage and publish a newer one. Stage it, and record the pin it prints:

```bash
uv run --package segdata-etl segdata taxonomy --stage-dir <stage>
```

Then publish it under the same name:

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --taxonomy mermaid-benthic --version <tax>
```

`curate` refuses a taxonomy pin that is only staged. See [Taxonomy snapshot](../../packages/segdata-etl/README.md#taxonomy-snapshot).

### Stage and publish the curated version

Stage the curated version against the published raw pin and the published taxonomy pin:

```bash
uv run --package segdata-etl segdata curate coralnet \
  --stage-dir <stage> \
  --raw-version <raw> --raw-manifest-sha256 <sha> \
  --taxonomy-version <tax> --taxonomy-manifest-sha256 <sha>
```

The curated version applies `annotation_overrides.parquet` and `image_blacklist.parquet`, which are package data in `packages/segdata-etl/src/segdata_etl/datasets/coralnet/curate/data/`. It stages tables only, on the image store of raw, with no plan file. Its manifest records the digest and fetch time of the API mapping. If the API is unreachable or returns an HTTP error, the command stops and stages nothing. Then publish it:

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset coralnet --version <curated>
```

`promote` publishes a version with carried image keys (no plan names them; a tables-only version carries every key) only when a published parent has the same image store. It HEAD-checks every carried key first.

### Find the published versions

| What | Where |
|---|---|
| `coralnet-raw` version | `s3://mermaid-ic-datasets/datasets/coralnet-raw/<version>/` |
| `coralnet` curated version | `s3://mermaid-ic-datasets/datasets/coralnet/<version>/` |
| `coralnet` v1 (`cleaned_up_rerun` labels) | `s3://mermaid-ic-datasets/datasets/coralnet/20261124-cfc389d/` (manifest sha256 in `configs/data_config.yaml`) |
| Taxonomy | `s3://mermaid-ic-datasets/taxonomy/mermaid-benthic/<version>/` |
| Original images | `s3://mermaid-ic-datasets/images/coralnet/s<source_id>/<image_id>.jpg` or `.png` |
| Staged versions | The same keys under `<stage>/mermaid-ic-datasets/` |

`coralnet@20261124-cfc389d` holds the labels the `cleaned_up_rerun` model trained on. Its manifest records `label_mapping.source = cleaned_up_rerun_json`. `configs/data_config.yaml` holds its pins. To reproduce that training set, pin that version; curate maps labels through the live API, so a new stage gives a different mapping.

Each version folder holds `manifest.json` and the tables. `segdata split build` takes the `coralnet` and `coralnet-raw` pins; see [CoralNet split](../../packages/segdata-etl/README.md#coralnet-split).

### When a publish fails

| Failure | What to do |
|---|---|
| Exit 2 with a usage error, for example `--parent-version and --parent-manifest-sha256 go together` | Fix the flags the message names. Nothing ran. |
| `FileNotFoundError: no git root holding uv.lock` | Run the command from a git checkout of this repository. |
| `PublishRefused: HEAD <sha> is not contained in any remote branch` | Push the branch and run again. A raw rerun probes only the images the checkpoint lacks. |
| `PublishRefused: working tree has uncommitted changes to tracked files` | Commit or discard the changes and run again. |
| `PublishRefused: ... is already published in mermaid-ic-datasets` (`promote`) | That version is already in the bucket. Nothing was written. Pin the published version. |
| `PublishRefused: taxonomy ... is not published` | The taxonomy pin is only staged or cached. [Publish the taxonomy](#publish-the-taxonomy) first. |
| `ObjectConflict` from `promote` (raw) | A destination key under `images/coralnet/` holds an object with a different ETag, size or CRC64NVME than the plan. Nothing was written. Find out what wrote it; `promote` does not replace it. |
| `PublishRefused: ... is not staged` or `... does not match the manifest` (`promote`) | The `--stage-dir`, `--dataset` or `--version` is wrong, or a staged file changed after staging. Stage the version again. |
| `SourceChanged` (raw stage) | An original was replaced between its HEAD and its read. Run the same command again. |
| `PublishRefused: a version with images no plan names, or neither copies.jsonl nor uploads.jsonl, needs a published parent` (`promote`) | The version has carried keys, as a curated version does, and no published parent. Publish the parent first, then stage the version with the parent pin. Nothing was written. |
| `PublishRefused` from `promote` (version with carried keys) with another message | The parent's image store differs, or a carried image key is missing or differs. Nothing was written. Publish the parent first. |
| `PublishRefused: plan keys differ from images.parquet` (`promote`) | `copies.jsonl` or `uploads.jsonl` names a key that is not in `images.parquet`, or names a key twice. Stage the version again. |
| `ValueError` from `build coralnet raw` naming carried images (size differs from the parent's, or "carried images have points outside the parent's size") | The scrape build disagrees with the published parent for an image it would carry. Each error names at most 10 ids. Do not carry the image. Investigate it. Restore or upload its original so the build probes it, or drop the parent pin. |
| `PinMismatch` | A pin's sha256 does not match the manifest in the store. Copy the pin again from the output of the command that published it. |
| `ValueError` from `curate` listing benthic attribute or growth form ids missing from the taxonomy | The API maps a label to an attribute or growth form that the pinned taxonomy lacks. Nothing was staged. Stage and publish a newer taxonomy, then pin it. |
| `curate` stops on a connection error or an HTTP error from `api.datamermaid.org` | The label mapping could not be fetched. Nothing was staged. Check network access, then run again. |
| HTTP or connection error from `api.datamermaid.org` (taxonomy) | Run again when the API answers. |

## When something fails

| Failure | What to do |
|---|---|
| Scraper exit 1: login failure | Check `CORALNET_USERNAME` and `CORALNET_PASSWORD`, then run again. |
| Scraper exit 1: a source failed with `cannot access source <id>: …` | The source is missing or private on CoralNet. Running again does not help. The failure does not stop the run. |
| Scraper stopped with `Run stopped early: circuit_open` (exit 1) | CoralNet is likely down or rejecting requests, so the [circuit breaker](../../packages/coralnet-scraper/README.md#retries-and-circuit-breaker) stopped the run. Failures such as `overview_forbidden` (HTTP 403) count toward the breaker. Wait, then check the transport tallies in the [run report](../../packages/coralnet-scraper/README.md#run-report). Run the same command again. If the tallies show 429 retries, lower `--requests-per-second`. |
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
| `build-images` reports `not_found` | The image object is not on S3. Run `invalidate --source-ids <id>` for its source and run the scraper without flags to upload it, then rebuild. For `coralnet-raw` with a parent pin whose published parent holds the image, no re-upload is needed: the build carries the image. |
| `build-images` reports `s3_error` or `worker_exception` | Rebuild. |
| `build-images` reports `corrupt` | The object on S3 is bad. Delete the corrupt image objects, run `coralnet-scrape invalidate --source-ids <id>`, then run `coralnet-scrape`. `invalidate` deletes the marker and `_partial/`, so the scraper runs the walk and the export again. Only the deleted images download again, because the image stage skips non-empty objects. `segdata build coralnet raw` drops the image until then. |

After a scrape, always rebuild under a new version tag.

## Known gaps

See the build's [Known gaps](../../packages/segdata-etl/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).
