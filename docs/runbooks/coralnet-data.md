# Recreate and update the CoralNet data

This runbook takes you from an empty bucket, or an older scrape, to a CoralNet build that training reads. It has two stages:

1. `coralnet-scraper` downloads the public CoralNet sources to S3.
2. `segdata build coralnet` builds the training manifest, the resized images and the label names from that scrape.

The package READMEs are the reference for flags, environment variables, exclusion kinds and exit codes; this page gives the order and links to them.

- [Before you start](#before-you-start)
- [Recreate the data from scratch](#recreate-the-data-from-scratch)
- [Update the data](#update-the-data)
- [Adopt an existing bucket](#adopt-an-existing-bucket)
- [Tune the request rate](#tune-the-request-rate)
- [Use the new build in training](#use-the-new-build-in-training)
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
- **Bucket.** Set `MERMAID_CORALNET_BUCKET` to the `source_bucket` of `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`). See [Use a build in training](../../packages/segdata-etl/README.md#use-a-build-in-training).

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
uv run --package segdata-etl segdata build coralnet all --version-tag <run> --upload-to-s3
```

To run the steps one at a time, pass the same `--version-tag` to each. Each invocation computes its own default tag, so without it the outputs land in different run directories. The step order and the flags of each step are in [Run](../../packages/segdata-etl/README.md#run).

Use an unused tag; see [Run](../../packages/segdata-etl/README.md#run).

### Check the build

1. Read the summary that each step logs: the excluded sources from `build-annotations` (it excludes a source with no `_complete.json` as `not_complete`), the count per `header_status` from `build-images`, and the resized, oversized, corrupted and failed counts from `resize`.
2. Check the exit code. Exit code 1 from `all` means at least one step reported failures. Every step still runs and writes its outputs.
3. Confirm that `etl-outputs/coralnet/<run>/` in the bucket holds four files:
   - `coralnet_annotations_<run>.parquet`
   - `coralnet_id2name_<run>.json`
   - `coralnet_images_<run>.parquet`
   - `coralnet_training_manifest_<run>.parquet`

The resized images are not in that directory. `resize` writes them under `<resized prefix>/resized/`, which every run shares. See [Outputs](../../packages/segdata-etl/README.md#outputs) and [Check a build for completeness](../../packages/segdata-etl/README.md#check-a-build-for-completeness).

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
uv run --package segdata-etl segdata build coralnet all --version-tag <new-run> --upload-to-s3
```

`build-annotations` and `build-images` read every source and every annotated image again. `resize` resizes only the images that have no resized object on S3.

Use an unused tag; see [Run](../../packages/segdata-etl/README.md#run).

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

## Use the new build in training

### Select the build

Set `manifest_path` on each CoralNet split (`train`, `val` and `test`) in `configs/data_config.yaml`:

```yaml
manifest_path: etl-outputs/coralnet/<run>/coralnet_training_manifest_<run>.parquet
```

Then regenerate the derived configs, starting with `configs/data_config_trainval.yaml`. See [Regenerate the derived configs](derived-configs.md). For the key layout, see [Use a build in training](../../packages/segdata-etl/README.md#use-a-build-in-training).

A new build changes model outputs, so item 7 of the working agreement in [AGENTS.md](../../AGENTS.md#2-working-agreement) applies. Run the evaluation, and put the metrics and the baseline they compare against in the PR.

### Add new sources to the splits

Each CoralNet split has a `whitelist_sources` list, and `CoralNetDataset` keeps only the sources on that list. A new source is not in any split until you add its id to a list:

1. Add the id to a split's `whitelist_sources` in `configs/data_config.yaml`. The id must already be in the manifest chosen under [Select the build](#select-the-build). Otherwise `CoralNetDataset` raises a `ValueError` that names it.
2. Regenerate the derived configs, starting with `configs/data_config_trainval.yaml`. See [Regenerate the derived configs](derived-configs.md).

3. A sync fails on any group that the split file does not list. To add sources without moving existing ones, append rows to `configs/splits/coralnet.csv` by hand. Each row has `group` `s<id>`, the `source_id`, its `split`, and a `split_group` that is either a new id not used elsewhere or blank.
   `segdata split build` re-runs the assignment over every source, so existing train, val and test membership changes, the test set included. Use it only to re-split on purpose. That is a change to model inputs, so it needs review. If you rebuild, copy `<out-dir>/coralnet.csv` and its meta over the committed files.

   Before you commit, run `segdata split leakage` on `configs/splits/coralnet.csv` against the new version; the full command is in [CoralNet split](../../packages/segdata-etl/README.md#coralnet-split).

### Map new labels

Training reads the CoralNet mapping from the MERMAID LabelMapping API (`/v1/classification/labelmappings/?provider=CoralNet`). Rows are keyed by CoralNet label id, and the build's `coralnet_id2name_<run>.json` names each id, so a label renamed in CoralNet still maps. A CoralNet label trains as background when the API has no row for its id, or when the row's `benthic_attribute_name` is null (`mermaidseg/dataset_reconciliation/label_mapping.py`, `mermaidseg/dataset_reconciliation/registry.py`). Before training, add the missing rows in MERMAID and fill in `benthic_attribute_name` on rows where it is null. There is no mapping file in this repo to edit.

Each run records the rows it used. `model.json` lineage has `data_version.label_mappings.coralnet` (source, sha256, row count, fetch time), and the tracker run has the rows as `metadata/label_mappings/coralnet.json`. An online run fails if the API call fails; it does not fall back to an old copy.

A change in the API changes the class set of the next run. Compare `metadata/id2label.json` with the baseline run before you compare metrics.

### Duplicates in new sources

`configs/coralnet/image_blacklist.parquet` and `configs/coralnet/annotation_overrides.parquet` cover only the sources they were built from. Duplicate images in new sources are not removed. See the build's [Known gaps](../../packages/segdata-etl/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).

### Warm the offline cache again

An offline run (`MERMAIDSEG_S3_OFFLINE=1`) fails on any cache miss. The new manifest, its id2name JSON and new images are not in a cache that you filled before. When `coralnet` is in the selected datasets, the prefetch also saves the CoralNet LabelMapping rows into the cache (`<cache>/_mermaid_api/classification/labelmappings/coralnet.json`). Only the prefetch writes that snapshot, and an online run does not refresh it. An offline run uses the snapshot and raises `DataLoadError` when it is missing, so fill the cache again to pick up mapping changes. If the API call fails, the prefetch raises and keeps the existing snapshot. Fill it again, using the same data config as the training run. The data config selects the build:

```bash
uv run python scripts/prefetch_cache.py --cache-dir /path/to/cache
```

`--data-config` defaults to `configs/data_config.yaml`. For the Slurm command, with `HF_HOME` and the final runs' data config, see [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).

## Publish CoralNet dataset versions

This procedure stages a CoralNet build locally, then publishes it as dataset versions in `s3://mermaid-ic-datasets/`: `coralnet-raw`, then `coralnet` v1, then `coralnet` v2. The flags, the tables and the resume rules are in [CoralNet dataset versions](../../packages/segdata-etl/README.md#coralnet-dataset-versions).

### Preconditions

- A build run in `etl-outputs/coralnet/<run>/`. See [Build the dataset](#build-the-dataset). `raw` stages `20260623_nogit` unless you pass `--build-run`.
- A git checkout of this repository. Commit and push before you stage: every stage and publish command refuses a tree with uncommitted changes to tracked files, and a HEAD that no remote branch contains. `raw` checks this before it probes any image.
- The environment from [Before you start](#before-you-start). `raw` requires `MERMAID_CORALNET_BUCKET` (it raises `ValueError` when unset) and reads `MERMAID_CORALNET_PREFIX`.
- AWS credentials that read the scrape bucket. `promote` and the taxonomy publish also need credentials that write `mermaid-ic-datasets`.
- Network access to `api.datamermaid.org` for the taxonomy and for v2.
- A local stage directory, `<stage>` below. Every `build` and `curate` command writes there and nothing to S3. Use one stage directory for the whole procedure.

Each stage command prints `version:` and `manifest_sha256:`. Record both: they are the pin that the next step takes. `promote` publishes the version you staged, under the same name, so the pin of a staged version is the pin of the published one.

Publishing is irreversible: the bucket denies overwrite and delete. Get approval before each `promote`.

### Stage a source subset and check parity

Stage a few sources and compare them with the old CoralNet loader before you publish anything. Pick the sources from the `coralnet` `whitelist_sources` in `configs/data_config.yaml`. This check uses a separate stage directory, `<check-stage>`, that you do not promote.

`check_dataset_parity.py` takes a `coralnet` version, so it checks v1 and v2. It cannot check `coralnet-raw`.

1. Stage a raw version of the sources:

   ```bash
   uv run --package segdata-etl segdata build coralnet raw --stage-dir <check-stage> \
     --source-ids <id>,<id>,... --build-run <run>
   ```

2. Stage a taxonomy version into the same directory:

   ```bash
   uv run --package segdata-etl segdata taxonomy --stage-dir <check-stage>
   ```

3. Stage v1 from the two pins. `--cache-dir <check-stage>` makes the command read the staged parents instead of S3:

   ```bash
   uv run --package segdata-etl segdata curate coralnet --label-map cleaned_up_rerun \
     --stage-dir <check-stage> --cache-dir <check-stage> \
     --raw-version <raw> --raw-manifest-sha256 <sha> \
     --taxonomy-version <tax> --taxonomy-manifest-sha256 <sha>
   ```

4. Stage v2 from the raw pin and the v1 pin:

   ```bash
   uv run --package segdata-etl segdata curate coralnet --label-map mermaid_api \
     --stage-dir <check-stage> --cache-dir <check-stage> \
     --raw-version <raw> --raw-manifest-sha256 <sha> \
     --parent-version <v1> --parent-manifest-sha256 <sha>
   ```

5. Copy `configs/data_config.yaml` to a path git ignores, for example `outputs/parity_data_config.yaml`. In the copy, cut the `whitelist_sources` of `coralnet` `train`, `val` and `test` to the staged sources of each split. The old loader reads every source in the list, so without this cut the old side holds sources that you did not stage.

6. Compare each version with the old loader, once for each split that holds staged sources:

   ```bash
   uv run python scripts/check_dataset_parity.py --no-sync --data-config outputs/parity_data_config.yaml \
     --dataset coralnet --split <split> --version <version> --manifest-sha256 <sha> \
     --cache-dir <check-stage> --resize-max-edge 2048
   ```

   - `--no-sync` reads the staged version from `--cache-dir` with no S3 call for it, and still checks the sha256 of every table.
   - `--resize-max-edge 2048` scales the new points into the old loader's 2048 px space.
   - The check compares points only. Native label names are compared lowercased.
   - The old loader reads its annotations from `dev-datamermaid-sm-sources`, so the default boto3 credentials must read that bucket.

   Expected result:

   | Version | Result |
   |---|---|
   | v2 | `PARITY OK`. Every line is `none`: images, points, native labels and mapped labels agree. The old loader maps labels through the live MERMAID API, as v2 does. |
   | v1 | Only the mapped labels differ, because v1 maps labels with `cleaned_up_rerun`. `mapped label counts` lists them. `points only in old` and `points only in new` also list points, because a point key includes its mapped label: the two lists hold the same positions with different mapped labels. Images and native label counts are `none`. The script prints `PARITY FAILED` and exits 1. |

The same command checks the full-corpus v1 and v2 that you stage below: pass the full stage directory as `--cache-dir`, with a data config that lists the whole corpus.

### Stage the raw version

Run this step on an EC2 host in `us-east-1`, not a laptop. A full corpus reads far more than a laptop link carries. Run it from the repository root of a checkout on that host:

```bash
uv run --package segdata-etl segdata build coralnet raw --stage-dir <stage> --build-run <run>
```

The command HEADs every kept original. It also reads in full each original that is multipart or has no CRC checksum, which is most scraper uploads, to compute the values the copy carries. It writes the tables and `copies.jsonl` under `<stage>`, with the `INTELLIGENT_TIERING` storage class for each copy. No image is copied and nothing is written to S3. `--workers N` sets the concurrency. To reuse the probe results of an earlier `coralnet-raw` version, stage that version under the same `<stage>` and add `--parent-version <v>` and `--parent-manifest-sha256 <sha>`. The command reads the parent from `<stage>`, not from S3.

The probe results are checkpointed under `<stage>/checkpoints/`. If the run stops, run the same command again with the same `--stage-dir`: it probes only the images the checkpoint lacks. See [Resume a raw stage](../../packages/segdata-etl/README.md#resume-a-raw-stage).

### Publish the raw version

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset coralnet-raw --version <raw>
```

`promote` copies the originals into `s3://mermaid-ic-datasets/images/coralnet/`, then writes the tables and `manifest.json`. See [Promote a version](../../packages/segdata-etl/README.md#promote-a-version).

### Publish the taxonomy

v1 resolves its labels against a `mermaid-benthic` taxonomy version on S3, so publish one before you stage v1. The steps are in [Publish the versioned datasets](training-datasets.md#publish-the-versioned-datasets): stage with `segdata taxonomy --stage-dir <stage>`, then publish with `--tables-from <staged taxonomy folder> --publish`. Record the pin that the publish prints. See [Taxonomy snapshot](../../packages/segdata-etl/README.md#taxonomy-snapshot).

### Stage and publish v1

Stage v1 against the published raw pin and the published taxonomy pin:

```bash
uv run --package segdata-etl segdata curate coralnet --label-map cleaned_up_rerun \
  --stage-dir <stage> \
  --raw-version <raw> --raw-manifest-sha256 <sha> \
  --taxonomy-version <tax> --taxonomy-manifest-sha256 <sha>
```

v1 applies `configs/coralnet/annotation_overrides.parquet` and `configs/coralnet/image_blacklist.parquet` from the checkout. It stages tables only, on the image store of raw, with no plan file. Then publish it:

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset coralnet --version <v1>
```

`promote` publishes a tables-only version only when a published parent has the same image store. It HEAD-checks every image key first.

### Stage and publish v2

```bash
uv run --package segdata-etl segdata curate coralnet --label-map mermaid_api \
  --stage-dir <stage> \
  --raw-version <raw> --raw-manifest-sha256 <sha> \
  --parent-version <v1> --parent-manifest-sha256 <sha>
```

Use the same raw pin as v1, and the pin of the published v1. The command stages the LabelMapping API rows it used as the extra file `coralnet_label_mappings.json`, which the manifest lists. Then publish it:

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset coralnet --version <v2>
```

`promote` uploads the manifest-listed extra file with the tables.

### Find the published versions

| What | Where |
|---|---|
| `coralnet-raw` version | `s3://mermaid-ic-datasets/datasets/coralnet-raw/<version>/` |
| `coralnet` v1 and v2 | `s3://mermaid-ic-datasets/datasets/coralnet/<version>/` |
| Taxonomy | `s3://mermaid-ic-datasets/taxonomy/mermaid-benthic/<version>/` |
| Original images | `s3://mermaid-ic-datasets/images/coralnet/s<source_id>/<image_id>.jpg` or `.png` |
| Staged versions | The same keys under `<stage>/mermaid-ic-datasets/` |

Each version folder holds `manifest.json` and the tables. A v2 folder also holds `coralnet_label_mappings.json`, which the manifest lists. `segdata split build` takes the `coralnet` and `coralnet-raw` pins; see [CoralNet split](../../packages/segdata-etl/README.md#coralnet-split).

### When a publish fails

| Failure | What to do |
|---|---|
| Exit 2 with a usage error, for example `--parent-version and --parent-manifest-sha256 go together` | Fix the flags the message names. Nothing ran. |
| `FileNotFoundError: no git root holding uv.lock` | Run the command from a git checkout of this repository. |
| `PublishRefused: HEAD <sha> is not contained in any remote branch` | Push the branch and run again. A raw rerun probes only the images the checkpoint lacks. |
| `PublishRefused: working tree has uncommitted changes to tracked files` | Commit or discard the changes and run again. |
| `PublishRefused: s3://.../manifest.json already exists` | Another run published that version name. Run again: the command takes the next free name. |
| `ObjectConflict` from `promote` (raw) | A destination key under `images/coralnet/` holds an object with a different ETag, size or CRC64NVME than the plan. Nothing was written. Find out what wrote it; `promote` does not replace it. |
| `PublishRefused: ... is not staged` or `... does not match the manifest` (`promote`) | The `--stage-dir`, `--dataset` or `--version` is wrong, or a staged file changed after staging. Stage the version again. |
| `SourceChanged` (raw stage) | An original was replaced between its HEAD and its read. Run the same command again. |
| `PublishRefused` from `promote` (tables-only v1 or v2) | No published parent has the same image store, or an image key is missing or differs. Nothing was written. Publish the parent first. |
| `PinMismatch` | A pin's sha256 does not match the manifest in the store. Copy the pin again from the output of the command that published it. |
| `ValueError: taxonomy has several benthic attributes named (ignoring case): [...]` (v1) | The pinned taxonomy has more than one attribute with a mapped name. v1 cannot be published against that taxonomy version. |
| `ValueError: the coralnet-raw pin ... is not a coralnet-raw parent of coralnet <v1>` (v2) | Pass the raw pin that the v1 manifest lists in `parents`. |
| `RuntimeError: coralnet would not reuse the bytes of <v1> for [...]` (v2) | The tables would not keep v1's bytes. Nothing was published. |
| HTTP or connection error from `api.datamermaid.org` (taxonomy or v2) | Run again when the API answers. v2 never uses an older mapping. |

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

See the build's [Known gaps](../../packages/segdata-etl/README.md#known-gaps) and [issue #231](https://github.com/data-mermaid/mermaid-segmentation/issues/231).
