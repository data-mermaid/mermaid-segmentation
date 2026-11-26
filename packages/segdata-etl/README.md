# segdata-etl

Publishes the datasets that segmentation training reads, as the versioned datasets and taxonomy that `segdata_spec` defines in `mermaid-ic-datasets`. It builds the CoralNet annotation and image tables from a scrape, loads four external raw releases, stages every dataset as a version, and builds the CoralNet split. Import package: `segdata_etl`; console script: `segdata`.

A new data source is an ETL module in `segdata_etl/datasets/` that publishes a version with its label mapping. `mermaidseg` does not change. See [Add a dataset](../../docs/runbooks/training-datasets.md#add-a-dataset).

| Command | Does |
|---|---|
| `segdata build coralnet <stage>` | Builds the CoralNet annotations and images tables from a scrape (see [CoralNet](#coralnet)). |
| `segdata build coralnet raw` | Stages a CoralNet build run as the `coralnet-raw` dataset version (see [Stage coralnet-raw](#stage-coralnet-raw)). |
| `segdata curate coralnet` | Stages a curated `coralnet` version from `coralnet-raw`, with labels mapped through the MERMAID API (see [Curate coralnet](#curate-coralnet)). |
| `segdata ingest <name>` | Loads a raw external release into the `external_validation_datasets/<name>/` prefix (see [Ingest an external dataset](#ingest-an-external-dataset)). |
| `segdata build <name>` | Builds an immutable versioned dataset from that prefix (see [Build a versioned external dataset](#build-a-versioned-external-dataset)). |
| `segdata build <uniform>` | Stages a uniform-format version of `mermaid`, `coralscapes_v2` or `ucsd_mosaics` on disk (see [Uniform datasets](#uniform-datasets)). |
| `segdata promote` | Uploads a staged dataset or taxonomy version to `mermaid-ic-datasets` (see [Promote a version](#promote-a-version)). |
| `segdata features attach` | Stages existing feature vectors as a feature set of a published version, for `segdata promote` (see [Attach a feature set](#attach-a-feature-set)). |
| `segdata taxonomy` | Stages the MERMAID benthic taxonomy as a version (see [Taxonomy snapshot](#taxonomy-snapshot)). |
| `segdata split build`, `segdata split leakage` | Builds and checks the CoralNet split (see [CoralNet split](#coralnet-split)). |

## Scope

**Owns:** turning a CoralNet scrape into the annotations and images tables and the id2name JSON; checking each raw external release and writing it to its S3 prefix; staging the external datasets, the MERMAID benthic taxonomy and the `mermaid`, `coralscapes_v2` and `ucsd_mosaics` versions, and promoting them to `mermaid-ic-datasets`. The CoralNet code is in `segdata_etl/datasets/coralnet/` (the CLI stages are in `build.py`). The curation inputs, `image_blacklist.parquet` and `annotation_overrides.parquet`, are package data in `segdata_etl/datasets/coralnet/curate/data/`. The raw-release ingest modules are in `segdata_etl/ingest/`. The versioned external and uniform dataset modules are in `segdata_etl/datasets/`, with the shared external steps in `datasets/external.py` and the curated label maps in `label_maps/curated/<name>.csv`. `promote` is in `segdata_etl/promote.py`, and `features attach` in `segdata_etl/features.py`. The taxonomy snapshot is `segdata_etl/taxonomy.py`. The shared helpers are in `segdata_etl/common/`: `images.py`, `jpeg_header.py`, `release.py` (the `ingest` S3 writes), `versions.py` (the image store and label counts that the versioned builds share), `labels.py`, `manifest.py`, and for the uniform builds `s3_objects.py`, `stage.py` and `hf_dense.py`:
  - `s3_objects.py`: write-once S3 copy and put. Each write sends `IfNoneMatch="*"`; a key that already holds the same content counts as done, and other content raises `ObjectConflict`. Also `make_s3_client`, the client every `segdata` command uses: adaptive retries, and for a command with `--workers` a connection pool of the worker count plus 4.
  - `stage.py`: the version naming and the default cache directory. `segdata_spec.stage` writes the staged files.
  - `hf_dense.py`: the shared build of a dense dataset from a pinned Hugging Face revision, used by `coralscapes_v2` and `ucsd_mosaics`.
**Does not own:** downloading CoralNet. That is [`coralnet-scraper`](../coralnet-scraper/README.md). Nor the MERMAID parquet or the Hugging Face datasets, except as the sources of the uniform `mermaid`, `coralscapes_v2` and `ucsd_mosaics` builds.
**Boundary:** The root `tests/test_package_boundaries.py` enforces these rules (run with the root suite): `segdata_etl` does not import `mermaidseg`; `mermaidseg` and `scripts` import neither `segdata_etl` nor `coralnet_scraper`; `coralnet_scraper` imports neither `mermaidseg` nor `segdata_etl`. Only `segdata_etl/datasets/coralnet/` may import `coralnet_scraper`, for its configuration and S3 key layout. The boundary test enforces this, so `common/`, `ingest/` and the external and uniform dataset modules stay free of it. The `segdata` CLI loads the CoralNet build code, and with it `coralnet_scraper` and `ibis`, only when a `build coralnet` stage runs. The argparse wiring is in `datasets/coralnet/commands.py`, which imports neither. A root `uv sync` does not install the package.

```bash
uv run --package segdata-etl segdata <command> ...
```

`<name>` is `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` or `benthos_yuval`. `<uniform>` is `mermaid`, `coralscapes_v2` or `ucsd_mosaics`. `build coralnet` takes a stage after it (see [Run](#run)). A staged uniform or external dataset version is published with `segdata promote`.

## CoralNet

Builds the CoralNet annotations and images tables, and the label names, from a scrape on S3. [CoralNet dataset versions](#coralnet-dataset-versions) stages them for training.

**End to end:** to recreate or update the CoralNet data, scrape through training, follow the [CoralNet data runbook](../../docs/runbooks/coralnet-data.md).

### Inputs

The build reads the per-source folders that the scraper writes under `s3://<bucket>/<prefix>/s<source_id>/`: `annotations.csv`, `image_list.csv`, `labelset.csv` and `images/<image_id>.jpg`. See the [scraper's S3 layout](../coralnet-scraper/README.md#s3-layout).

### Outputs

Each run has a version tag. The default tag is the UTC date and the short git commit, `YYYYMMDD_<sha>`, or `YYYYMMDD_nogit` outside a git checkout. `--version-tag` sets it.

Files are written locally to `outputs/coralnet-dataset/<version>/` (change the base with `--output-dir`):

| File | Written by | Content |
|---|---|---|
| `coralnet_annotations_<version>.parquet` | `build-annotations` | Confirmed point annotations of the included sources: `source_id`, `image_id`, `row`, `col`, `coralnet_id`, `status`. |
| `coralnet_id2name_<version>.json` | `build-annotations` | Label ID (as a string) to label name, merged from the included sources' `labelset.csv`. |
| `coralnet_images_<version>.parquet` | `build-images` | One row per annotated image, from its JPEG header: `source_id`, `image_id`, `s3_key`, `width`, `height`, `longest_edge`, `file_size`, `needs_resize`, `header_status`, `error_message`. `needs_resize` is true when the longest edge is more than 2048 pixels. |

With `--upload-to-s3`, each file is also uploaded to `s3://<bucket>/etl-outputs/coralnet/<version>/<file>`. An upload whose key already exists raises `RuntimeError` and stops the command, whether or not `--version-tag` is given. Pass `--overwrite` to replace existing keys.

### Run

Run from the repository root. `uv run --package segdata-etl` installs the package and its dependencies on first use:

```bash
uv run --package segdata-etl segdata build coralnet <stage>
```

Run the stages in this order, or run `all` to do both in one invocation with one version tag:

| Stage | Reads | Flags in addition to the common ones |
|---|---|---|
| `build-annotations` | The source CSVs on S3 | `--source-ids 1,2,3`: the sources to read. Default: every `s<id>/` folder under the prefix. |
| `build-images` | The annotations parquet, and the image headers on S3 | `--annotations <path>` (required) |
| `all` | As above | `--source-ids` |

Common flags, on every stage:

| Flag | Default | Effect |
|---|---|---|
| `--version-tag` | `YYYYMMDD_<sha>` | Run version tag. It names the run only. |
| `--overwrite` | off | Let an upload replace an existing key under `etl-outputs/coralnet/<version>/`. Without it, an upload to an existing key raises `RuntimeError`. A `coralnet-raw` stage that reads this run then reads different data. |
| `--output-dir` | `outputs/coralnet-dataset` | Local base directory. Files go under `<output-dir>/<version>/`. |
| `--upload-to-s3` | off | Upload each output file to `etl-outputs/coralnet/<version>/`. |
| `--workers` | `MERMAID_CORALNET_ETL_WORKERS`, else 16 | Concurrent S3 workers. Must be at least 1. |
| `-v`, `--verbose` | off | Debug logging. |

Each invocation computes its own default version tag. If you run the stages one at a time, pass the same `--version-tag` to each so the outputs share one run directory.

### Check a build for completeness

`build-annotations` excludes a source when it fails a check. Each excluded source has exactly one kind, the first check it failed:

| Kind | Meaning |
|---|---|
| `not_complete` | The source has no `_complete.json` [completion marker](../coralnet-scraper/README.md#resume). The scraper writes it only after it verifies the source, so the build reads none of the source's CSVs. |
| `annotations_missing` | `annotations.csv` is not on S3. |
| `image_list_missing` | `image_list.csv` is not on S3. |
| `labelset_missing` | `labelset.csv` is not on S3. |
| `csv_read_failed` | A CSV exists but could not be read. |
| `columns_missing` | A CSV lacks a required column. `annotations.csv` needs `Name`, `Row`, `Column`, `Label ID`; `image_list.csv` needs `Name`, `Image Page`; `labelset.csv` needs `Label ID`, `Name`. |
| `annotations_empty` | `annotations.csv` has no rows. |
| `image_list_incomplete` | An annotation row matches no `image_list.csv` row by `Name`. |
| `label_unnamed` | A Label ID used by the source is not named by any included source's labelset. |

A bucket filled by an older scraper has no markers. Run `coralnet-scrape verify-existing` first; see [Adopt an existing bucket](../../docs/runbooks/coralnet-data.md#adopt-an-existing-bucket). Otherwise every source is excluded as `not_complete` and `build-annotations` exits 1.

Rows with a non-numeric `Row`, `Column` or `Label ID` are dropped with a warning; they do not exclude the source. A Label ID with two different names across the included sources raises `ValueError` and stops the build.

At the end, `build-annotations` logs a summary at error level: the number of excluded sources, then one line per kind with the sorted source ids, for example `labelset_missing: [3, 4]`.

`build-images` logs the image count per `header_status`. `ok`, `sof_not_found_extended` and `pil_fallback` carry dimensions. `not_found`, `corrupt`, `s3_error` and `worker_exception` do not, and `segdata build coralnet raw` drops those images. A `not_found` image is carried instead when a parent pin holds its `image_id`; see [Stage coralnet-raw](#stage-coralnet-raw).

Exit codes:

| Stage | Exit code 1 when |
|---|---|
| `build-annotations` | Any source is excluded, or no annotation rows are built. |
| `build-images` | Any image has no dimensions. |
| `all` | Any step returns 1. Every step runs. |

Every stage also ends with a non-zero code on an uncaught error, such as invalid configuration or an upload refused as above. The output files are written before the exit code is set, so a run with exit code 1 still leaves its outputs.

#### Environment variables

Nothing loads `.env` for you: neither the package nor `uv run`. Copy the variables from `.env.example` into `.env`, then export them in your shell before the command:

```bash
set -a; source .env; set +a
```

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `MERMAID_CORALNET_BUCKET` | yes | none | Bucket that holds the scrape and the uploads. Shared with the scraper. |
| `MERMAID_CORALNET_PREFIX` | no | `coralnet-public-images` | Key prefix of the scrape. Shared with the scraper. |
| `MERMAID_CORALNET_ETL_WORKERS` | no | `16` | Concurrent S3 workers. Must be an integer of at least 1. `--workers` overrides it. |

S3 access uses the default boto3 credential chain, for example `AWS_PROFILE`.

### Known gaps

Duplicate images (the same photo in several sources) are removed only for the sources that `image_blacklist.parquet` lists. A source that is not in that file keeps its duplicates. Tracked in issue #231.

## CoralNet dataset versions

Two commands stage CoralNet as dataset versions in the [segdata-spec](../segdata-spec/README.md) format, and `segdata promote` publishes each one. For the order to run them in, see [Publish CoralNet dataset versions](../../docs/runbooks/coralnet-data.md#publish-coralnet-dataset-versions).

| Command | Stages | Built from |
|---|---|---|
| `segdata build coralnet raw` | `coralnet-raw`, with `copies.jsonl` | A build run under `etl-outputs/coralnet/<run>/`, and the original images of the scrape. |
| `segdata curate coralnet` | `coralnet`, tables only | A `coralnet-raw` version, a `mermaid-benthic` taxonomy version, and the MERMAID LabelMapping API. |

### Stage directory, version name and publish guard

These rules apply to both commands and to `segdata taxonomy --stage-dir`.

**Stage.** `--stage-dir` is a local directory. A command writes each object at `<stage>/mermaid-ic-datasets/<key>`, which is the segdata cache layout. No stage command writes to S3.
**Version name.** `YYYYMMDD-<sha7>`: the UTC date and the first 7 characters of the HEAD commit. If the stage or the bucket already holds that version of the dataset, the command takes the lowest free suffix: `-2`, then `-3`, and so on. The bucket is read to check the name.
**Publish guard.** `refuse_unpublishable` (`segdata_spec/git.py`) raises `PublishRefused` when tracked files have uncommitted changes, when no remote branch contains HEAD, or when the repository root has no `uv.lock`. The repository root is the nearest directory at or above the package that holds `.git` and `uv.lock`, so run the commands from a git checkout. `raw` checks the git state before it creates any S3 client, so it refuses an unpublishable tree before it probes any image.
**Pin.** On success the command prints `version: <version>` and `manifest_sha256: <sha256>`. These two values are the pin that the next command takes. `segdata promote` publishes the staged version under the same name, so the pin of a staged version is the pin of the published one.

### Stage coralnet-raw

`segdata build coralnet raw` stages one build run as a `coralnet-raw` version and plans the copy of the originals of its kept images.

```bash
uv run --package segdata-etl segdata build coralnet raw --stage-dir <stage> --build-run <run>
```

The command reads `coralnet_annotations_<run>.parquet`, `coralnet_images_<run>.parquet` and `coralnet_id2name_<run>.json` from `etl-outputs/coralnet/<run>/` in `MERMAID_CORALNET_BUCKET`, which is required: the command raises `ValueError` when it is unset. It reads each source's `image_list.csv` and lists its `images/` folder under `MERMAID_CORALNET_PREFIX`. See [Environment variables](#environment-variables).

| Flag | Default | Effect |
|---|---|---|
| `--stage-dir` | required | Local directory the version is staged in. |
| `--build-run` | `20260623_nogit` | Build run under `etl-outputs/coralnet/` to stage. |
| `--source-ids 1,2,3` | all | Keep only these sources of the build run. |
| `--parent-version`, `--parent-manifest-sha256` | none | A `coralnet-raw` version to reuse, staged under `--stage-dir`. Give both or neither. |
| `--workers` | `MERMAID_CORALNET_ETL_WORKERS`, else 16 | Concurrent S3 workers. |
| `-v`, `--verbose` | off | Debug logging. |

One flag of the parent pin without the other is a usage error (exit 2) before any S3 read.

The version holds:

| Table | Content |
|---|---|
| `images.parquet` | One row per kept image. `key` is `s<source_id>/<image_id>.jpg` or `.png` under the image store prefix `images/coralnet/`. `width` and `height` are the original's. `group` is `s<source_id>`. `source_fingerprint` is the original's S3 ETag. `source_name` is the image's name in `image_list.csv` without the status suffix, or null. |
| `points.parquet` | The build's annotations of the kept images, in the pixel space of the original. `label_id` is the CoralNet Label ID; `annotator` is `default`. |
| `labels.parquet` | One row for every ID in the build's id2name JSON, used or not. Native names only: the MERMAID columns are null and `mapping_source` is `identity`. |
| `stats/label_counts.parquet` | Points per image and label. |

An image is kept when it has dimensions in the images parquet, has at least one annotation, has its original on S3, and is a JPEG or PNG. The media type is JPEG when `header_status` is `ok`; otherwise the command reads the first 8 bytes of the original.

The command probes every kept original with a HEAD. It also reads an original in full when it is multipart or has no CRC64NVME checksum, which covers most scraper uploads, and computes the ETag, size and CRC64NVME its copy will carry. No image is copied. `copies.jsonl`, staged beside the tables, plans one server-side copy for each image except a carried one. With a parent pin, an image whose `image_id` and source ETag match a parent row keeps the parent's key and object fields, and is not probed. The new version lists the parent in `parents`.

With a parent pin, an image the build keeps whose scrape original is gone is carried. The original is gone when the image is absent from the scrape listing, or when its `header_status` is `not_found`. It is carried when the parent holds its `image_id`. The parent's `images.parquet` row is reused unchanged, its points come from the build, and no copy is planned: the object is already published. Without a parent pin, or when the parent lacks the `image_id`, the image is dropped as `source_missing` or `no_dimensions`. Two checks refuse a carried image with a `ValueError`. Each error names at most 10 image ids.

- A carried image that is absent from the listing, and whose build width and height differ from the parent's, fails the size check.
- A carried `not_found` image has no build dimensions, so it gets a point-bounds check instead: a point with `row` or `col` below 0, or at or beyond the parent's `height` or `width`, fails it. The message ends "carried images have points outside the parent's size".

Because a carried image needs no original, the scrape originals of already-published images can be deleted without breaking a later raw build. Delete in this order:

1. Publish the parent version.
2. Stage the new raw version with the parent pin.
3. Check `carried_from_parent` and `source_missing` in the staged manifest's `stats.dropped`.
4. Promote the new version.
5. Delete originals, only those whose `image_id` the published parent holds.

A raw version staged before the originals are deleted still plans copies for the images it reuses from the parent. Promote it before the deletion, or stage it again after.

Run a full corpus on an EC2 host in `us-east-1`. The full reads dominate the run time and the transfer.

`stats.dropped` in the manifest:

| Key | Unit | Counts |
|---|---|---|
| `no_dimensions` | images | Images whose `header_status` carries no dimensions, except carried images. |
| `no_annotations` | images | Images with dimensions and no annotation. |
| `annotations_without_image` | annotation rows | Annotations whose image is not in the images parquet. |
| `source_missing` | images | Kept images whose original is not in the source's `images/` folder and that the parent does not hold. |
| `carried_from_parent` | images | Images kept from the parent's row with no original. Present whenever a parent pin is given. |
| `unsupported_format` | images | Originals that are neither JPEG nor PNG. |
| `source_name_missing` | images | Published images with a null `source_name`. These images stay in the version. |

#### Resume a raw stage

The probe results are checkpointed in `<stage>/checkpoints/coralnet-raw-<build-run>-<sha7>.parquet`, where `<sha7>` is the first 7 characters of the HEAD commit. A rerun of the same command with the same `--stage-dir` and commit probes only the images that the checkpoint lacks.

The checkpoint is written after every 20th batch of 1000 probes, and once after the last batch. A crash loses the probes since the last write.
A checkpoint row wins over a parent row for the same image.
A new commit starts a new checkpoint.

### Curate coralnet

`segdata curate coralnet` stages a `coralnet` version from pinned parents. Training reads its `labels.parquet` for the label mapping. The version holds tables only, on the image store of raw: no plan file is staged, and no image is read or copied. It maps CoralNet labels only through the MERMAID LabelMapping API, which it fetches at stage time. The command needs network access to `api.datamermaid.org`.

The published `coralnet@20261124-cfc389d` holds the labels the `cleaned_up_rerun` model trained on. Its manifest records `label_mapping.source = cleaned_up_rerun_json`. `configs/data_config.yaml` holds its pins. To reproduce that training set, pin that version; curate maps labels through the live API, so a new stage gives a different mapping.

```bash
uv run --package segdata-etl segdata curate coralnet \
  --stage-dir <stage> \
  --raw-version <v> --raw-manifest-sha256 <sha> \
  --taxonomy-version <v> --taxonomy-manifest-sha256 <sha>
```

| Flag | Default | Effect |
|---|---|---|
| `--stage-dir` | required | Local directory the version is staged in. |
| `--raw-version`, `--raw-manifest-sha256` | required | Pin of the `coralnet-raw` parent. |
| `--taxonomy-version`, `--taxonomy-manifest-sha256` | required | Pin of the `mermaid-benthic` taxonomy. The pin must be published in `mermaid-ic-datasets`, even when `--cache-dir` already holds it. |
| `--cache-dir` | `outputs/segdata-cache` | Where the parent tables and the taxonomy are synced from S3. The command reads this directory first, so `--cache-dir <stage>` reads a raw parent that is only staged. The taxonomy pin must be published: a taxonomy that is only staged is refused. No image is synced. |
| `-v`, `--verbose` | off | Debug logging. |

A missing pin flag, or one flag of a pin without the other, is a usage error (exit 2). The message names the flags.

The version has one parent, the raw version, and uses its image store. The taxonomy is an input to the label mapping, not a parent.

1. Overrides. For each image in `annotation_overrides.parquet` (package data in `segdata_etl/datasets/coralnet/curate/data/`), the override points replace all of that image's raw points. Override points are in the pixel space of the image scaled to a longest edge of 2048. For an original whose longest edge is more than 2048, each point is mapped back to original coordinates: a raw point with the same label that scales to the same position gives its exact coordinates; otherwise the command takes the smallest original coordinate that scales to the override position. Overrides of smaller images are used as they are.
2. Blacklist. The command drops the images listed by `image_id` in `image_blacklist.parquet`, in the same directory, and their points.
3. Labels. One row per label of the raw version. The command fetches `/v1/classification/labelmappings/?provider=CoralNet` from `api.datamermaid.org` and joins the rows to the raw labels by CoralNet label id. A label with no API record, or a record with no target, stays unmapped: its MERMAID columns are null. `mermaid_growth_form` holds the API record's growth form name, or null when the record has none. Each mapped record must have a benthic attribute id, and its benthic attribute id and growth form id must exist in the pinned taxonomy. Otherwise the command raises `ValueError` listing the labels or ids, and stages nothing. Stage and publish a newer taxonomy, then pin it. An unreachable API or an HTTP error also stops the command with nothing staged.

The manifest records the mapping under `label_mapping`: `source` is `mermaid_api`, `sha256` is the digest of the fetched mapping, and `fetched_at`, `n_labels` and `n_mapped` give the fetch time, the raw label count and the mapped count. A later run can fetch a different mapping, so two versions staged from the same pins can differ in labels.

`stats.dropped` of a curated `coralnet` version:

| Key | Unit | Counts |
|---|---|---|
| `blacklisted_images` | images | Raw images dropped by the blacklist. |
| `override_images_not_in_raw` | images | Distinct override images that the raw version does not hold. Their overrides are dropped. |

## CoralNet split

`segdata split build` assigns the curated CoralNet dataset to train, val and test by source group, and lists excluded sources as `excluded`. `segdata split leakage` checks a split file for images shared across splits. Both run only on demand: when you re-split on purpose, or check a version. Training never runs them. The committed result is `configs/splits/coralnet.csv` with its provenance in `configs/splits/coralnet.meta.yaml`. The committed `coralnet.meta.yaml` records the provenance of the committed split (whitelists, builder commit, manual overrides), not the `pins` and `inputs` layout that `build` writes. The file format and coverage check are in [segdata-spec](../segdata-spec/README.md#splits).

Both commands read pinned versions from `--bucket` (default `mermaid-ic-datasets`) through `--cache-dir`. They sync tables only, never images. Each needs `--version` and `--manifest-sha256` for the `coralnet` dataset.

### Build a split

`build` groups sources that must stay together: sources that share a byte-identical image in the raw parent version, sources with a near-duplicate pair labelled `dup` or `unsure`, and sources of one project (a shared admin, with nearby source IDs or locations). Groups of the `--test-sources` go to test. A group may go to val or test only if none of its sources shares an image with another source and it is small enough: the group's images must not exceed `--max-group-frac` of the image target of the smaller eval split. `--val-frac` and `--test-frac` are shares of images. Simulated annealing assigns the eligible groups, and the best of `--restarts` runs wins.

Required inputs:

| Flag | Content |
|---|---|
| `--version`, `--manifest-sha256` | Pin of the curated `coralnet` dataset. |
| `--raw-version`, `--raw-manifest-sha256` | Pin of the `coralnet-raw` parent. |
| `--taxonomy-version`, `--taxonomy-manifest-sha256` | Pin of the `mermaid-benthic` taxonomy. |
| `--cache-dir` | Local segdata cache. |
| `--near-dup-labels` | Near-duplicate pair labels CSV. |
| `--metadata` | CSV with `source_id`, `admins`, `latitude`, `longitude`. |
| `--test-sources` | Source IDs fixed to test, one per line. |
| `--exclude-sources` | Source IDs kept out of train, val and test, one per line. Each one that is in the curated version is written to the split file as `excluded`. |
| `--training-config` | Training config. Its `training.class_subset` is the class space. |
| `--out-dir` | Output directory. |

The tuning flags are `--val-frac`, `--test-frac`, `--max-group-frac`, `--max-id-gap`, `--max-km`, `--min-class-points`, `--min-eval-points`, `--w-coverage`, `--w-size`, `--iters`, `--restarts` and `--seed`. Run `segdata split build --help` for the defaults.

Outputs, all in `--out-dir`:

| File | Content |
|---|---|
| `coralnet.csv` | The split file: `group`, `source_id`, `split_group`, `split`. `group` is `s<source_id>`. `split` is `train`, `val`, `test` or `excluded`. An `excluded` row has a blank `split_group`, and a blank `split_group` means none. An id in `--exclude-sources` that is not in the curated version has no row. |
| `coralnet.meta.yaml` | Pins, input file paths and sha256, excluded sources, parameters, git SHA and a summary. |
| `class_distribution.csv` | Points per class in each split. |
| `summary.json` | Sources, groups, images and points per split, plus classes missing from or thin in val and test. |

`build` writes its outputs only to `--out-dir` and the synced tables to `--cache-dir`. Point `--out-dir` at a scratch directory, not `configs/splits/`, then copy the reviewed output in by hand. The change is then reviewed in git.

```bash
uv run --package segdata-etl segdata split build \
  --version <v> --manifest-sha256 <sha> \
  --raw-version <v> --raw-manifest-sha256 <sha> \
  --taxonomy-version <v> --taxonomy-manifest-sha256 <sha> \
  --cache-dir <cache> \
  --near-dup-labels <labels.csv> --metadata <metadata.csv> \
  --test-sources <test.txt> --exclude-sources <exclude.txt> \
  --training-config <training.yaml> \
  --out-dir <scratch-dir>
```

`build` stops with a `ValueError` before it writes any output file when:

The `--raw-version` and `--raw-manifest-sha256` pin is not a `coralnet-raw` parent that the `coralnet` manifest declares. The error lists the declared parents.
A raw image of a source in the split universe has no `crc64nvme`, so it cannot be matched as a duplicate. The universe is the curated sources minus `--exclude-sources`. The error gives the count and the sorted source ids.
A `--test-sources` id is not in the universe (for example, it is also in `--exclude-sources`). The error lists the sorted ids.

### Check a split for leakage

`leakage` reads a split file and the `images.parquet` of the pinned `coralnet` version. Two images are the same when their `crc64nvme` and `size` match. Groups the split file does not list, and rows with `split` `excluded`, are ignored. The check reports:

images of val that also appear in train, and images of test that also appear in train or val, per group;
pairs of groups in different splits that share an image;
`split_group` values that span more than one split. Blank values are skipped.

The command prints the report. With `--out-dir` it also writes `leakage_per_source.csv`, `leakage_pairs.csv` and `leakage_split_groups.csv`. It exits 1 if any of the three findings is non-empty, and 0 otherwise.

```bash
uv run --package segdata-etl segdata split leakage \
  --split-file configs/splits/coralnet.csv \
  --version <v> --manifest-sha256 <sha> \
  --cache-dir <cache> --out-dir <report-dir>
```

`leakage` stops with a `ValueError` before it prints the report when any image of a group that the split file lists has no `crc64nvme`. The error gives the count and the sorted group names.

## Uniform datasets

`segdata build mermaid`, `segdata build coralscapes_v2` and `segdata build ucsd_mosaics` stage a dataset version in the format of [segdata-spec](../segdata-spec/README.md). A build never writes to S3. `segdata promote` uploads a staged version to `mermaid-ic-datasets`.

### Stage a version

Run from the repository root. S3 access uses the default boto3 credential chain. Each build reads the bucket to check that its version name is free.

```bash
uv run --package segdata-etl segdata build <dataset> --stage-dir <stage> ...
```

The build writes the version to `<stage>/mermaid-ic-datasets/datasets/<dataset>/<version>/`. This is the [cache layout](../segdata-spec/README.md#cache-layout) that segdata `sync` writes.

The version is `YYYYMMDD-<sha7>`: the UTC date and the first seven characters of the git commit. If that version is already staged, or any object exists under `datasets/<dataset>/<version>/` in `mermaid-ic-datasets` (a half-promoted version included), the build appends `-2`, `-3` and so on until the name is free.

Flags on every uniform build:

| Flag | Default | Effect |
|---|---|---|
| `--stage-dir` | none (required) | Local directory the version is staged in. |
| `--workers` | 16 | Concurrent S3 reads. Only `mermaid` uses it. |
| `-v`, `--verbose` | off | Debug logging. |

Each dataset:

| Dataset | Flags | Source | Group | Plan file |
|---|---|---|---|---|
| `mermaid` | `--annotation-date-cutoff YYYY-MM-DD` (required): keep annotations with `updated_on` on or before that date (UTC); the whole day is included. `--annotations-uri` (default `s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet`). `--parent-version` and `--parent-manifest-sha256`, given together: a staged version whose image rows are reused when the source ETag is unchanged. `--extractor-spec` (required), `--feature-family` and `--allow-missing-features`: see [MERMAID feature set](#mermaid-feature-set). | Images `mermaid/<image_id>.png`, else `.jpg`, in `coral-reef-training`. | `mermaid` for every image. | `copies.jsonl`; `mermaid` also a feature set |
| `coralscapes_v2` | `--revision` (default `8a4567b2fbec14882e09aebff63a7a1ca5e24dca`) | `josauder/coralscapesV2`, `instances` config. Licence `apache-2.0`. | The HF split: `train`, `validation` or `test`. | `uploads.jsonl` |
| `ucsd_mosaics` | `--revision` (default `6a6483841f8ee20bd7d8ee2aa760f10d1e9db704`) | `josauder/UCSD-mosaics-mirror`. Licence CC BY-NC 4.0, under UC San Diego Library's copyright terms. | The `site` column, lowercased. | `uploads.jsonl` |

`mermaid` is a points dataset. Its labels are the BA+GF pairs, numbered from 1 and named `BA::GF`. The trailing `::` stays when there is no growth form. The plan file `copies.jsonl` lists one server-side copy per image, from `coral-reef-training` to `mermaid-ic-datasets`. The build stages no image bytes.

With `--parent-version`, a pair keeps the `label_id` it has in the parent's `labels.parquet`, matched on benthic attribute id and growth-form name, and a new pair is numbered from the parent's highest id plus 1. Ids are stable against the direct parent only: a pair with no points in a version leaves that version's `labels.parquet`, so a later version can reuse its id for a different pair. The match uses the growth-form name, so renaming a growth form in MERMAID gives its pairs new ids.

#### MERMAID feature set

`build mermaid` also stages a feature set from the pyspacer feature vectors that the MERMAID platform keeps beside each image, `s3://coral-reef-training/mermaid/<image_id>_featurevector`. The build gathers the vectors as [`features attach`](#attach-a-feature-set) does, with that path as its source template: a vector already in the family is adopted, and every vector is read and checked before anything is staged. The build writes nothing to S3. The layout of a feature set is in the [segdata-spec README](../segdata-spec/README.md#feature-sets).

| Flag | Default | Effect |
|---|---|---|
| `--extractor-spec` | none (required) | JSON file with `class_path`, `crop_size`, `feature_dim`, `weights_uri` and `weights_sha256`. `class_path`, `crop_size`, `feature_dim` and `weights_sha256` identify the extractor; the other keys are stored with the spec. |
| `--feature-family` | `efficientnet-v1` | Family the set is published under, a lowercase key segment. |
| `--allow-missing-features` | off | Keep an image that has no source vector, and leave it out of the feature index. Without the flag, the build stops and names the images. |

A vector that is present but refused (unreadable, wrong width, or missing a point) stops the build whether or not the flag is set. A build that stops stages nothing. The feature set goes under `<stage>/mermaid-ic-datasets/datasets/mermaid/<version>/features/<family>/`:

| File | Content |
|---|---|
| `index.parquet` | The feature index: one row per vector, adopted or to copy. |
| `manifest.json` | The set manifest. It pins the staged dataset manifest and records the extractor spec. Promote publishes these bytes unchanged. |
| `copies.jsonl` | One server-side copy per vector that is not already in the family, from `coral-reef-training` to `features/<store>/<family>/<image_id>.featurevector` in `mermaid-ic-datasets`. The lines have the shape of the version's `copies.jsonl`. |

`coralscapes_v2` and `ucsd_mosaics` are dense datasets. The build stages each image and mask byte for byte under `<stage>/mermaid-ic-datasets/images/<dataset>/`, and `uploads.jsonl` lists each staged file. Their labels map to MERMAID benthic attributes through the curated CSVs in `segdata_etl/label_maps/curated/` (`coralscapes_v2.csv`, `ucsd_mosaics.csv`; columns `label_id`, `name`, `mermaid_benthic_attribute`).

The build fails when no file in the pinned revision matches the shard pattern. It also fails when a `.parquet` under the shard directory does not match the pattern. The shard directory is `instances/` for `coralscapes_v2` and `data/` for `ucsd_mosaics`.

The manifest's `stats.dropped` counts what the build left out:

| Dataset | Key | Count |
|---|---|---|
| `mermaid` | `annotation_after_cutoff` | Annotations updated after the cutoff. |
| `mermaid` | `image_deleted_at_source` | Images with neither a `.png` nor a `.jpg` object. |
| `mermaid` | `points_of_deleted_images` | Points of those images. |
| `mermaid` | `image_unreadable_header` | Images whose size cannot be read from the header. |
| `coralscapes_v2`, `ucsd_mosaics` | `image_unreadable_header` | Rows whose image is not a JPEG or PNG with a readable size, or whose mask is not a PNG. |
| `coralscapes_v2`, `ucsd_mosaics` | `mask_invalid` | Rows whose mask cannot be decoded, differs in size from the image, is not single-channel, or holds an id outside the vocabulary. |

### Promote a version

Publishing is irreversible. `mermaid-ic-datasets` denies overwrite and delete, so a promoted version stays as it is. Get explicit approval before you run `promote`.

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset <dataset> --version <version>
```

`promote` also takes `--workers` (default 16) and `-v`, `--verbose`.

A staged version has one of three shapes:

- A version with `copies.jsonl`, `uploads.jsonl` or both, whose plans name every key in `images.parquet`. `copies.jsonl` plans server-side copies of images from S3; `uploads.jsonl` plans uploads of staged files. `promote` writes every image and mask in the `INTELLIGENT_TIERING` storage class; tables and the manifest keep the bucket default.
- A version whose plans leave some keys unnamed. An unnamed key is carried: its image is already published. `promote` accepts carried keys only when the version has a parent on the same image store.
- A tables-only version, with neither file. Every key is carried.

The version folder can also hold feature sets, one folder for each family under `features/`. Each one is a unit of its own; see [Feature sets in promote](#feature-sets-in-promote).

`promote` writes in this order:

1. Preflight, before any write. It checks the staged tables against `manifest.json`, and refuses a manifest whose `stats.n_images` is 0. It checks that `copies.jsonl` and `uploads.jsonl`, whichever are staged, together name each image and mask key in `images.parquet` at most once and no other key, with the same size and checksums, and checks each staged upload file. Every key that no plan names is carried. For a version with carried keys, it checks that a published parent has the same image store, then HEADs each carried key in the dataset bucket: each must exist with the row's ETag, size and CRC64NVME. It refuses a version whose `manifest.json` is already in the bucket with other bytes. When the manifest pins a taxonomy, it refuses the version unless that taxonomy's `manifest.json` is in the bucket with the pinned sha256. It then HEADs every planned destination key: each image and mask key, and each table key under `datasets/<dataset>/<version>/`. It refuses when a key already exists with a different ETag, size or CRC64NVME, and names up to five such keys. A key that exists with a matching fingerprint passes, so a rerun after a partial failure completes. The feature-set checks are under [Feature sets](#feature-sets-in-promote).
2. The images and masks: server-side copies for `copies.jsonl`, uploads for `uploads.jsonl`. A tables-only version writes none, and a carried key is never written.
3. The tables.
4. `manifest.json`.
5. Each staged feature set, in family order (see below).

Every write is write-once. A key that already holds the planned content counts as done, and other content stops the run with `ObjectConflict`. A version is visible only when its `manifest.json` exists. If `promote` stops, run it again: it continues from the objects already written.

A unit whose `manifest.json` is already published with the staged bytes is done, and `promote` skips it. A rerun therefore writes only what is left, for example the feature sets after the dataset. A unit published with other bytes is refused. When every staged unit is already published, `promote` refuses the version as already published.

#### Feature sets in promote

Each `features/<family>/` folder is a feature set. `promote` refuses a family folder that has no `manifest.json`, because the stage stopped before the set was complete. `segdata_spec.stage.stage_feature_set` wrote it and already checked the index against the version (see the [segdata-spec README](../segdata-spec/README.md#feature-sets)). A stage can hold feature sets without the dataset files: `promote` then reads the dataset manifest from the bucket, and refuses when the version is not published.

Preflight refuses, before any write, when:

- The set manifest pins another dataset manifest than the staged one, or than the published one when no dataset is staged.
- `index.parquet` differs from its entry in the set manifest (`ChecksumMismatch`).
- A line of the family's `copies.jsonl` names a key that is not in the index, names a key twice, names a bucket other than `mermaid-ic-datasets`, or differs from its index row in ETag, size or CRC64NVME.
- An index key that no line names is not already in `mermaid-ic-datasets` with its row's ETag, size and CRC64NVME. Such a key is carried, as a carried image is.
- The family's `_extractor_spec.json` is already in the bucket and names another extractor than the set manifest: `ExtractorMismatch`, listing each differing identity field. Stage the set under a new family name instead.
- A planned vector key or the index key already holds other content.

For each set, `promote` claims the family with `segdata_spec.features.claim_extractor`, which writes `_extractor_spec.json` from the set manifest's extractor. It then copies the vectors in the `STANDARD` storage class, puts `index.parquet`, and puts the set's `manifest.json` last.

## Attach a feature set

`segdata features attach` stages feature vectors that already exist as a feature set of a published dataset version. `segdata promote` then publishes the set. An image whose file already exists in the family, at `features/<store>/<family>/<image_id>.featurevector` in `mermaid-ic-datasets`, adopts that file. Every other image's vector is copied in from the source that the template names. The layout of a feature set is in the [segdata-spec README](../segdata-spec/README.md#feature-sets). Use it for a version that was published without features; `build mermaid` stages one with the version. The command is in `segdata_etl/features.py`, and the shared gathering code in `segdata_etl/common/featurevector.py`.

```bash
uv run --package segdata-etl segdata features attach <dataset> \
  --version <v> --manifest-sha256 <sha> \
  --family <family> --source-template <s3://bucket/key-template> \
  --extractor-spec <spec.json> --stage-dir <stage>
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset <dataset> --version <v>
```

The command syncs the version's tables (not its images) to `--cache-dir` and writes nothing to S3. It reads the sources, the family files and the family's `_extractor_spec.json`. It refuses a dirty or unpushed git tree before any S3 call, and records the command line in the set's manifest as its ETL provenance.

| Flag | Default | Effect |
|---|---|---|
| `dataset` | required | Published dataset name. |
| `--version`, `--manifest-sha256` | required | Pin of the published version. The command stops with `PinMismatch` when the published manifest differs. |
| `--family` | required | Feature family, a lowercase key segment. Published sets are add-only, so a corrected set needs a new family. |
| `--source-template` | required | `s3://bucket/key` template that names the source vector of each image that adopts no family file. |
| `--extractor-spec` | required | JSON file with `class_path`, `crop_size`, `feature_dim`, `weights_uri` and `weights_sha256`. |
| `--stage-dir` | required | Local directory the set is staged in, under the version's folder. Pass the same directory to `promote`. |
| `--allow-missing` | off | Leave out of the set an image that has neither a family file nor a template source. Without it, such images stop the command. |
| `--workers` | 16 | Concurrent S3 requests. |
| `--cache-dir` | `outputs/segdata-cache` | Local cache for the synced tables. |
| `-v`, `--verbose` | off | Debug logging. |

### Source template

The template is a Python format string. Its fields are the `images.parquet` row's `image_id`, `key` and `group`, plus every key of the row's `attrs` JSON; a field that is none of these stops the command before any source call. The template is resolved only for images that adopt no family file. Fields must be plain names (no positional, indexed or dotted fields), and the result must be an `s3://bucket/key` URI. A template that resolves two of those images to the same source object is refused.

| Source | Template |
|---|---|
| MERMAID | `s3://coral-reef-training/mermaid/{image_id}_featurevector` |

CoralNet has no template source. mermaid-classifier's `scripts/build_feature_set.py` writes CoralNet vectors straight into the family, so a CoralNet attach adopts every image's family file and never resolves the template.

### Phases

1. Refuse a set that is already published. Read the version's `images.parquet` and `points.parquet`; a version without `points.parquet` is refused.
2. Read the family's `_extractor_spec.json`, if it exists. Stop with `ExtractorMismatch` when it names another extractor.
3. HEAD each image's family file in `mermaid-ic-datasets`. An image whose file exists adopts it: its template source is never called and nothing is copied for it.
4. Resolve the template for the other images and HEAD their sources. Absent sources are collected. Without `--allow-missing` they stop the command, which names the count and the first ten image ids.
5. For each `images.parquet` group in turn, GET the vectors of that group's images, adopted files and template sources alike, guarded by the HEAD ETag, and verify them. Only one group's points are held in memory at a time.
6. Stage the set: `index.parquet` with adopted and copied files together, the set's `manifest.json`, and `copies.jsonl` with one write-once copy for each verified template source, guarded by the source ETag.

### What is verified

A vector is usable when it is a readable npz, its `feat` array is `feature_dim` wide, and it holds every `(row, col)` point that `points.parquet` lists for its image. An adopted file must also have the ETag, size and CRC64NVME of its bytes, as a copy must; its index row records them. A file without a CRC64NVME checksum is refused: writers put feature files with `ChecksumAlgorithm='CRC64NVME'`. The command reads all vectors and reports every refused vector, by source key and reason, in one error before it stages anything. A source that changes between its HEAD and its read stops the command with `SourceChanged`.

With `--allow-missing`, the images with neither a family file nor a source vector are counted in the set's `n_missing` and have no index row.

### Resume

`promote` resumes the set as it resumes a dataset: after a stop, run it again, and the copies already in place count as done. A staged set is never staged again; to restage it, delete its folder from the stage directory first. A rerun of `attach` with a new stage adopts the family files that an earlier promote copied.

## Ingest an external dataset

`segdata ingest <name>` loads the raw release of an external dataset into S3. `<name>` is `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` or `benthos_yuval`. The code is in `segdata_etl/ingest/`. It writes the `external_validation_datasets/<name>/` prefix, which training does not read. [Build a versioned external dataset](#build-a-versioned-external-dataset) reads that prefix and publishes the version that training reads. The [External datasets](#external-datasets) section lists the raw files to download for each release.

### Published files

Every ingest command writes the same set of files under `s3://<bucket>/<prefix>/`:

| Key | Content |
|---|---|
| `<name>_annotations.parquet` | The annotations. The columns differ for each dataset; `manifest.json` lists them. |
| `images/...` | The images. The key layout differs for each dataset; `manifest.json` gives it as `s3.image_key_template`. |
| `labels/<site>/<image_id>.png` | Benthos only. Dense label masks, with pixel values from `classes.json`. |
| `classes.json` | Label name to class ID. `unlabeled` is 0. The other names are sorted alphabetically and numbered from 1. |
| `colors.json` | Label name to an RGB triple. `unlabeled` is `[0, 0, 0]`. The colors come from a generator with seed 1337, so a rerun gives the same colors. |
| `manifest.json` | A description of the published dataset (below). |

`manifest.json` has these keys:

| Key | Content |
|---|---|
| `dataset_name` | The dataset name. |
| `source.citation`, `source.raw_data_url`, `source.license` | Where the raw release comes from, and its licence. |
| `created_at_utc` | When the run wrote the manifest. |
| `s3.bucket`, `s3.prefix`, `s3.annotations_parquet`, `s3.image_key_template` | Where the files are. |
| `schema.parquet_columns` | Each parquet column and what it holds. |
| `ingestion` | Facts about this run that differ for each dataset (see [Ingestion keys](#ingestion-keys)). |

The defaults are `--bucket dev-datamermaid-sm-sources` and `--prefix external_validation_datasets/<name>`.

### Write order and safety

A run writes in this order:

1. Before it reads the raw data, the run checks that the parquet, `classes.json`, `colors.json` and `manifest.json` are not on S3. If one is there, the run stops with `FileExistsError`. Pass `--overwrite` to replace them.
2. The run reads the raw release and checks that every image decodes. JPEGs are decoded at reduced scale (about 1/8 per edge) to bound memory. The full scan data is still read, so truncated JPEGs are still caught. Other formats are fully decoded. Catlin Seaview and Moorea drop each damaged image and its annotations, and record it in `ingestion.damaged_images`. An image over Pillow's decompression-bomb limit (about 179 megapixels) stops the run with `DecompressionBombError` before anything is written to S3. Benthos lifts that limit only while it decodes its two mosaics. Pacific stops with an error before it writes to S3.
3. The run uploads the images. An image whose key is already on S3 is skipped, whatever its content. Pass `--no-skip-existing-s3` to upload every image again. `--overwrite` does not change how images are uploaded, except for the Benthos `labels/` tiles: with `--overwrite`, Benthos uploads them again even if their keys exist. Label pixel values are the IDs in `classes.json`, so a rewritten `classes.json` must not keep old masks.
4. The run writes the parquet, then `classes.json`, then `colors.json`, then `manifest.json`.

`manifest.json` is written last, so a prefix without it holds an incomplete run. If a run stops before step 4, run it again: the images already on S3 are skipped. If it stops during step 4, run it again with `--overwrite`.

`--dry-run` writes nothing to S3. It still reads S3 for the step 1 check and for the list of existing images, so it needs credentials and fails where the real run fails. It does not stop local work: Catlin Seaview still downloads its zips, and Benthos still writes its tiles.

### Ingestion keys

Every dataset records `images_uploaded`: the number of images this run sent to S3, not counting images skipped because their keys exist.

| Dataset | Keys |
|---|---|
| `catlin_seaview` | `damaged_images`: the dropped images. `images_uploaded`: a count for each region. |
| `moorea_labeled_corals` | `damaged_images`: the dropped images. `case_merged_labels`: raw label to merged label, for labels merged by letter case. `off_annotations_dropped`: the number of annotations dropped because their label is `off`. `images_uploaded`. |
| `pacific_labeled_corals` | `excluded_sites`. `annotations_dropped_unknown_archived`: annotations dropped because their `archived` label ID is not in the label table. `annotations_dropped_unmapped_imageid`: annotations dropped because their image ID is not in the image map. `images_uploaded`. The run logs each drop count as a warning. |
| `benthos_yuval` | `excluded_sites`, `tile_size`, `label_key_template`, `num_tiles` (a count for each site), `images_uploaded` and `labels_uploaded`. |

### Run an ingest

Run from the repository root. `uv run --package segdata-etl` installs the package and its dependencies on first use. S3 access uses the default boto3 credential chain.

```bash
uv run --package segdata-etl segdata ingest <name> --data-dir <raw dir>
```

Flags on every ingest command:

| Flag | Default | Effect |
|---|---|---|
| `--bucket` | `dev-datamermaid-sm-sources` | Destination bucket. |
| `--prefix` | `external_validation_datasets/<name>` | Destination key prefix. |
| `--workers` | 16 | Threads for decoding and uploading images. |
| `--dry-run` | off | Write nothing to S3. |
| `--no-skip-existing-s3` | off | Upload every image, also those already on S3. |
| `--overwrite` | off | Replace the parquet and the JSON files if they exist. Benthos also uploads the `labels/` tiles again. |
| `-v`, `--verbose` | off | Debug logging. |

Flags for each ingest command:

| Dataset | Flags |
|---|---|
| `catlin_seaview` | `--data-dir`: a raw tree already on disk. Without it, the run downloads the zips into `--workdir` (default `outputs/external-datasets/catlin_seaview/`). `--force-download` downloads and extracts the zips again. |
| `moorea_labeled_corals` | `--data-dir` (required): the unzipped EDI package. |
| `pacific_labeled_corals` | `--data-dir` (required): the unzipped Dryad package. |
| `benthos_yuval` | `--data-dir` (required): the six Dryad files. `--workdir` (default `outputs/external-datasets/benthos_yuval/`): where the tiles are written before upload. |

The raw files to download and their layout are in [External datasets](#external-datasets).

## Build a versioned external dataset

`segdata build <name>` stages an immutable dataset version in the `segdata_spec` uniform format from the prefix that `segdata ingest <name>` wrote. `<name>` is `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` or `benthos_yuval`. The build never writes to S3, and it copies no image. [Promote a version](#promote-a-version) publishes it. The code is in `segdata_etl/datasets/`; the shared steps are in `datasets/external.py`. The format and the publish rules are in the [segdata-spec README](../segdata-spec/README.md).

The build reads the ingest prefix from `--legacy-bucket` and the pinned taxonomy from `mermaid-ic-datasets`, so the taxonomy must be published before you stage a dataset. The build refuses a taxonomy pin whose manifest is not in `mermaid-ic-datasets`, even when `--cache-dir` already holds it. It maps the dataset's labels to the taxonomy with the curated map `segdata_etl/label_maps/curated/<name>.csv`. It stages the tables and the plan file `copies.jsonl` under `<stage>/mermaid-ic-datasets/datasets/<name>/<version>/`. Benthos also stages its masks, with `uploads.jsonl`. The build ends by printing two lines:

```text
version: <YYYYMMDD-sha7>
manifest_sha256: <sha256>
```

`manifest_sha256` is the pin: a consumer needs both values to read the version. The manifest records the taxonomy pin in its `taxonomy` field, and `promote` checks it again. A version string is `YYYYMMDD-<sha7>` (the UTC date and short git commit), with `-2`, `-3` for a rebuild on the same day.

```bash
uv run --package segdata-etl segdata build <name> --stage-dir <stage> \
  --taxonomy-version <v> --taxonomy-manifest-sha256 <sha>
```

| Flag | Default | Effect |
|---|---|---|
| `--stage-dir` | none (required) | Local directory the version is staged in. |
| `--taxonomy-version`, `--taxonomy-manifest-sha256` | required | Pin of the `mermaid-benthic` taxonomy that the labels map to. |
| `--cache-dir` | `outputs/segdata-cache` | Where the build syncs the taxonomy from S3. |
| `--workers` | 16 | Threads for inspecting images. |
| `--legacy-bucket` | `dev-datamermaid-sm-sources` | Bucket that holds the ingest prefix. |
| `-v`, `--verbose` | off | Debug logging. |

An image whose header cannot be read is dropped with its points. The count is `stats.dropped["image_unreadable_header"]` in the manifest.

The build refuses a dirty or unpushed git tree. Publish the taxonomy first, then stage each dataset, then [promote](#promote-a-version) it.

## External datasets

The source, citation, licence and rebuild steps of each dataset that training reads. [Build a versioned external dataset](#build-a-versioned-external-dataset) and [Uniform datasets](#uniform-datasets) stage the published version. The flags of `segdata ingest` are in [Run an ingest](#run-an-ingest).

The four ingested datasets (Catlin Seaview, Moorea, Pacific, Benthos) go through two steps. `segdata ingest` writes the checked raw release to `s3://dev-datamermaid-sm-sources/external_validation_datasets/<name>/`. `segdata build <name>` stages a version from that prefix. Run the ingest again only when the prefix is lost or the source publishes a new release. Every ingest accepts `--dry-run`, which writes nothing to S3. If the parquet or a JSON file is already on S3, the run stops; `--overwrite` replaces them.

### Catlin Seaview

Point-annotated photo-quadrats from nine reef regions.

**Source:** Seaview Survey Photo-quadrat and Image Classification Dataset (González-Rivero et al., 2019), at [data.qld.edu.au/public/Q1281](http://data.qld.edu.au/public/Q1281/).
**Licence:** CC BY-NC-SA 4.0. See the UQ eSpace record UQ:734799 (doi:10.14264/uql.2019.930).

| Region code | Place | Images | Annotations | Label classes present | Damaged JPG files |
|---|---|---|---|---|---|
| ATL | Atlantic | 1407 | 92900 | 67 | 0 |
| IND_CHA | Indian Ocean, Chagos Archipelago | 686 | 52450 | 42 | 0 |
| IND_MDV | Indian Ocean, Maldives | 1612 | 144100 | 38 | 0 |
| PAC_AUS | Pacific Ocean, Australia | 2657 | 186420 | 29 | 0 |
| PAC_IDN_PHL | Pacific Ocean, Indonesia and Philippines | 1608 | 118600 | 67 | 30 |
| PAC_SLB | Pacific Ocean, Solomon Islands | 732 | 59200 | 55 | 0 |
| PAC_TLS | Pacific Ocean, Timor-Leste | 864 | 71600 | 59 | 0 |
| PAC_TWN | Pacific Ocean, Taiwan | 638 | 50000 | 54 | 0 |
| PAC_USA | Pacific Ocean, USA (Hawaii) | 1153 | 83100 | 39 | 0 |
| **Total** | | **11357** | **858370** | **194** | **30** |

The release has 11,387 images. The per-region image counts sum to 11,357.

**Ingest prefix:** `external_validation_datasets/catlin_seaview/` holds `catlin_seaview_annotations.parquet` (columns `region`, `image_id`, `row`, `col`, `source_label_name`), `images/<region>/<image_id>.jpg`, `classes.json`, `colors.json` and `manifest.json`. `manifest.json` lists the damaged images that the run dropped, by region.

**Rebuild:**

```bash
uv run --package segdata-etl segdata ingest catlin_seaview
```

Without `--data-dir`, the run downloads `tabular-data.zip` and one zip for each region from `http://data.qld.edu.au/public/Q1281/` into `--workdir` (default `outputs/external-datasets/catlin_seaview/`), and extracts them there. A rerun uses the zips already on disk. To use a raw tree that you downloaded before, pass `--data-dir`:

```text
<data-dir>/tabular-data/annotations_<region>.csv
<data-dir>/<region>/<quadrat_id>.jpg
```

### Moorea Labeled Corals

Point-annotated photo-quadrats from the Moorea Coral Reef LTER, 2008 to 2010.

**Source:** Beijbom O., Edmunds P.J., Kline D.I., Mitchell G.B., Kriegman D., "Automated Annotation of Coral Reef Survey Images", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012. The data is the EDI package [knb-lter-mcr.5006](https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006), a subset of the MCR LTER packaged for computer vision research. It has 2055 images from three habitats, with random point annotations (row, col, label). Nine labels account for 96% of the points: Crustose Coralline Algae (CCA), Turf algae, Macroalgae, Sand, Acropora, Pavona, Montipora, Pocillopora and Porites. Less abundant genera are also present.
**Licence:** CC BY 4.0, as stated by the EDI data package.

| Year | Images | Annotations (9 classes) | Annotations (20 classes) | Label classes present | Damaged or missing images |
|---|---|---|---|---|---|
| 2008 | 671 | 131260 | 133045 | 26 | 0 |
| 2009 | 695 | 132112 | 133072 | 21 | 0 |
| 2010 | 689 | 113256 | 120324 | 26 | 53 |
| **Total** | **2055** | **376628** | **386441** | **29** | **53** |

**Ingest prefix:** `external_validation_datasets/moorea_labeled_corals/` holds `moorea_labeled_corals_annotations.parquet` (columns `year`, `image_id`, `image_ext`, `row`, `col`, `source_label_name`) and `images/<year>/<image_id><image_ext>`. `source_label_name` is the label after case merging: spellings that differ only in case become one label, and the mixed-case spelling is kept. The parquet also has `raw_label_name`, the spelling in the raw file. Points labelled `OFF` (in any case) are not in the parquet. `manifest.json` records the damaged images, the merged spellings and the number of `OFF` points.

**Rebuild:** the EDI package needs a login, so download it by hand from the source link. Unzip it so that the layout is:

```text
<data-dir>/<year>/<image>.<jpg|jpeg|png|tif|tiff>
<data-dir>/<year>/<image>.<ext>.txt      # one "row;col;label" line per point
```

```bash
uv run --package segdata-etl segdata ingest moorea_labeled_corals --data-dir <data-dir>
```

The run drops each damaged image and its annotations.

### Pacific Labeled Corals

Point-annotated photo-quadrats from three Pacific reef sites, with up to seven annotators for each point.

**Source:** Beijbom, O., Edmunds, P. J., Roelfsema, C., Smith, J., Kline, D. I., Neal, B. P., Dunlap, M. J., Moriarty, V., Fan, T.-Y., Tan, C.-J., Chan, S., Treibitz, T., Gamst, A., Mitchell, B. G., Kriegman, D. (2015). Towards Automated Annotation of Benthic Survey Images: Variability of Human Experts and Operational Modes of Automation. PLOS ONE 10(7): e0130312. [doi:10.1371/journal.pone.0130312](https://doi.org/10.1371/journal.pone.0130312). The data is on Dryad at [doi:10.5061/dryad.m5pr3](https://datadryad.org/dataset/doi:10.5061/dryad.m5pr3). It has four sites that share one 20-class label set. The `moorea` site is not loaded, because the same images are in [Moorea Labeled Corals](#moorea-labeled-corals).
**Licence:** CC0 1.0 (Dryad).

| Site | Images | Annotations (points) | Label classes present |
|---|---|---|---|
| `heron_reef` | 2,797 | 64,328 | 5 |
| `line_islands` | 732 | 55,200 | 20 |
| `nanwan_bay` | 890 | 36,260 | 20 |
| **Total** | 4,419 | 155,788 | 20 |

Heron Reef has 5 classes because, in Beijbom et al.: *"Corals were not resolved to genus level in the original Heron Reef label-set and all `Archived` coral annotations were therefore mapped to the generic 'other scleractinians' label for this location."*

Each site has two subsets:

- `reference`: the survey images, with one label for each point, `archived`, from the original ecological survey.
- `evaluation`: 200 images for each site, 10 points for each image. Each point has seven labels: `archived`, `host` (the same local expert, annotating again for the study), and `visitor1` to `visitor5` (five visiting experts).

All images in the Dryad release decode.

**Ingest prefix:** `external_validation_datasets/pacific_labeled_corals/` holds `pacific_labeled_corals_annotations.parquet` and `images/<site>/<subset>/<image_id><image_ext>`. The parquet has the columns `site`, `subset`, `image_id`, `image_ext`, `row`, `col` and `archived`, and the annotator columns `host` and `visitor1` to `visitor5`, which are empty outside the `evaluation` subsets.

**Rebuild:** Dryad has no stable download API, so download the package by hand from the source link. Unzip it so that the layout is:

```text
<data-dir>/<site>/labelmap.txt                    # labelid,name
<data-dir>/<site>/<subset>/imagemap.txt           # imageid,imagefile
<data-dir>/<site>/<subset>/annotations.txt        # imageid,row,col,archived[,host,visitor1..visitor5]
<data-dir>/<site>/<subset>/imgs/<file>.JPG
```

```bash
uv run --package segdata-etl segdata ingest pacific_labeled_corals --data-dir <data-dir>
```

The run stops before it writes to S3 if the three `labelmap.txt` files differ, or if an image does not decode.

### Benthos

Dense label masks over two reef photomosaics, cut into tiles.

**Source:** Yuval, M., Alonso, I., Eyal, G., Tchernov, D., Loya, Y., Murillo, A. C., Treibitz, T. (2021). Repeatable Semantic Reef-Mapping through Photogrammetry and Label-Augmentation. Remote Sensing 13(4): 659. [doi:10.3390/rs13040659](https://doi.org/10.3390/rs13040659). The data is on Dryad at [doi:10.5061/dryad.8cz8w9gm3](https://datadryad.org/dataset/doi:10.5061/dryad.8cz8w9gm3). It has fully labelled photomosaics of three sites, over eight classes (`Algae`, `Calcareous Algae`, `Coral`, `Other`, `Rock`, `Sand`, `Soft Coral`, `Sponge`). Pixels outside the labelled area are `unlabeled`.
**Licence:** CC0 1.0 (Dryad).

| Site | Region | Depth (m) | Mosaic size (px) | Loaded |
|---|---|---|---|---|
| `RS24` | IUI Eilat reef, Gulf of Aqaba, Red Sea | 24–28 | 11070x11786 | Yes |
| `CR_DoubleWreck` | Double-Wreck reef, St. Eustatius, Caribbean | 20 | 24192x3982 | Yes |
| `MD_spartan` | Spartan Reef, offshore Haifa, Mediterranean | 20 | — | No |

`MD_spartan` is excluded because it is a rocky Mediterranean site with no scleractinian coral community.

**Ingest prefix:** `external_validation_datasets/benthos_yuval/` holds `benthos_yuval_annotations.parquet` (one row for each class present in a tile; columns include `site`, `image_id`, `source_label_name`, `tile_height` and `tile_width`), `images/<site>/<image_id>.png` (RGB tiles), `labels/<site>/<image_id>.png` (single-channel masks, with pixel values from `classes.json`) and `classes.json`. Each mosaic is cut into 2048x2048 tiles with no overlap. `image_id` is `<y0>_<x0>`, the pixel position of the tile's top-left corner in the mosaic. The tiles on the bottom and right edges are smaller than 2048 and are not padded.

**Rebuild:** the Dryad download needs a login, so download these six files by hand from the source link into one directory:

| File | Content |
|---|---|
| `RS24.png` | RGB mosaic of `RS24`. |
| `ManualRS24.png` | Label mask of `RS24`. |
| `CR_DoubleWreck.png` | RGB mosaic of `CR_DoubleWreck`. |
| `ManualDW.png` | Label mask of `CR_DoubleWreck`. |
| `dictionary_labels.txt` | `{class_name: mask_value}` for `ManualRS24.png`. |
| `dictionary_labelsDW.txt` | `{class_name: mask_value}` for `ManualDW.png`. |

```bash
uv run --package segdata-etl segdata ingest benthos_yuval --data-dir <data-dir>
```

The run writes the tiles to `--workdir` (default `outputs/external-datasets/benthos_yuval/`) before it uploads them.

### Coralscapes V2

Dense label masks over 95 classes for video frames of Red Sea reefs.

**Source:** CoralscapesV2: Panoptic and Fine-Grained Visual Scene Understanding in Coral Reefs (ECCV 2026 Workshop on Marine Vision), [arXiv:2609.12826](https://arxiv.org/abs/2609.12826). It extends [Coralscapes](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) from 39 to 95 classes. Each of the 95 classes maps onto one of the 39 Coralscapes classes. The build reads the `instances` configuration of the Hugging Face dataset [`josauder/coralscapesV2`](https://huggingface.co/datasets/josauder/coralscapesV2) at a pinned revision. The dataset card gives 2433 frames at 1024x2048 px from 41 dive sites.
**Licence:** Apache 2.0, as stated on the Hugging Face dataset card.

| Split | Images | Dive sites |
|---|---|---|
| `train` | 1790 | 34 |
| `validation` | 170 | 3 |
| `test` | 473 | 5 |

**Rebuild:** there is no ingest step. `segdata build coralscapes_v2` reads the Hugging Face dataset. See [Uniform datasets](#uniform-datasets).

### UCSD Mosaics

Dense label masks over 34 classes, in 512x512 patches cut from reef mosaics of 16 dive sites in Palau and the Northern Line Islands.

**Source:** the data has three layers of processing. Cite all three works:

1. Edwards et al. 2017 released the large-area mosaics as a supplementary PDF ([MOESM5](https://static-content.springer.com/esm/art%3A10.1007%2Fs00338-017-1624-3/MediaObjects/338_2017_1624_MOESM5_ESM.pdf)). The masks in that PDF are JPEG-compressed and some legend colors are shared, so the labels cannot be recovered from it.
2. Alonso et al. 2019 (CoralSeg) made pixel-accurate masks for the same mosaics and cut them into 512x512 patches. They published the train and test patches. The validation patches are not available, so the patches do not cover the full mosaics.
3. Raine et al. 2024 removed corrupted pixels from the top rows of the Alonso masks. This variant, GT-Clean, is the one in the mirror.

```bibtex
@article{edwards2017largearea,
  title   = {Large-area imaging reveals biologically driven non-random spatial patterns of corals at a remote reef},
  author  = {Edwards, Clinton B. and others},
  journal = {Coral Reefs},
  volume  = {36},
  year    = {2017}
}

@article{alonso2019coralseg,
  title   = {CoralSeg: Learning coral segmentation from sparse annotations},
  author  = {Alonso, I{\~n}igo and Yuval, Matan and Eyal, Gal and Treibitz, Tali and Murillo, Ana C.},
  journal = {Journal of Field Robotics},
  volume  = {36},
  year    = {2019}
}

@inproceedings{raine2024human,
  title     = {Human-in-the-loop segmentation of multi-species coral imagery},
  author    = {Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Sunderhauf, Niko and Fischer, Tobias},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {2723--2732},
  year      = {2024}
}
```

**Licence:** CC BY-NC 4.0, under UC San Diego Library's copyright terms. Use of the data, directly or through the mirror, carries the same terms and the obligation to cite the three works above.

| Site | Hugging Face split | Patches | Train | Test |
|---|---|---|---|---|
| FR3 | `fr3` | 220 | 184 | 36 |
| FR5 | `fr5` | 262 | 225 | 37 |
| FR7 | `fr7` | 415 | 353 | 62 |
| FR9 | `fr9` | 310 | 258 | 52 |
| PAL132 | `pal132` | 323 | 274 | 49 |
| PAL239 | `pal239` | 292 | 244 | 48 |
| PAL36 | `pal36` | 259 | 221 | 38 |
| PAL69 | `pal69` | 304 | 269 | 35 |
| PALStrawn | `palstrawn` | 477 | 413 | 64 |
| PALWave13 | `palwave13` | 314 | 272 | 42 |
| PALWave14 | `palwave14` | 318 | 275 | 43 |
| PALWave37 | `palwave37` | 297 | 251 | 46 |
| PALWave38 | `palwave38` | 242 | 198 | 44 |
| PALWave39 | `palwave39` | 254 | 217 | 37 |
| PALWave4 | `palwave4` | 145 | 126 | 19 |
| PALWave40 | `palwave40` | 238 | 194 | 44 |
| **Total** | | **4670** | **3974** | **696** |

The build reads the Hugging Face dataset [`josauder/UCSD-mosaics-mirror`](https://huggingface.co/datasets/josauder/UCSD-mosaics-mirror) at a pinned revision. It has one split for each dive site, named by the lowercased site ID. The columns are `image` (RGB, 512x512), `label` (single channel, 512x512, values 0 to 34) and `original_split` (`train` or `test` in GT-Clean). `classes.json` at the top of the repo lists the 34 classes with `id`, `raw_legend_id`, `name`, `description` and `color_rgb`. Label value 0 is the ignore label. It includes the `Unidentified` row of the source legend (raw mosaic ID 34).

**Rebuild:** there is no ingest step. `segdata build ucsd_mosaics` reads the Hugging Face dataset. See [Uniform datasets](#uniform-datasets). The mirror was uploaded by hand. The data and the processing are the GT-Clean release, unchanged. No script in this repository builds or uploads it.

### CoralNet sources

Point-annotated images from the public sources on [CoralNet](https://coralnet.ucsd.edu/), an online repository of coral reef images with expert point annotations. A source groups the images of one project, survey, site or contributor. The labels are CoralNet Label IDs, mapped to MERMAID benthic attributes by `segdata curate coralnet`.

**Rebuild:** [`coralnet-scraper`](../coralnet-scraper/README.md) downloads the sources to S3. The [CoralNet](#coralnet) and [CoralNet dataset versions](#coralnet-dataset-versions) sections of this README build and stage them. The order of the steps is in the [CoralNet data runbook](../../docs/runbooks/coralnet-data.md).

### MERMAID

Point-annotated images confirmed by users of the [MERMAID platform](https://datamermaid.org/). Users upload images and confirm annotations, so the dataset grows over time. Each image has 25 points on a 5x5 grid. The labels are MERMAID benthic attributes, the target label space of the project.

**Source:** the bucket `coral-reef-training` holds `mermaid/mermaid_confirmed_annotations.parquet` (columns include `image_id`, `row`, `col`, `benthic_attribute_name`, `updated_on`, `region_id` and `region_name`) and `mermaid/<image_id>.png`. All image keys end in `.png`, but some objects hold JPEG data. A check on 2026-07-24 found 14,258 JPEG and 2,423 PNG objects among the 16,681 that the parquet references. The parquet changes in place, so `--annotation-date-cutoff` freezes a version: it keeps only annotations with `updated_on` on or before that date, in UTC.

**Rebuild:** this repository does not write the parquet or the images. The MERMAID platform exports them. To stage a version of newer annotations, see [Update the MERMAID data](../../docs/runbooks/training-datasets.md#update-the-mermaid-data).

## Taxonomy snapshot

`segdata taxonomy` stages the MERMAID benthic hierarchy, from `/v1/benthicattributes/` and the growth forms in `/v1/choices/`, as `taxonomy/mermaid-benthic/<version>/` under `<stage>/mermaid-ic-datasets/`. It writes no S3 object. The version string follows the same `YYYYMMDD-<sha7>[-N]` form, and is named against the stage and the bucket, so the staged name is the name `promote` publishes. The command prints the version and its manifest sha256, which are the taxonomy pin for `segdata build` and `segdata curate`.

```bash
uv run --package segdata-etl segdata taxonomy --stage-dir <stage>
uv run --package segdata-etl segdata promote --stage-dir <stage> --taxonomy mermaid-benthic --version <version>
```

| Flag | Effect |
|---|---|
| `--stage-dir <dir>` | Required. Fetch the API and stage a version in this local directory. |
| `-v`, `--verbose` | Debug logging. |

`promote --taxonomy` takes the place of `--dataset`; the two flags are mutually exclusive. It checks the staged tables against the staged manifest, refuses a version already in the bucket or a destination key that holds other content, then writes the tables and `manifest.json` last. Publishing is irreversible: get approval first, as for [Promote a version](#promote-a-version).

A taxonomy pin that is only staged or cached is refused by `segdata build`, `segdata curate`, `segdata split build` and `promote` of a dataset that pins it.

## Tests

```bash
uv run --package segdata-etl --group tests pytest packages/segdata-etl/tests
```

The CoralNet suite is in `packages/segdata-etl/tests/coralnet/`, the raw-release ingest suite in `packages/segdata-etl/tests/ingest/`, the external dataset builder suite in `packages/segdata-etl/tests/datasets/`, the split suite in `packages/segdata-etl/tests/splits/`, the uniform dataset suite in `packages/segdata-etl/tests/uniform/`, the release, label and ingest-prefix helper suite in `packages/segdata-etl/tests/common/` and the label map suite in `packages/segdata-etl/tests/label_maps/`. `test_taxonomy.py` and `test_promote_taxonomy.py` sit directly in `packages/segdata-etl/tests/`. Pass one directory to run one suite.

The CoralNet tests are offline and use no network. The ingest and external builder tests use an in-memory S3 fake (`FakeS3`, defined in `tests/s3_fake.py`) and synthetic raw trees. `pytest-socket` blocks every host except localhost.
