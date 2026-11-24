# segdata-etl

Builds the segmentation training data in S3: the CoralNet training manifest, the four external datasets, and the versioned datasets and taxonomy that `segdata_spec` defines. Import package: `segdata_etl`; console script: `segdata`.

| Command | Does |
|---|---|
| `segdata build coralnet <stage>` | Builds the CoralNet training dataset (see [CoralNet](#coralnet)). |
| `segdata build coralnet raw` | Publishes a CoralNet build run as the `coralnet-raw` dataset version (see [Publish coralnet-raw](#publish-coralnet-raw)). |
| `segdata curate coralnet` | Publishes the curated `coralnet` v1 and v2 versions from `coralnet-raw` (see [Curate coralnet v1 and v2](#curate-coralnet-v1-and-v2)). |
| `segdata ingest <name>` | Loads a raw external release into the legacy `external_validation_datasets/<name>/` prefix (see [Ingest an external dataset](#ingest-an-external-dataset)). |
| `segdata build <name>` | Builds an immutable versioned dataset from that legacy prefix (see [Build a versioned external dataset](#build-a-versioned-external-dataset)). |
| `segdata taxonomy` | Snapshots the MERMAID benthic taxonomy as a version (see [Taxonomy snapshot](#taxonomy-snapshot)). |
| `segdata split build`, `segdata split leakage` | Builds and checks the CoralNet split (see [CoralNet split](#coralnet-split)). |

## Scope

- **Owns:** turning a CoralNet scrape into the training manifest, the resized images and the id2name JSON; checking each raw external release and writing it to the legacy S3 prefix; building the external datasets and the MERMAID benthic taxonomy as published versions. The CoralNet code is in `segdata_etl/datasets/coralnet/` (the CLI stages are in `build.py`). The raw-release ingest modules are in `segdata_etl/ingest/`. The versioned external builders are in `segdata_etl/datasets/`, with their shared steps in `datasets/external.py` and the curated label maps in `label_maps/curated/<name>.csv`. The taxonomy snapshot is `segdata_etl/taxonomy.py`. The helpers they share are in `segdata_etl/common/`.
- **Does not own:** downloading CoralNet. That is [`coralnet-scraper`](../coralnet-scraper/README.md). Nor the MERMAID parquet or the Hugging Face datasets.
- **Boundary:** The root `tests/test_package_boundaries.py` enforces these rules (run with the root suite): `segdata_etl` does not import `mermaidseg`; `mermaidseg` and `scripts` import neither `segdata_etl` nor `coralnet_scraper`; `coralnet_scraper` imports neither `mermaidseg` nor `segdata_etl`. Only `segdata_etl/datasets/coralnet/` may import `coralnet_scraper`, for its configuration and S3 key layout. The boundary test enforces this, so `common/`, `ingest/` and the external builders stay free of it. The `segdata` CLI loads the CoralNet build code, and with it `coralnet_scraper` and `ibis`, only when a `build coralnet` stage runs. The argparse wiring is in `datasets/coralnet/commands.py`, which imports neither. A root `uv sync` does not install the package.

```bash
uv run --package segdata-etl segdata <command> ...
```

`<name>` is `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` or `benthos_yuval`. `build coralnet` takes a stage after it (see [Run](#run)).

## CoralNet

Builds the CoralNet training dataset from a scrape on S3.

- **End to end:** to recreate or update the CoralNet data, scrape through training, follow the [CoralNet data runbook](../../docs/runbooks/coralnet-data.md).
- **Consumer:** `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`) reads the manifest and the id2name JSON.

### Inputs

The build reads the per-source folders that the scraper writes under `s3://<bucket>/<prefix>/s<source_id>/`: `annotations.csv`, `image_list.csv`, `labelset.csv` and `images/<image_id>.jpg`. See the [scraper's S3 layout](../coralnet-scraper/README.md#s3-layout).

### Outputs

Each run has a version tag. The default tag is the UTC date and the short git commit, `YYYYMMDD_<sha>`, or `YYYYMMDD_nogit` outside a git checkout. `--version-tag` sets it.

Files are written locally to `outputs/coralnet-dataset/<version>/` (change the base with `--output-dir`):

| File | Written by | Content |
|---|---|---|
| `coralnet_annotations_<version>.parquet` | `build-annotations` | Confirmed point annotations of the included sources: `source_id`, `image_id`, `row`, `col`, `coralnet_id`, `status`. |
| `coralnet_id2name_<version>.json` | `build-annotations` | Label ID (as a string) to label name, merged from the included sources' `labelset.csv`. |
| `coralnet_images_<version>.parquet` | `build-images` | One row per annotated image, from its JPEG header: `source_id`, `image_id`, `s3_key`, `width`, `height`, `longest_edge`, `file_size`, `needs_resize`, `header_status`, `error_message`. |
| `coralnet_training_manifest_<version>.parquet` | `build-manifest` | One row per annotation, ready for training. |

With `--upload-to-s3`, each file is also uploaded to `s3://<bucket>/etl-outputs/coralnet/<version>/<file>`. An upload whose key already exists raises `RuntimeError` and stops the command, whether or not `--version-tag` is given. Pass `--overwrite` to replace existing keys.

`resize` writes to S3 only, with or without `--upload-to-s3`. An image needs a resize when its longest edge is more than 2048 pixels. The resized copy has a longest edge of 2048, keeps the aspect ratio, and is a JPEG at quality 95:

```text
s3://<bucket>/<resized prefix>/resized/s<source_id>/images/<image_id>.jpg
```

The objects on S3 are the resize state. A rerun lists `<resized prefix>/resized/` and resizes only the images that need a resize and have no object there.

The training manifest has these columns. `CoralNetDataset` reads `source_id`, `image_id`, `row`, `col`, `coralnet_id` and `image_s3_key`:

| Column | Content |
|---|---|
| `source_id`, `image_id` | CoralNet ids. |
| `row`, `col` | Point position, scaled to the image that is loaded. |
| `coralnet_id` | CoralNet Label ID. |
| `source_label_name` | `coralnet_id` as a string. `CoralNetDataset` replaces it with the lowercased name from the id2name JSON. |
| `image_s3_key` | The resized copy when the image was resized, else the original. |
| `load_width`, `load_height` | Dimensions of the image at `image_s3_key`. |
| `uses_resized_image` | Whether `image_s3_key` is the resized copy. |

The manifest drops an image, with all its annotations, when the image has no width or height, or when it needs a resize and has no resized object on S3.

### Run

Run from the repository root. `uv run --package segdata-etl` installs the package and its dependencies on first use:

```bash
uv run --package segdata-etl segdata build coralnet <stage>
```

Run the stages in this order, or run `all` to do the four in one invocation with one version tag:

| Stage | Reads | Flags in addition to the common ones |
|---|---|---|
| `build-annotations` | The source CSVs on S3 | `--source-ids 1,2,3`: the sources to read. Default: every `s<id>/` folder under the prefix. |
| `build-images` | The annotations parquet, and the image headers on S3 | `--annotations <path>` (required) |
| `resize` | The images parquet, and the images on S3 | `--images <path>` (required) |
| `build-manifest` | Both parquets, and the resized objects on S3 | `--annotations <path>`, `--images <path>` (both required) |
| `all` | As above | `--source-ids` |

Common flags, on every stage:

| Flag | Default | Effect |
|---|---|---|
| `--version-tag` | `YYYYMMDD_<sha>` | Run version tag. It names the run only. |
| `--overwrite` | off | Let an upload replace an existing key under `etl-outputs/coralnet/<version>/`. Without it, an upload to an existing key raises `RuntimeError`. A training run that pins this tag then reads different data. |
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

`build-images` logs the image count per `header_status`. `ok`, `sof_not_found_extended` and `pil_fallback` carry dimensions. `not_found`, `corrupt`, `s3_error` and `worker_exception` do not, and the manifest drops those images.

`resize` logs the number of images resized, oversized, corrupted and failed. An image is oversized when it exceeds twice Pillow's `MAX_IMAGE_PIXELS`, the decompression-bomb error threshold. It is corrupted when it fails inspection for any other reason. It is failed when the download, resize or upload raises. Oversized and corrupted images are skipped.

Exit codes:

| Stage | Exit code 1 when |
|---|---|
| `build-annotations` | Any source is excluded, or no annotation rows are built. |
| `build-images` | Any image has no dimensions. |
| `resize` | Any image is oversized, corrupted or failed. |
| `build-manifest` | Only on an uncaught error. |
| `all` | Any step returns 1. Every step runs. |

Every stage also ends with a non-zero code on an uncaught error, such as invalid configuration or an upload refused as above. The output files are written before the exit code is set, so a run with exit code 1 still leaves its outputs.

#### Environment variables

Nothing loads `.env` for you: neither the package nor `uv run`. Copy the variables from `.env.example` into `.env`, then export them in your shell before the command:

```bash
set -a; source .env; set +a
```

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `MERMAID_CORALNET_BUCKET` | yes | none | Bucket that holds the scrape, the resized images and the uploads. Shared with the scraper. |
| `MERMAID_CORALNET_PREFIX` | no | `coralnet-public-images` | Key prefix of the scrape. Shared with the scraper. |
| `MERMAID_CORALNET_RESIZED_PREFIX` | no | `dev/images` | Key prefix that holds `resized/`. Surrounding slashes are stripped; an empty prefix raises `ValueError`. |
| `MERMAID_CORALNET_ETL_WORKERS` | no | `16` | Concurrent S3 workers. Must be an integer of at least 1. `--workers` overrides it. |

S3 access uses the default boto3 credential chain, for example `AWS_PROFILE`.

### Use a build in training

`CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`) reads the manifest from its `source_bucket`. The data config selects the build. Every CoralNet split in `configs/data_config.yaml` must set `coralnet.<split>.manifest_path` to the manifest key:

```yaml
coralnet:
  train:
    manifest_path: etl-outputs/coralnet/<run>/coralnet_training_manifest_<run>.parquet
```

There is no default build. If a split has no `manifest_path`, building the dataset fails.

The dataset reads the label names from `coralnet_id2name_<run>.json` in the same directory as the manifest, where `<run>` is the directory name. A custom path must keep that layout.

`image_s3_key` and the id2name JSON are keys in `MERMAID_CORALNET_BUCKET`, so a build is readable only when that bucket is the dataset's `source_bucket` (the `CoralNetDataset` default).

### Known gaps

- Duplicate images (the same photo in several sources) are removed in training by `CoralNetDataset`'s `image_blacklist` and `annotation_overrides`. This package does not produce those files. Tracked in issue #231.

## CoralNet dataset versions

Two commands publish CoralNet as dataset versions in the [segdata-spec](../segdata-spec/README.md) format. For the order to run them in, see [Publish CoralNet dataset versions](../../docs/runbooks/coralnet-data.md#publish-coralnet-dataset-versions).

| Command | Publishes | Built from |
|---|---|---|
| `segdata build coralnet raw` | `coralnet-raw` | A build run under `etl-outputs/coralnet/<run>/`, and the original images of the scrape. |
| `segdata curate coralnet --label-map cleaned_up_rerun` | `coralnet` v1 | A `coralnet-raw` version and a `mermaid-benthic` taxonomy version. |
| `segdata curate coralnet --label-map mermaid_api` | `coralnet` v2 | A v1 `coralnet` version, its `coralnet-raw` parent, and the MERMAID LabelMapping API. |

### Store, version name and publish guard

These rules apply to both commands and to `segdata taxonomy`.

- **Store.** `--store` is a local directory (default `outputs/segdata-store`) or `s3`. A directory store holds each object at `<dir>/mermaid-ic-datasets/<key>`, which is the segdata cache layout. `--store s3` writes to the bucket `mermaid-ic-datasets` with the default boto3 credentials. The pinned parent versions are read from the same store, so a version on S3 needs its parents on S3.
- **Version name.** `YYYYMMDD-<sha7>`: the local date and the first 7 characters of the HEAD commit. If the store already holds that version of the dataset, the command takes the lowest free suffix: `-2`, then `-3`, and so on.
- **Publish guard.** `publish_dataset` (`segdata_spec/publish.py`) raises `PublishRefused` when tracked files have uncommitted changes, when no remote branch contains HEAD, or when the repository root has no `uv.lock`. The repository root is the nearest directory at or above the package that holds `.git` and `uv.lock`, so run the commands from a git checkout. The guard runs at publish time, after the build work: `raw` copies its images first, then refuses.
- **Write-once.** Tables, the manifest and copied images are written only if absent. A key that holds other bytes raises `ObjectExists`.
- **Pin.** On success the command prints `version: <version>` and `manifest_sha256: <sha256>`. These two values are the pin that the next command takes.

### Publish coralnet-raw

`segdata build coralnet raw` publishes one build run as a `coralnet-raw` version, and copies the originals of its kept images into the store.

```bash
uv run --package segdata-etl segdata build coralnet raw --store s3
```

The command reads `coralnet_annotations_<run>.parquet`, `coralnet_images_<run>.parquet` and `coralnet_id2name_<run>.json` from `etl-outputs/coralnet/<run>/` in `MERMAID_CORALNET_BUCKET` (default `dev-datamermaid-sm-sources`). It reads each source's `image_list.csv` and lists its `images/` folder under `MERMAID_CORALNET_PREFIX`. See [Environment variables](#environment-variables).

| Flag | Default | Effect |
|---|---|---|
| `--store` | `outputs/segdata-store` | Destination: a directory or `s3`. |
| `--build-run` | `20260623_nogit` | Build run under `etl-outputs/coralnet/` to publish. |
| `--source-ids 1,2,3` | all | Keep only these sources of the build run. A directory store needs it. `--store s3` refuses it, so an S3 version holds every source of the build run. |
| `--parent-version`, `--parent-manifest-sha256` | none | A `coralnet-raw` version to reuse. Give both or neither. |
| `--workers` | `MERMAID_CORALNET_ETL_WORKERS`, else 16 | Concurrent S3 workers. |
| `-v`, `--verbose` | off | Debug logging. |

A wrong flag combination is a usage error (exit 2) before any S3 read.

The version holds:

| Table | Content |
|---|---|
| `images.parquet` | One row per kept image. `key` is `s<source_id>/<image_id>.jpg` or `.png` under the image store prefix `images/coralnet/`. `width` and `height` are the original's. `group` is `s<source_id>`. `source_fingerprint` is the original's S3 ETag. `source_name` is the image's name in `image_list.csv` without the status suffix, or null. |
| `points.parquet` | The build's annotations of the kept images, in the pixel space of the original. `label_id` is the CoralNet Label ID; `annotator` is `default`. |
| `labels.parquet` | One row for every ID in the build's id2name JSON, used or not. Native names only: the MERMAID columns are null and `mapping_source` is `identity`. |
| `stats/label_counts.parquet` | Points per image and label. |

An image is kept when it has dimensions in the images parquet, has at least one annotation, has its original on S3, and is a JPEG or PNG. The media type is JPEG when `header_status` is `ok`; otherwise the command reads the first 8 bytes of the original.

Each kept original is copied server side into the store, write-once, with a CRC64NVME checksum and the `INTELLIGENT_TIERING` storage class. A directory store downloads the originals into the directory. With a parent pin, an image whose `image_id` and source ETag match a parent row keeps the parent's key and object fields, and is not copied. The new version lists the parent in `parents`.

`stats.dropped` in the manifest:

| Key | Unit | Counts |
|---|---|---|
| `no_dimensions` | images | Images whose `header_status` carries no dimensions. |
| `no_annotations` | images | Images with dimensions and no annotation. |
| `annotations_without_image` | annotation rows | Annotations whose image is not in the images parquet. |
| `source_missing` | images | Kept images whose original is not in the source's `images/` folder. |
| `unsupported_format` | images | Originals that are neither JPEG nor PNG. |
| `source_name_missing` | images | Published images with a null `source_name`. These images stay in the version. |

#### Resume a raw publish

A rerun copies only what the store does not hold:

1. The command lists every object under `images/coralnet/` in the destination.
2. A listed key counts as copied when its ETag is the original's ETag, or when the checkpoint has a row with the same key, original ETag and listed ETag. Its CRC64NVME comes from that checkpoint row, else from a HEAD.
3. A listed key with any other ETag stops the run with `ObjectExists` before any copy, unless the original is multipart (its ETag contains `-`). A copy writes a multipart original as one part with a new ETag, so the listing cannot match it. Such a key goes to the copy step instead.
4. The command copies the rest in batches of 1000 and rewrites the checkpoint after each batch. When a conditional copy fails with 412 because the key exists, the command compares the size and CRC64NVME of both objects (MD5 when either has no CRC). It adopts the object when they match and raises `ObjectExists` when they differ.

The checkpoint is a parquet of each copied key with its original ETag, ETag, size and CRC64NVME:

- `--store s3`: `outputs/coralnet-raw-checkpoints/s3-<build-run>.parquet`, relative to the working directory.
- Directory store: `<store>/checkpoints/coralnet-raw-<build-run>.parquet`.

Without the checkpoint a rerun is still correct, but it sends a HEAD for each held key and sends each multipart original through the copy and compare again.

### Curate coralnet v1 and v2

`segdata curate coralnet` publishes a `coralnet` version from pinned parents. `--label-map` selects the version kind.

```bash
uv run --package segdata-etl segdata curate coralnet --label-map cleaned_up_rerun --store s3 \
  --raw-version <v> --raw-manifest-sha256 <sha> \
  --taxonomy-version <v> --taxonomy-manifest-sha256 <sha>

uv run --package segdata-etl segdata curate coralnet --label-map mermaid_api --store s3 \
  --raw-version <v> --raw-manifest-sha256 <sha> \
  --parent-version <v1> --parent-manifest-sha256 <sha>
```

| Flag | Default | Effect |
|---|---|---|
| `--label-map` | required | `cleaned_up_rerun` (v1) or `mermaid_api` (v2). |
| `--store` | `outputs/segdata-store` | Destination, and where the parents are read from. |
| `--raw-version`, `--raw-manifest-sha256` | required | Pin of the `coralnet-raw` parent. |
| `--taxonomy-version`, `--taxonomy-manifest-sha256` | none | Pin of the `mermaid-benthic` taxonomy. `cleaned_up_rerun` needs both; `mermaid_api` takes neither. |
| `--parent-version`, `--parent-manifest-sha256` | none | Pin of the v1 `coralnet` version. `mermaid_api` needs both; `cleaned_up_rerun` takes neither. |
| `--cache-dir` | `outputs/segdata-cache` | Where the parent tables are synced. No image is synced. |
| `--out-dir` | `outputs/coralnet-curate` | `mermaid_api` writes the API snapshot to `<out-dir>/<version>/coralnet_label_mappings.json`. |
| `-v`, `--verbose` | off | Debug logging. |

A missing pin flag, a pin flag of the other label map, or one flag of a pin without the other is a usage error (exit 2). The message names the flags.

**v1 (`cleaned_up_rerun`)** has one parent, the raw version, and uses its image store. It copies no image.

1. Overrides. For each image in `configs/coralnet/annotation_overrides.parquet`, the override points replace all of that image's raw points. Override points are in the pixel space of the image resized to a longest edge of 2048. For an original whose longest edge is more than 2048, each point is mapped back to original coordinates: a raw point with the same label that scales to the same position gives its exact coordinates; otherwise the command takes the smallest original coordinate that scales to the override position. Overrides of smaller images are used as they are.
2. Blacklist. The images in `configs/coralnet/image_blacklist.parquet` (by `image_id`) and their points are dropped.
3. Labels. One row per label of the raw version. The lowercased CoralNet name is looked up in `segdata_etl/label_maps/coralnet_cleaned_up_rerun.json`, a copy of `configs/coralnet_to_mermaid_mapping_temporary.json` at commit `e0d3876`. The target name is resolved to a benthic attribute ID in the pinned taxonomy by name, ignoring case. A target with no match keeps its name with a null ID, and `label_mapping.n_unresolved` counts it. A target that matches more than one attribute stops the run with `ValueError`. `mermaid_growth_form` is null.

**v2 (`mermaid_api`)** has two parents, `[v1, raw]`.

1. The raw pin must be a `coralnet-raw` parent that the v1 manifest declares. Otherwise the run stops with `ValueError`.
2. The command fetches every CoralNet row from `https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet&limit=1000`. It retries a GET 3 times on connection errors and 502, 503 and 504, then fails. It never uses an older copy.
3. `images.parquet`, `points.parquet` and `stats/label_counts.parquet` are v1's. Before it publishes, the command checks that each one serialises to the sha256 and size in v1's manifest, and raises `RuntimeError` if one differs. `stats.dropped` repeats v1's.
4. Labels. One row per label of the raw version, joined to the API rows on the integer CoralNet Label ID. A label is unmapped when its name is not a string, the API has no row for it, or the row's `benthic_attribute_name` is null. Mapped rows take the attribute ID, attribute name and growth form name from the API.
5. The command writes the snapshot under `--out-dir`, then publishes. The snapshot holds `fetched_at`, `sha256` and the rows sorted by ID. The manifest's `label_mapping` records the same sha256 and fetch time. If the snapshot cannot be written, nothing is published.

`stats.dropped` of a v1 version:

| Key | Unit | Counts |
|---|---|---|
| `blacklisted_images` | images | Raw images dropped by the blacklist. |
| `override_images_not_in_raw` | images | Distinct override images that the raw version does not hold. Their overrides are dropped. |

## CoralNet split

`segdata split build` assigns the curated CoralNet dataset to train, val and test by source group, and lists excluded sources as `excluded`. `segdata split leakage` checks a split file for images shared across splits. Both run only on demand: when you re-split on purpose, or check a version. Training never runs them. The committed result is `configs/splits/coralnet.csv` with its provenance in `configs/splits/coralnet.meta.yaml`. That meta file predates `build`. It records the original split's provenance (whitelists, builder commit, manual overrides), not the `pins` and `inputs` layout that `build` writes. The file format and coverage check are in [segdata-spec](../segdata-spec/README.md#splits).

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

- The `--raw-version` and `--raw-manifest-sha256` pin is not a `coralnet-raw` parent that the `coralnet` manifest declares. The error lists the declared parents.
- A raw image of a source in the split universe has no `crc64nvme`, so it cannot be matched as a duplicate. The universe is the curated sources minus `--exclude-sources`. The error gives the count and the sorted source ids.
- A `--test-sources` id is not in the universe (for example, it is also in `--exclude-sources`). The error lists the sorted ids.

### Check a split for leakage

`leakage` reads a split file and the `images.parquet` of the pinned `coralnet` version. Two images are the same when their `crc64nvme` and `size` match. Groups the split file does not list, and rows with `split` `excluded`, are ignored. The check reports:

- images of val that also appear in train, and images of test that also appear in train or val, per group;
- pairs of groups in different splits that share an image;
- `split_group` values that span more than one split. Blank values are skipped.

The command prints the report. With `--out-dir` it also writes `leakage_per_source.csv`, `leakage_pairs.csv` and `leakage_split_groups.csv`. It exits 1 if any of the three findings is non-empty, and 0 otherwise.

```bash
uv run --package segdata-etl segdata split leakage \
  --split-file configs/splits/coralnet.csv \
  --version <v> --manifest-sha256 <sha> \
  --cache-dir <cache> --out-dir <report-dir>
```

`leakage` stops with a `ValueError` before it prints the report when any image of a group that the split file lists has no `crc64nvme`. The error gives the count and the sorted group names.

## Ingest an external dataset

`segdata ingest <name>` loads the raw release of an external dataset into S3, in the layout that its dataset class in `mermaidseg/datasets/` reads. `<name>` is `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` or `benthos_yuval`. The code is in `segdata_etl/ingest/`. It writes the legacy `external_validation_datasets/<name>/` prefix, which training reads. [Build a versioned external dataset](#build-a-versioned-external-dataset) reads that prefix.

### Published files

Every ingest command writes the same set of files under `s3://<bucket>/<prefix>/`:

| Key | Content |
|---|---|
| `<name>_annotations.parquet` | The annotations that the dataset class reads. The columns differ for each dataset; `manifest.json` lists them. |
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

The defaults are `--bucket dev-datamermaid-sm-sources` and `--prefix external_validation_datasets/<name>`. These match the defaults of the dataset classes.

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

The README of each dataset lists the raw files to download and their layout:

- [Catlin Seaview](../../mermaidseg/datasets/catlin_seaview/README.md)
- [Moorea Labeled Corals](../../mermaidseg/datasets/moorea_labeled_corals/README.md)
- [Pacific Labeled Corals](../../mermaidseg/datasets/pacific_labeled_corals/README.md)
- [Benthos](../../mermaidseg/datasets/benthos_yuval/README.md)

## Build a versioned external dataset

`segdata build <name>` builds an immutable dataset version in the `segdata_spec` uniform format from the legacy prefix that `segdata ingest <name>` wrote, and publishes it with `segdata_spec.publish.publish_dataset`. `<name>` is `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` or `benthos_yuval`. The code is in `segdata_etl/datasets/`; the shared steps are in `datasets/external.py`. The format and the publish rules are in the [segdata-spec README](../segdata-spec/README.md).

The build reads the legacy prefix from `--legacy-bucket` and the pinned taxonomy from the store. It maps the dataset's labels to the taxonomy with the curated map `segdata_etl/label_maps/curated/<name>.csv`, copies each image to `images/<name>/<group>/<image_id>.<ext>` in the store, and publishes the tables. The build ends by printing two lines:

```text
version: <YYYYMMDD-sha7>
manifest_sha256: <sha256>
```

`manifest_sha256` is the pin: a consumer needs both values to read the version. A version string is `YYYYMMDD-<sha7>` (the UTC date and short git commit), with `-2`, `-3` for a rebuild on the same day.

```bash
uv run --package segdata-etl segdata build <name> \
  --taxonomy-version <v> --taxonomy-manifest-sha256 <sha>
```

| Flag | Default | Effect |
|---|---|---|
| `--store` | `outputs/segdata-store` | Where to publish: a local directory, or `s3` for the `mermaid-ic-datasets` bucket. A local store is laid out as `<dir>/<bucket>/<key>`. Publishing to S3 happens only when you pass `s3`. |
| `--taxonomy-version`, `--taxonomy-manifest-sha256` | required | Pin of the `mermaid-benthic` taxonomy that the labels map to. |
| `--cache-dir` | The store directory; `outputs/segdata-cache` with `--store s3` | Where the build syncs the taxonomy. |
| `--workers` | 16 | Threads for copying and inspecting images. |
| `--legacy-bucket` | `dev-datamermaid-sm-sources` | Bucket that holds the legacy prefix. |
| `-v`, `--verbose` | off | Debug logging. |

The build refuses a dirty or unpushed git tree, and a version folder that already holds objects. A published version is permanent: the bucket denies overwrite and delete. The order of staging, checking and publishing is in the [runbook](../../docs/runbooks/training-datasets.md#publish-the-versioned-datasets).

## Taxonomy snapshot

`segdata taxonomy` fetches the MERMAID benthic hierarchy, from `/v1/benthicattributes/` and the growth forms in `/v1/choices/`, and publishes it as `taxonomy/mermaid-benthic/<version>/`. The version string follows the same `YYYYMMDD-<sha7>[-N]` form. The command prints the version and its manifest sha256, which are the taxonomy pin for `segdata build`.

| Flag | Default | Effect |
|---|---|---|
| `--store` | `outputs/segdata-store` | Where to publish: a local directory, or `s3`. Same as for `segdata build`. |
| `--tables-from <folder>` | none | Republish the tables of a staged taxonomy version folder, verified against its manifest, with no HTTP call. Use it to publish a version that you staged and checked locally. |
| `-v`, `--verbose` | off | Debug logging. |

## Tests

```bash
uv run --package segdata-etl --group tests pytest packages/segdata-etl/tests
```

The CoralNet suite is in `packages/segdata-etl/tests/coralnet/` and the external dataset suite in `packages/segdata-etl/tests/external/`. Pass one directory to run one suite.

The CoralNet tests are offline and use no network. The external dataset tests use an in-memory S3 fake and synthetic raw trees. `pytest-socket` blocks every host except localhost.
