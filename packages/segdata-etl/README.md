# segdata-etl

Builds the segmentation training data in S3: the CoralNet training manifest, the four external datasets, and the uniform-format versions of MERMAID, Coralscapes V2 and UCSD mosaics. Import package: `segdata_etl`; console script: `segdata`. Every dataset runs as `segdata build <dataset>`. The CoralNet split runs as `segdata split build` and `segdata split leakage` (see [CoralNet split](#coralnet-split)). A staged uniform version is published with `segdata promote` (see [Uniform datasets](#uniform-datasets)).

## Scope

- **Owns:** turning a CoralNet scrape into the training manifest, the resized images and the id2name JSON; checking each raw external release, converting it, and writing it to S3; and staging the `mermaid`, `coralscapes_v2` and `ucsd_mosaics` versions and promoting them to `mermaid-ic-datasets`. The CoralNet code is in `segdata_etl/datasets/coralnet/` (the CLI stages are in `build.py`). The external and uniform dataset modules are in `segdata_etl/datasets/`, and `promote` is in `segdata_etl/promote.py`. The shared helpers are in `segdata_etl/common/`: `images.py`, `jpeg_header.py`, `publish.py`, `labels.py`, `manifest.py`, and for the uniform builds `s3_objects.py`, `stage.py` and `hf_dense.py`:
  - `s3_objects.py`: write-once S3 copy and put. Each write sends `IfNoneMatch="*"`; a key that already holds the same content counts as done, and other content raises `ObjectConflict`.
  - `stage.py`: `LocalDirS3`, a directory that stands in for the S3 client so a build publishes to disk, and the version naming.
  - `hf_dense.py`: the shared build of a dense dataset from a pinned Hugging Face revision, used by `coralscapes_v2` and `ucsd_mosaics`.
- **Does not own:** downloading CoralNet. That is [`coralnet-scraper`](../coralnet-scraper/README.md).
- **Boundary:** The root `tests/test_package_boundaries.py` enforces these rules (run with the root suite): `segdata_etl` does not import `mermaidseg`; `mermaidseg` and `scripts` import neither `segdata_etl` nor `coralnet_scraper`; `coralnet_scraper` imports neither `mermaidseg` nor `segdata_etl`. Only `segdata_etl/datasets/coralnet/` may import `coralnet_scraper`, for its configuration and S3 key layout. The boundary test enforces this, so `common/` and the external dataset modules stay free of it. The `segdata` CLI loads the CoralNet build code, and with it `coralnet_scraper` and `ibis`, only when a `build coralnet` stage runs. The argparse wiring is in `datasets/coralnet/commands.py`, which imports neither. A root `uv sync` does not install the package.

```bash
uv run --package segdata-etl segdata build <dataset> ...
```

`<dataset>` is `coralnet`, `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals`, `benthos_yuval`, `mermaid`, `coralscapes_v2` or `ucsd_mosaics`. `coralnet` takes a stage after it (see [Run](#run)).

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
| `mermaid` | `--annotation-date-cutoff YYYY-MM-DD` (required): keep annotations with `updated_on` on or before that date (UTC); the whole day is included. `--annotations-uri` (default `s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet`). `--parent-version` and `--parent-manifest-sha256`, given together: a staged version whose image rows are reused when the source ETag is unchanged. | Images `mermaid/<image_id>.png`, else `.jpg`, in `coral-reef-training`. | `mermaid` for every image. | `copies.jsonl` |
| `coralscapes_v2` | `--revision` (default `8a4567b2fbec14882e09aebff63a7a1ca5e24dca`) | `josauder/coralscapesV2`, `instances` config. Licence `apache-2.0`. | The HF split: `train`, `validation` or `test`. | `uploads.jsonl` |
| `ucsd_mosaics` | `--revision` (default `6a6483841f8ee20bd7d8ee2aa760f10d1e9db704`) | `josauder/UCSD-mosaics-mirror`. Licence research-only: the repo has no LICENSE file at the pinned revision. | The `site` column, lowercased. | `uploads.jsonl` |

`mermaid` is a points dataset. Its labels are the BA+GF pairs, numbered from 1 and named `BA::GF`. The trailing `::` stays when there is no growth form. The plan file `copies.jsonl` lists one server-side copy per image, from `coral-reef-training` to `mermaid-ic-datasets`. The build stages no image bytes.

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

### Check a staged version

`scripts/check_dataset_parity.py` (at the repository root, outside this package) compares a version with the old loaders. With `--no-sync` it reads the version from `--cache-dir` and never calls S3, so `--cache-dir <stage>` checks a staged version before promote. Run `uv run python scripts/check_dataset_parity.py --help` for the other flags.

### Promote a version

Publishing is irreversible. `mermaid-ic-datasets` denies overwrite and delete, so a promoted version stays as it is. Get explicit approval before you run `promote`.

```bash
uv run --package segdata-etl segdata promote --stage-dir <stage> --dataset <dataset> --version <version>
```

`promote` also takes `--workers` (default 16) and `-v`, `--verbose`. It writes in this order:

1. Preflight, before any write. It checks the staged tables against `manifest.json`, refuses a manifest whose `stats.n_images` is 0, checks that the plan file names exactly the image and mask keys in `images.parquet` with the same size and checksums, and checks each staged upload file. It refuses a version whose `manifest.json` is already in the bucket. It then HEADs every planned destination key: each image and mask key, and each table key under `datasets/<dataset>/<version>/`. It refuses when a key already exists with a different ETag, size or CRC64NVME, and names up to five such keys. A key that exists with a matching fingerprint passes, so a rerun after a partial failure completes.
2. The images and masks: server-side copies for `copies.jsonl`, uploads for `uploads.jsonl`.
3. The tables.
4. `manifest.json`.

Every write is write-once. A key that already holds the planned content counts as done, and other content stops the run with `ObjectConflict`. A version is visible only when its `manifest.json` exists. If `promote` stops before step 4, run it again: it continues from the objects already written.

## External datasets

Loads the raw releases of four external datasets into S3, in the layout that their dataset classes in `mermaidseg/datasets/` read. One command for each dataset: `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` and `benthos_yuval`.

### Published files

Every dataset command writes the same set of files under `s3://<bucket>/<prefix>/`:

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

### Run an external dataset

Run from the repository root. `uv run --package segdata-etl` installs the package and its dependencies on first use. S3 access uses the default boto3 credential chain.

```bash
uv run --package segdata-etl segdata build <name> --data-dir <raw dir>
```

Flags on every dataset command:

| Flag | Default | Effect |
|---|---|---|
| `--bucket` | `dev-datamermaid-sm-sources` | Destination bucket. |
| `--prefix` | `external_validation_datasets/<name>` | Destination key prefix. |
| `--workers` | 16 | Threads for decoding and uploading images. |
| `--dry-run` | off | Write nothing to S3. |
| `--no-skip-existing-s3` | off | Upload every image, also those already on S3. |
| `--overwrite` | off | Replace the parquet and the JSON files if they exist. Benthos also uploads the `labels/` tiles again. |
| `-v`, `--verbose` | off | Debug logging. |

Flags for each dataset:

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

## Tests

```bash
uv run --package segdata-etl --group tests pytest packages/segdata-etl/tests
```

The CoralNet suite is in `packages/segdata-etl/tests/coralnet/`, the external dataset suite in `packages/segdata-etl/tests/external/`, the split suite in `packages/segdata-etl/tests/splits/` and the uniform dataset suite in `packages/segdata-etl/tests/uniform/`. Pass one directory to run one suite.

The CoralNet tests are offline and use no network. The external dataset tests use an in-memory S3 fake and synthetic raw trees. `pytest-socket` blocks every host except localhost.
