# coralnet-dataset

Builds the CoralNet training dataset from a scrape on S3. Import package: `coralnet_dataset`; console script: `coralnet-dataset`.

## Scope

- **Owns:** turning a scrape into the training manifest, the resized images and the id2name JSON that `CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`) reads.
- **Does not own:** downloading CoralNet. That is [`coralnet-scraper`](../coralnet-scraper/README.md).
- **End to end:** to recreate or update the CoralNet data, scrape through training, follow the [CoralNet data runbook](../../docs/runbooks/coralnet-data.md).
- **Boundary:** the package imports the `coralnet_scraper` configuration and S3 key layout. It does not import `mermaidseg`, and `mermaidseg` does not import it. The root `tests/test_package_boundaries.py` enforces both (run with the root suite). A root `uv sync` does not install it.

## Inputs

The build reads the per-source folders that the scraper writes under `s3://<bucket>/<prefix>/s<source_id>/`: `annotations.csv`, `image_list.csv`, `labelset.csv` and `images/<image_id>.jpg`. See the [scraper's S3 layout](../coralnet-scraper/README.md#s3-layout).

## Outputs

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

## Run

Run from the repository root. `uv run --package coralnet-dataset` installs the package and its dependencies on first use:

```bash
uv run --package coralnet-dataset coralnet-dataset <subcommand>
```

Run the subcommands in this order, or run `all` to do the four in one invocation with one version tag:

| Subcommand | Reads | Flags in addition to the common ones |
|---|---|---|
| `build-annotations` | The source CSVs on S3 | `--source-ids 1,2,3`: the sources to read. Default: every `s<id>/` folder under the prefix. |
| `build-images` | The annotations parquet, and the image headers on S3 | `--annotations <path>` (required) |
| `resize` | The images parquet, and the images on S3 | `--images <path>` (required) |
| `build-manifest` | Both parquets, and the resized objects on S3 | `--annotations <path>`, `--images <path>` (both required) |
| `all` | As above | `--source-ids` |

Common flags, on every subcommand:

| Flag | Default | Effect |
|---|---|---|
| `--version-tag` | `YYYYMMDD_<sha>` | Run version tag. It names the run only. |
| `--overwrite` | off | Let an upload replace an existing key under `etl-outputs/coralnet/<version>/`. Without it, an upload to an existing key raises `RuntimeError`. A training run that pins this tag then reads different data. |
| `--output-dir` | `outputs/coralnet-dataset` | Local base directory. Files go under `<output-dir>/<version>/`. |
| `--upload-to-s3` | off | Upload each output file to `etl-outputs/coralnet/<version>/`. |
| `--workers` | `MERMAID_CORALNET_ETL_WORKERS`, else 16 | Concurrent S3 workers. Must be at least 1. |
| `-v`, `--verbose` | off | Debug logging. |

Each invocation computes its own default version tag. If you run the subcommands one at a time, pass the same `--version-tag` to each so the outputs share one run directory.

## Check a build for completeness

`build-annotations` excludes a source when it fails a check. Each excluded source has exactly one kind, the first check it failed:

| Kind | Meaning |
|---|---|
| `annotations_missing` | `annotations.csv` is not on S3. |
| `image_list_missing` | `image_list.csv` is not on S3. |
| `labelset_missing` | `labelset.csv` is not on S3. |
| `csv_read_failed` | A CSV exists but could not be read. |
| `columns_missing` | A CSV lacks a required column. `annotations.csv` needs `Name`, `Row`, `Column`, `Label ID`; `image_list.csv` needs `Name`, `Image Page`; `labelset.csv` needs `Label ID`, `Name`. |
| `annotations_empty` | `annotations.csv` has no rows. |
| `image_list_incomplete` | An annotation row matches no `image_list.csv` row by `Name`. |
| `label_unnamed` | A Label ID used by the source is not named by any included source's labelset. |

Rows with a non-numeric `Row`, `Column` or `Label ID` are dropped with a warning; they do not exclude the source. A Label ID with two different names across the included sources raises `ValueError` and stops the build.

At the end, `build-annotations` logs a summary at error level: the number of excluded sources, then one line per kind with the sorted source ids, for example `labelset_missing: [3, 4]`.

`build-images` logs the image count per `header_status`. `ok`, `sof_not_found_extended` and `pil_fallback` carry dimensions. `not_found`, `corrupt`, `s3_error` and `worker_exception` do not, and the manifest drops those images.

`resize` logs the number of images resized, corrupted and failed. An image is corrupted when its download fails or it does not pass inspection. It is failed when the resize or the upload raises.

Exit codes:

| Subcommand | Exit code 1 when |
|---|---|
| `build-annotations` | Any source is excluded, or no annotation rows are built. |
| `build-images` | Any image has no dimensions. |
| `resize` | Any image is corrupted or failed. |
| `build-manifest` | Only on an uncaught error. |
| `all` | Any step returns 1. Every step runs. |

Every subcommand also ends with a non-zero code on an uncaught error, such as invalid configuration or an upload refused as above. The output files are written before the exit code is set, so a run with exit code 1 still leaves its outputs.

### Environment variables

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

## Use a build in training

`CoralNetDataset` (`mermaidseg/datasets/coralnet/coralnet_dataset.py`) reads the manifest from its `source_bucket`. Set one of these to select a build; the path wins:

- `MERMAID_CORALNET_MANIFEST_VERSION=<version>` reads `etl-outputs/coralnet/<version>/coralnet_training_manifest_<version>.parquet`.
- `MERMAID_CORALNET_MANIFEST_PATH=<key>` reads that key.

With neither set, it reads a pinned default build. It reads the label names from `coralnet_id2name_<run>.json` in the same directory as the manifest, where `<run>` is the directory name. A custom path must keep that layout.

`image_s3_key` and the id2name JSON are keys in `MERMAID_CORALNET_BUCKET`, so a build is readable only when that bucket is the dataset's `source_bucket` (the `CoralNetDataset` default).

## Known gaps

- A partially exported `annotations.csv` cannot be detected from S3. A paginated source whose export was interrupted leaves a partial file (see the scraper's [Resume and `--force`](../coralnet-scraper/README.md#resume-and---force)). The build includes that source with fewer images than CoralNet has. Tracked in issue #231.
- Duplicate images (the same photo in several sources) are removed in training by `CoralNetDataset`'s `image_blacklist` and `annotation_overrides`. This package does not produce those files. Tracked in issue #231.

## Tests

```bash
uv run --package coralnet-dataset --group tests pytest packages/coralnet-dataset/tests
```

The tests are offline and use no network.
