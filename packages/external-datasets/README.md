# external-datasets

Loads the raw releases of four external datasets into S3, in the layout that their dataset classes in `mermaidseg/datasets/` read. Import package: `external_datasets`; console script: `external-datasets`.

## Scope

- **Owns:** checking a raw release, converting it to an annotations parquet and an image tree, and writing both to S3 with `classes.json`, `colors.json` and `manifest.json`. One subcommand for each dataset: `catlin_seaview`, `moorea_labeled_corals`, `pacific_labeled_corals` and `benthos_yuval`.
- **Does not own:** the CoralNet data (see the [CoralNet data runbook](../../docs/runbooks/coralnet-data.md)), the MERMAID parquet, or the Hugging Face datasets. The README beside each dataset class says where its data comes from.
- **Boundary:** the package does not import `mermaidseg`, `coralnet_scraper` or `coralnet_dataset`, and `mermaidseg` and `scripts` do not import it. The root `tests/test_package_boundaries.py` enforces this (run with the root suite). A root `uv sync` does not install it.

## Outputs

Every subcommand writes the same set of files under `s3://<bucket>/<prefix>/`:

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
| `dataset_name` | The subcommand name. |
| `source.citation`, `source.raw_data_url`, `source.license` | Where the raw release comes from, and its licence. |
| `created_at_utc` | When the run wrote the manifest. |
| `s3.bucket`, `s3.prefix`, `s3.annotations_parquet`, `s3.image_key_template` | Where the files are. |
| `schema.parquet_columns` | Each parquet column and what it holds. |
| `ingestion` | Facts about this run that differ for each dataset (see [Ingestion keys](#ingestion-keys)). |

The defaults are `--bucket dev-datamermaid-sm-sources` and `--prefix external_validation_datasets/<name>`. These match the defaults of the dataset classes.

## Write order and safety

A run writes in this order:

1. Before it reads the raw data, the run checks that the parquet, `classes.json`, `colors.json` and `manifest.json` are not on S3. If one is there, the run stops with `FileExistsError`. Pass `--overwrite` to replace them.
2. The run reads the raw release and decodes every image fully. Catlin Seaview and Moorea drop each damaged image and its annotations, and record it in `ingestion.damaged_images`. Pacific stops with an error before it writes to S3.
3. The run uploads the images. An image whose key is already on S3 is skipped, whatever its content. Pass `--no-skip-existing-s3` to upload every image again. `--overwrite` does not change how images are uploaded, except for the Benthos `labels/` tiles: with `--overwrite`, Benthos uploads them again even if their keys exist. Label pixel values are the IDs in `classes.json`, so a rewritten `classes.json` must not keep old masks.
4. The run writes the parquet, then `classes.json`, then `colors.json`, then `manifest.json`.

`manifest.json` is written last, so a prefix without it holds an incomplete run. If a run stops before step 4, run it again: the images already on S3 are skipped. If it stops during step 4, run it again with `--overwrite`.

`--dry-run` writes nothing to S3. It still reads S3 for the step 1 check and for the list of existing images, so it needs credentials and fails where the real run fails. It does not stop local work: Catlin Seaview still downloads its zips, and Benthos still writes its tiles.

## Ingestion keys

Every dataset records `images_uploaded`: the number of images this run sent to S3, not counting images skipped because their keys exist.

| Subcommand | Keys |
|---|---|
| `catlin_seaview` | `damaged_images`: the dropped images. `images_uploaded`: a count for each region. |
| `moorea_labeled_corals` | `damaged_images`: the dropped images. `case_merged_labels`: raw label to merged label, for labels merged by letter case. `off_annotations_dropped`: the number of annotations dropped because their label is `off`. `images_uploaded`. |
| `pacific_labeled_corals` | `excluded_sites`. `annotations_dropped_unknown_archived`: annotations dropped because their `archived` label ID is not in the label table. `annotations_dropped_unmapped_imageid`: annotations dropped because their image ID is not in the image map. `images_uploaded`. The run logs each drop count as a warning. |
| `benthos_yuval` | `excluded_sites`, `tile_size`, `label_key_template`, `num_tiles` (a count for each site), `images_uploaded` and `labels_uploaded`. |

## Run

Run from the repository root. `uv run --package external-datasets` installs the package and its dependencies on first use. S3 access uses the default boto3 credential chain.

```bash
uv run --package external-datasets external-datasets <name> --data-dir <raw dir>
```

Flags on every subcommand:

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

| Subcommand | Flags |
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
uv run --package external-datasets --group tests pytest packages/external-datasets/tests
```

The tests use an in-memory S3 fake and synthetic raw trees. `pytest-socket` blocks every host except localhost.
