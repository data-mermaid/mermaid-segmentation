# Get the training datasets

Training and evaluation read pinned dataset versions from a local cache. `configs/data_config.yaml` pins the versions, `scripts/sync_data.py` fills the cache, and `scripts/train.py` and `scripts/evaluate.py` read only the cache. This runbook says where the versions live, how to fill the cache, and how to add a dataset or update MERMAID.

CoralNet has its own runbook: [Recreate and update the CoralNet data](coralnet-data.md).

- [How a dataset reaches training](#how-a-dataset-reaches-training)
- [Where each dataset lives](#where-each-dataset-lives)
- [Get the data](#get-the-data)
- [Add a dataset](#add-a-dataset)
- [Update the MERMAID data](#update-the-mermaid-data)
- [Versions that cannot be pinned](#versions-that-cannot-be-pinned)
- [Known gaps](#known-gaps)

## How a dataset reaches training

1. An ETL module in `packages/segdata-etl` stages a dataset version and `segdata promote` publishes it to the `mermaid-ic-datasets` bucket. The version has a manifest, parquet tables (`images`, `labels`, `points` for point datasets, `stats/label_counts`) and the image files. Each label is mapped to a MERMAID benthic attribute and growth form in `labels.parquet`. The [segdata-spec README](../../packages/segdata-spec/README.md) defines the layout.
2. The data config pins the version by `version` and `manifest_sha256`. It also pins the taxonomy snapshot that the label roll-up uses (`segdata taxonomy` stages it and `segdata promote --taxonomy` publishes it), and selects the groups of each split.
3. `scripts/sync_data.py` downloads each pinned version into the cache and verifies it. For CoralNet it also resizes each image to 2048 px on its longest edge.
4. The readers in `mermaidseg/data/` open the cache: `PointDataset` for point datasets and `DenseDataset` for mask datasets.

The run records the pins in `model.json` under `lineage.data_version`. See [Data lineage](export-a-model.md#data-lineage).

## Where each dataset lives

| Dataset | Kind | Original source | Used in `data_config.yaml` for |
|---|---|---|---|
| `coralnet` | points | [CoralNet](https://coralnet.ucsd.edu/), through `coralnet-scraper` | train (train and val groups of the split file), test |
| `mermaid` | points | The MERMAID platform | train |
| `coralscapes_v2` | masks | Hugging Face `josauder/coralscapesV2` | train (train and validation groups), test |
| `ucsd_mosaics` | masks | Hugging Face `josauder/UCSD-mosaics-mirror` | train |
| `catlin_seaview` | points | [data.qld.edu.au/public/Q1281](http://data.qld.edu.au/public/Q1281/) | train |
| `moorea_labeled_corals` | points | EDI package [knb-lter-mcr.5006](https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006) | train |
| `pacific_labeled_corals` | points | Dryad [doi:10.5061/dryad.m5pr3](https://datadryad.org/dataset/doi:10.5061/dryad.m5pr3) | validation |
| `benthos_yuval` | masks | Dryad [doi:10.5061/dryad.8cz8w9gm3](https://datadryad.org/dataset/doi:10.5061/dryad.8cz8w9gm3) | validation |

Every version is under `s3://mermaid-ic-datasets/datasets/<name>/<version>/`, the taxonomy under `s3://mermaid-ic-datasets/taxonomy/mermaid-benthic/<version>/`, and the image files under the `image_store` prefix of the manifest. The citation, licence and rebuild steps of each external dataset are in the [segdata-etl README](../../packages/segdata-etl/README.md#external-datasets).

## Get the data

Prerequisites: AWS credentials that read `mermaid-ic-datasets`, and disk space for the cache. The SageMaker run YAMLs set `volume_gb: 1500` for the cache and the checkpoints.

1. Sync the pinned versions:

   ```bash
   uv run python scripts/sync_data.py --data-config configs/data_config.yaml --cache-dir <cache-dir>
   ```

   `--workers N` sets the download threads (default 16). The script prints one line per dataset: the version, the resize id (`original` when the dataset is not resized) and the number of dropped images. It prints the taxonomy version last.

2. Pass the same directory to the run: `scripts/train.py --cache-dir <cache-dir>` or `scripts/evaluate.py --cache-dir <cache-dir>`.

The sync checks each manifest against its pinned sha256, each table against the manifest, and each image or mask against its recorded size and checksums. It then writes a marker for each synced version. A reader raises `NotSynced` for a version, group selection or resize that has no marker, so sync with the data config that the run uses. A second sync into the same cache downloads only what is missing.

The launchers sync for you: [Slurm](train-on-slurm.md#fill-the-data-cache) syncs on the login node, and the [SageMaker](train-on-sagemaker.md) entrypoint syncs into `/opt/ml/cache` before training.

## Add a dataset

A new data source adds an ETL module that publishes a version. `mermaidseg` does not change.

1. Write the ETL module in `packages/segdata-etl/src/segdata_etl/datasets/`, on the pattern of an existing module of the same kind: `catlin_seaview.py` (points), `benthos_yuval.py` or `ucsd_mosaics.py` (masks).
2. Map its labels to the MERMAID taxonomy. External datasets use a curated CSV in `packages/segdata-etl/src/segdata_etl/label_maps/curated/<name>.csv`. The mapping goes into the published version's `labels.parquet`.
3. Register the module in `packages/segdata-etl/src/segdata_etl/cli.py` so that `segdata build <name>` runs it.
4. If the taxonomy pin that the dataset maps to is not yet published, stage it with `segdata taxonomy --stage-dir <stage>` and publish it with `segdata promote --stage-dir <stage> --taxonomy mermaid-benthic --version <tax>`. Dataset builds refuse a pin that is only staged. See [Taxonomy snapshot](../../packages/segdata-etl/README.md#taxonomy-snapshot).
5. Stage, check and publish the version. Commit and push first, because staging refuses a dirty or unpushed tree. The commands are in the [segdata-etl README](../../packages/segdata-etl/README.md). Record the `version` and `manifest_sha256` that each command prints.
6. Add the dataset to `configs/data_config.yaml` with that pin and a `train`, `val` or `test` block that selects its groups.
7. Add a row for each label to `configs/class_to_concepts.csv`, with the dataset in `source_dataset_source` and the label in `source_label_class_name`.
8. [Regenerate the derived configs](derived-configs.md).
9. Sync the cache. See [Get the data](#get-the-data).

## Update the MERMAID data

`mermaid_confirmed_annotations.parquet` in `s3://coral-reef-training/mermaid/` grows as MERMAID users confirm annotations. This repository does not write it. A `mermaid` dataset version freezes it at a cutoff date.

1. Stage a new version with the new cutoff. Pass the previous version as the parent so that labels keep their ids and unchanged images are reused:

   ```bash
   uv run --package segdata-etl segdata build mermaid --stage-dir <stage> \
     --annotation-date-cutoff YYYY-MM-DD \
     --parent-version <v> --parent-manifest-sha256 <sha> \
     --extractor-spec <spec.json>
   ```

   The parent version must already be in `<stage>`. The build also stages a feature set from the feature vectors next to the images in `coral-reef-training`, and stops when an image has none unless you pass `--allow-missing-features`. See [Stage a version](../../packages/segdata-etl/README.md#stage-a-version) and [MERMAID feature set](../../packages/segdata-etl/README.md#mermaid-feature-set) for the flags.

2. Publish it with `segdata promote --stage-dir <stage> --dataset mermaid --version <version>`. Get approval first: a published version cannot be changed. Promote writes the images, the feature vectors, the tables and the dataset manifest, then publishes the feature set. If it stops after the dataset manifest, run it again: it writes only the missing feature vectors and the feature set. See [Promote a version](../../packages/segdata-etl/README.md#promote-a-version).
3. Replace the `mermaid` pin in `configs/data_config.yaml` with the printed `version` and `manifest_sha256`.
4. [Regenerate the derived configs](derived-configs.md).
5. Sync the cache. See [Get the data](#get-the-data).

## Versions that cannot be pinned

A version whose manifest has an `extra_files` key cannot be read: the manifest schema in `packages/segdata-spec/src/segdata_spec/manifest.py` has no such field and forbids unknown keys. Pin a newer version of the same dataset. The current pins are in `configs/data_config.yaml`.

## Known gaps

- The MERMAID parquet changes in place. A new `mermaid` version with the same cutoff holds the same annotations only if no annotation older than the cutoff changed in between.
