# Regenerate the derived configs

Some committed configs are built by scripts from other configs and from the data. This runbook rebuilds them after their inputs change. If you change an input and do not rebuild, the next run trains on the old values.

- [What to rebuild, and when](#what-to-rebuild-and-when)
- [Before you start](#before-you-start)
- [Rebuild the train + val data config](#rebuild-the-train--val-data-config)
- [Rebuild the concept removal list](#rebuild-the-concept-removal-list)
- [Rebuild the concept source mask](#rebuild-the-concept-source-mask)
- [After a rebuild](#after-a-rebuild)
- [When something fails](#when-something-fails)
- [Known gaps](#known-gaps)

## What to rebuild, and when

| Derived config | Built by | Inputs | Rebuild after a change to |
|---|---|---|---|
| `configs/data_config_trainval.yaml` | `scripts/build_trainval_data_config.py` | `configs/data_config.yaml` | Any part of `data_config.yaml` |
| `configs/concept_removals/min_coverage_trainval.csv` | `scripts/build_concept_removal.py` | The trainval data config, its label counts, `configs/class_to_concepts.csv` | The trainval data config, the data it reads, or `class_to_concepts.csv` |
| `configs/concept_source_masks/coralnet_only_false.csv` | `scripts/build_concept_source_mask.py` | The CoralNet train split of the trainval data config, `configs/class_to_concepts.csv` | The CoralNet sources or build, the CoralNet overrides in `configs/coralnet/`, or `class_to_concepts.csv` |

Both training configs (`configs/training_config_cbm_final.yaml` and `configs/training_config_cbm_final_hires.yaml`) point at the removal list and the source mask.

Each CSV has a `.meta.yaml` sidecar beside it. It records the inputs, their sha1 hashes and the git commit that built it. To see whether a CSV is stale, compare its recorded inputs with the current files.

The CoralscapesV2 probe point sets are derived too. They belong to evaluation; see [Rebuild the probe point sets](evaluate-a-model.md#rebuild-the-probe-point-sets).

## Before you start

- Complete [Set up a development environment](set-up-dev-environment.md). The removal list and the source mask read the data, so you need AWS credentials or a filled cache.
- `coralscapes_v2` and `ucsd_mosaics` load from Hugging Face into `HF_HOME`, not from the S3 cache. For these two you need a filled `HF_HOME` or Hugging Face access. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).
- Load the env file: `set -a; source .env; set +a`.
- If the CoralNet build changed, select it in the environment. See [Select the build](coralnet-data.md#select-the-build).

Rebuild in the order of this page. Each step reads the output of the step before it.

## Rebuild the train + val data config

`configs/data_config_trainval.yaml` is the final-run data mixture. The script copies `configs/data_config.yaml`, moves the CoralNet validation sources onto the train whitelist, and adds the CoralscapesV2 validation split to train. The test splits stay held out.

```bash
uv run python scripts/build_trainval_data_config.py
```

The script takes no arguments and does not change `data_config.yaml`. Never edit `data_config_trainval.yaml` by hand.

## Rebuild the concept removal list

The removal list names the concept channels that have too little support to train. The rules are in the docstring of `scripts/build_concept_removal.py` and in `mermaidseg/dataset_reconciliation/concept_removal.py`.

1. Count the labels in the train splits of the trainval config:

   ```bash
   uv run python scripts/train_label_counts.py \
       --data-config configs/data_config_trainval.yaml \
       --output dataset_stats/label_counts_trainval.parquet \
       --cache-dir <cache-root>
   ```

   The script counts only the train splits. It writes a parquet with the columns `dataset`, `image_id`, `label` and `n`. `--datasets` limits the count to a subset of dataset keys; a removal list built from a subset is incomplete, so omit it for a rebuild. Omit `--cache-dir` to read from S3.

   The dense datasets (`coralscapes_v2`, `ucsd_mosaics` and `benthos_yuval`) read every label mask, so this step is the slow one. `coralscapes_v2` and `ucsd_mosaics` come from Hugging Face, so they need `HF_HOME` filled (see [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache)) or Hugging Face access. `benthos_yuval` reads its masks from the S3 cache. It has `train: null` in the trainval config, so it matters only for a config that gives it a train split. The S3 cache covers the other datasets.

   Check: the script prints one summary row (images, labels, n) for each dataset that has a train split in `configs/data_config_trainval.yaml`, then `Wrote <path> (<rows> rows)`. At the time of writing those datasets are `coralnet`, `mermaid`, `coralscapes_v2`, `catlin_seaview`, `moorea_labeled_corals` and `ucsd_mosaics`. `benthos_yuval` and `pacific_labeled_corals` have `train: null` and do not appear. Compare the rows with the file, not with this list.

   The parquet is an intermediate output. Never commit it.

2. Build the removal list from the counts:

   ```bash
   uv run python scripts/build_concept_removal.py \
       --data-config configs/data_config_trainval.yaml \
       --label-counts dataset_stats/label_counts_trainval.parquet \
       --out configs/concept_removals/min_coverage_trainval.csv
   ```

Pass `--cache-dir` to read from the local S3 cache, as in step 1. `--coverage-out` writes an extra CSV with the coverage of every channel, which helps you review the result. Run the script with `--help` for the rule thresholds.

Do not commit `dataset_stats/`. Commit only the CSV and its `.meta.yaml`.

## Rebuild the concept source mask

The source mask lists the pairs of CoralNet source and binary concept that training masks: a source that has `False` points for a concept and no `True` points. The script reads annotations, not images.

```bash
uv run python scripts/build_concept_source_mask.py \
    --out configs/concept_source_masks/coralnet_only_false.csv
```

`--data-config` defaults to `configs/data_config_trainval.yaml`, the config that the final runs train on. Pass `--cache-dir` to read from the local S3 cache, and `--counts-out` to write the counts of every pair.

The `.meta.yaml` sidecar lists the CoralNet sources that the mask was built from (`source_ids`). Training and `evaluate.py --split train` stop with an error when a CoralNet source of the run is not in that list. If you see that error, rebuild the mask with the run's data config.

## After a rebuild

1. Review the diff of each CSV and its `.meta.yaml`.
2. Commit the derived configs together with the input change that caused them.
3. Fill the offline data cache again if the data changed. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).
4. For SageMaker, merge and build the image again. A job reads `configs/` from its image. See [Build the training image](train-on-sagemaker.md#build-the-training-image).

A change to the derived configs changes model outputs. Run the evaluation and put the metrics in the PR. See [Compare two models](evaluate-a-model.md#compare-two-models).

## When something fails

| Symptom | Cause | Fix |
|---|---|---|
| `DataLoadError: cache miss for s3://<bucket>/<key> in offline mode (MERMAIDSEG_S3_OFFLINE set)` | `MERMAIDSEG_S3_OFFLINE=1` is set, and the cache lacks an object that the step reads. | Unset `MERMAIDSEG_S3_OFFLINE`. On a host with AWS credentials, run `scripts/prefetch_cache.py --data-config <config> --cache-dir <cache-root>`, with the same data config and cache root as the failing step. The script exits with status 2 if `MERMAIDSEG_S3_OFFLINE` is set or no cache directory is configured. Or run the step itself with `MERMAIDSEG_S3_OFFLINE` unset and AWS credentials loaded. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache). |
| `train_label_counts.py` fails on `coralscapes_v2` or `ucsd_mosaics` with `ConnectionError: Couldn't reach '<repo>' on the Hub (OfflineModeIsEnabled)`. `<repo>` is `josauder/coralscapesV2` or `josauder/UCSD-mosaics-mirror`. | `HF_HUB_OFFLINE=1` is set, and `HF_HOME` does not hold the dataset. | Fill `HF_HOME` with `scripts/prefetch_cache.py`, or unset `HF_HUB_OFFLINE` and run with Hugging Face access. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache). |
| `train_label_counts.py` raises `ValueError: <key>: image <id> has mask id <n> with no class name` | The mask data and the class table of the dataset disagree. | For CoralscapesV2 and UCSD, check the Hugging Face dataset cached in `HF_HOME`. For Benthos, check `classes.json` on S3. If the data is inconsistent, do not build the removal list from its counts. |
| `train_label_counts.py` raises `ValueError: no train splits to count` | No selected dataset has a train split. Either `--datasets` has a misspelled key, or it names only datasets whose train split is `null` in the data config. | Correct the names in `--datasets`, or omit `--datasets` for a rebuild. |
| `build_concept_removal.py` exits with `N CoralNet count rows have no source in <config>; were the label counts built from the same data config?` | The label counts come from a different data config than `--data-config`. | Run step 1 again with the same `--data-config` that step 2 uses. |
| At training start, `ValueError: <path> was built for a concept table with sha1 ..., but this run uses ... (sha1 ...)` from `load_concept_removal` or `load_concept_source_mask` | `configs/class_to_concepts.csv` changed after the CSV was built. | Rebuild both derived configs, in the order of this page. |

## Known gaps

One measurement exists: a count of the 1,960 CoralscapesV2 train masks (the trainval config folds validation into train) took 51 s, including startup, from a filled Hugging Face cache on an Apple M1. It was measured on 2026-11-19 at commit 8029e3c. A full count across all six datasets has not been timed.
