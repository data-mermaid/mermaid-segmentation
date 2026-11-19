# Regenerate the derived configs

Some committed configs are built by scripts from other configs and from the data. This runbook rebuilds them after their inputs change. If you change an input and do not rebuild, the next run trains on the old values.

- [What to rebuild, and when](#what-to-rebuild-and-when)
- [Before you start](#before-you-start)
- [Rebuild the train + val data config](#rebuild-the-train--val-data-config)
- [Rebuild the concept removal list](#rebuild-the-concept-removal-list)
- [Rebuild the concept source mask](#rebuild-the-concept-source-mask)
- [After a rebuild](#after-a-rebuild)
- [Known gaps](#known-gaps)

## What to rebuild, and when

| Derived config | Built by | Inputs | Rebuild after a change to |
|---|---|---|---|
| `configs/data_config_trainval.yaml` | `scripts/build_trainval_data_config.py` | `configs/data_config.yaml` | Any part of `data_config.yaml` |
| `configs/concept_removals/min_coverage_trainval.csv` | `scripts/build_concept_removal.py` | The trainval data config, its label counts, `configs/class_to_concepts.csv` | The trainval data config, the data it reads, or `class_to_concepts.csv` |
| `configs/concept_source_masks/coralnet_only_false.csv` | `scripts/build_concept_source_mask.py` | The CoralNet train split, `configs/class_to_concepts.csv` | The CoralNet sources or build, the CoralNet overrides in `configs/coralnet/`, or `class_to_concepts.csv` |

Both training configs (`configs/training_config_cbm_final.yaml` and `configs/training_config_cbm_final_hires.yaml`) point at the removal list and the source mask.

Each CSV has a `.meta.yaml` sidecar beside it. It records the inputs, their sha1 hashes and the git commit that built it. To see whether a CSV is stale, compare its recorded inputs with the current files.

The CoralscapesV2 probe point sets are derived too. They belong to evaluation; see [Rebuild the probe point sets](evaluate-a-model.md#rebuild-the-probe-point-sets).

## Before you start

- Complete [Set up a development environment](set-up-dev-environment.md). The removal list and the source mask read the data, so you need AWS credentials or a filled cache.
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

1. Count the labels in the train splits of the trainval config. This needs `scripts/train_label_counts.py`, which is not on `main` (see [Known gaps](#known-gaps)):

   ```bash
   uv run python scripts/train_label_counts.py \
       --data-config configs/data_config_trainval.yaml \
       --output dataset_stats/label_counts_trainval.parquet
   ```

2. Build the removal list from the counts:

   ```bash
   uv run python scripts/build_concept_removal.py \
       --data-config configs/data_config_trainval.yaml \
       --label-counts dataset_stats/label_counts_trainval.parquet \
       --out configs/concept_removals/min_coverage_trainval.csv
   ```

Pass `--cache-dir` to read from the local S3 cache. `--coverage-out` writes an extra CSV with the coverage of every channel, which helps you review the result. Run the script with `--help` for the rule thresholds.

Do not commit `dataset_stats/`. Commit only the CSV and its `.meta.yaml`.

## Rebuild the concept source mask

The source mask lists the pairs of CoralNet source and binary concept that training masks: a source that has `False` points for a concept and no `True` points. The script reads annotations, not images.

```bash
uv run python scripts/build_concept_source_mask.py \
    --out configs/concept_source_masks/coralnet_only_false.csv
```

`--data-config` defaults to `configs/data_config.yaml`. The committed mask was built from that file, as its `.meta.yaml` records. Pass `--cache-dir` to read from the local S3 cache, and `--counts-out` to write the counts of every pair.

## After a rebuild

1. Review the diff of each CSV and its `.meta.yaml`.
2. Commit the derived configs together with the input change that caused them.
3. Fill the offline data cache again if the data changed. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).
4. For SageMaker, merge and build the image again. A job reads `configs/` from its image. See [Build the training image](train-on-sagemaker.md#build-the-training-image).

A change to the derived configs changes model outputs. Run the evaluation and put the metrics in the PR. See [Compare two models](evaluate-a-model.md#compare-two-models).

## Known gaps

`scripts/build_concept_removal.py` needs the label counts from `scripts/train_label_counts.py`. That script, and `scripts/dataset_table_stats.py` that it imports, are not on `main`. They are in commit `88f7bac^`. Until they are restored, extract both to run step 1:

```bash
git show 88f7bac^:scripts/train_label_counts.py > scripts/train_label_counts.py
git show 88f7bac^:scripts/dataset_table_stats.py > scripts/dataset_table_stats.py
```

With both files in place, `train_label_counts.py --help` runs on the current code. A full count has not been checked against the current code. Do not commit the extracted copies without a review.
