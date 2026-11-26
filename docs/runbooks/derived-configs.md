# Regenerate the derived configs

Two committed configs are built by scripts from `configs/data_config.yaml`, the synced data and `configs/class_to_concepts.csv`. This runbook rebuilds them after their inputs change. If you change an input and do not rebuild, the next run trains on the old values.

- [What to rebuild, and when](#what-to-rebuild-and-when)
- [Before you start](#before-you-start)
- [Rebuild the concept removal list](#rebuild-the-concept-removal-list)
- [Rebuild the concept source mask](#rebuild-the-concept-source-mask)
- [After a rebuild](#after-a-rebuild)
- [When something fails](#when-something-fails)

## What to rebuild, and when

| Derived config | Built by | Inputs | Rebuild after a change to |
|---|---|---|---|
| `configs/concept_removals/min_coverage_trainval.csv` | `scripts/build_concept_removal.py` | The data config, the `stats/label_counts.parquet` of each dataset it pins, `configs/class_to_concepts.csv` | A pin or a split selection in the data config, or `class_to_concepts.csv` |
| `configs/concept_source_masks/coralnet_only_false.csv` | `scripts/build_concept_source_mask.py` | The data config, the `points.parquet` of the pinned `coralnet` version, `configs/class_to_concepts.csv` | The `coralnet` pin or its split selection, `configs/splits/coralnet.csv`, or `class_to_concepts.csv` |

Both training configs (`configs/training_config_cbm_final.yaml` and `configs/training_config_cbm_final_hires.yaml`) point at the removal list and the source mask.

Each CSV has a `.meta.yaml` sidecar beside it. It records the inputs, their sha1 hashes, the dataset versions, the full commit sha that built it (`git_sha`) and whether tracked files differed from that commit (`git_dirty`). To see whether a CSV is stale, compare its recorded inputs with the current files.

The CoralscapesV2 probe point sets are derived too. They belong to evaluation; see [Rebuild the probe point sets](evaluate-a-model.md#rebuild-the-probe-point-sets).

## Before you start

- Complete [Set up a development environment](set-up-dev-environment.md).
- Fill the cache from the data config that the builders read. Both builders read only the synced cache and make no S3 call:

  ```bash
  uv run python scripts/sync_data.py --data-config configs/data_config.yaml --cache-dir <cache-dir>
  ```

  See [Get the data](training-datasets.md#get-the-data).
- Load the env file: `set -a; source .env; set +a`.

Rebuild in the order of this page.

## Rebuild the concept removal list

The removal list names the concept channels that have too little support to train. The rules are in the docstring of `scripts/build_concept_removal.py` and in `mermaidseg/dataset_reconciliation/concept_removal.py`.

```bash
uv run python scripts/build_concept_removal.py \
    --cache-dir <cache-dir> \
    --out configs/concept_removals/min_coverage_trainval.csv
```

The script counts labels from `stats/label_counts.parquet` of every dataset that has a train block in `--data-config`, over the groups of that block. `--data-config` defaults to `configs/data_config.yaml`. `--coverage-out` writes an extra CSV with the coverage of every channel, which helps you review the result. Run the script with `--help` for the rule thresholds.

Commit only the CSV and its `.meta.yaml`.

## Rebuild the concept source mask

The source mask lists the pairs of CoralNet source and binary concept that training masks: a source that has `False` points for a concept and no `True` points. The script reads `points.parquet` of the pinned `coralnet` version, not images.

```bash
uv run python scripts/build_concept_source_mask.py \
    --cache-dir <cache-dir> \
    --out configs/concept_source_masks/coralnet_only_false.csv
```

`--data-config` defaults to `configs/data_config.yaml`. `--split` selects the block that supplies the groups; the default is `train`. `--counts-out` writes the counts of every pair.

The `.meta.yaml` sidecar lists the CoralNet sources that the mask was built from (`source_ids`). Training and `evaluate.py --split train` stop with an error when a CoralNet source of the run is not in that list. If you see that error, rebuild the mask with the run's data config.

## After a rebuild

1. Build from a clean, committed tree. A dirty tree is recorded in the sidecar, not refused. If you rebuild both configs, write the first one outside the repo (or commit it) before you build the second; otherwise the second build sees a dirty tree. Before you commit, check that each sidecar says `git_dirty: false`.
2. Review the diff of each CSV and its `.meta.yaml`.
3. Commit the derived configs together with the input change that caused them.
4. For SageMaker, merge and build the image again. A job reads `configs/` from its image. See [Build the training image](train-on-sagemaker.md#build-the-training-image).

A change to the derived configs changes model outputs. Run the evaluation and put the metrics in the PR. See [Compare two models](evaluate-a-model.md#compare-two-models).

## When something fails

| Symptom | Cause | Fix |
|---|---|---|
| `NotSynced` from a builder | `--cache-dir` has no sync marker for a version that the data config selects. | Run `scripts/sync_data.py` with the same data config and `--cache-dir`. |
| `FileNotFoundError` for a manifest or table | The cache is empty, or was synced from a different data config. | Run `scripts/sync_data.py` with the same data config and `--cache-dir`. |
| At training start, `ValueError: <path> was built for a concept table with sha1 ..., but this run uses ... (sha1 ...)` from `load_concept_removal` or `load_concept_source_mask` | `configs/class_to_concepts.csv` changed after the CSV was built. | Rebuild both derived configs, in the order of this page. |
