# Regenerate the derived configs

Three committed configs are built by scripts from `configs/data_config.yaml`, the synced data and `configs/class_to_concepts.csv`. This runbook rebuilds them after their inputs change. If you change an input and do not rebuild, the next run trains on the old values. Training refuses a stale label space.

- [What to rebuild, and when](#what-to-rebuild-and-when)
- [Before you start](#before-you-start)
- [Rebuild the concept removal list](#rebuild-the-concept-removal-list)
- [Rebuild the concept source mask](#rebuild-the-concept-source-mask)
- [Rebuild the label space](#rebuild-the-label-space)
- [After a rebuild](#after-a-rebuild)
- [When something fails](#when-something-fails)

## What to rebuild, and when

| Derived config | Built by | Inputs | Rebuild after a change to |
|---|---|---|---|
| `configs/concept_removals/min_coverage_trainval.csv` | `scripts/build_concept_removal.py` | The data config, the `stats/label_counts.parquet` of each dataset it pins, `configs/class_to_concepts.csv` | A pin or a split selection in the data config, or `class_to_concepts.csv` |
| `configs/concept_source_masks/coralnet_only_false.csv` | `scripts/build_concept_source_mask.py` | The data config, the `points.parquet` of the pinned `coralnet` version, `configs/class_to_concepts.csv` | The `coralnet` pin or its split selection, `configs/splits/coralnet.csv`, or `class_to_concepts.csv` |
| `configs/label_spaces/<name>/` (`cbm_final`) | `scripts/build_label_space.py` | The data config, the tables and taxonomy of each pinned dataset, the label keys of the training config (`class_subset`, `label_roll_up`, concept paths), `configs/class_to_concepts.csv`, the removal list and the source mask | A change to `class_subset`, `label_roll_up`, `class_to_concepts.csv`, the removal list or the source mask, or a new dataset or taxonomy pin. Rebuild it after the other two. |

Both training configs (`configs/training_config_cbm_final.yaml` and `configs/training_config_cbm_final_hires.yaml`) point at the removal list, the source mask and `configs/label_spaces/cbm_final` (`training.label_space`).

Each CSV has a `.meta.yaml` sidecar beside it. It records the inputs, the sha1 hash of the concept table (`concepts_csv_sha1`), the dataset versions and the full commit sha that built it (`git_sha`). To see whether a CSV is stale, compare its recorded inputs with the current files. A test in CI, `tests/configs/test_derived_configs.py`, checks each committed training config. For every `training.concept_removal` and `training.concept_source_mask` CSV it names, the sidecar's `concepts_csv_sha1` must match the file at that config's `training.concept_mapping_path`. A missing sidecar fails the test.

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

## Rebuild the label space

The label space is the resolved list of classes and concept channels of a model. The directory holds four files:

- `classes.csv`: `class_id`, `name`, `kind` (`ignore`, `mermaid` or `extra`) and `mermaid_benthic_attribute_id`.
- `concepts.csv`: the concept channel layout.
- `source_labels.csv`: every row of each `labels.parquet`, with its target, roll-up, class, drop reason, train-split count and concept values.
- `label_space.meta.yaml`: the dataset and taxonomy pins, the sha1 of the label keys, the sha1 of each input file and the git state.

Rebuild the removal list and the source mask first, because their sha1 hashes go into the label space.

```bash
uv run python scripts/build_label_space.py \
    --training-config configs/training_config_cbm_final.yaml \
    --cache-dir <cache-dir> \
    --out configs/label_spaces/cbm_final
```

The script reads tables only, so `--cache-dir` needs no images. To fill a cache with only the manifests, tables and taxonomy that the data config pins, add `--tables-only` to the sync:

```bash
uv run python scripts/sync_data.py --data-config configs/data_config.yaml --cache-dir <cache-dir> --tables-only
```

A cache from a full sync (see [Before you start](#before-you-start)) works too.

`--data-config` defaults to `configs/data_config.yaml`. `--acknowledge` names a CSV of accepted refusals (columns `dataset`, `native_name`, `kind`, `reason`); the default is `acknowledged.csv` in `--out`.

The build refuses, and writes nothing, in these cases:

- A label that trains a class has no concept row, and `acknowledged.csv` does not list it.
- A family, genus or species concept names a MERMAID attribute under a different top-level root than the target of the label, and `acknowledged.csv` does not list it.
- A row of `acknowledged.csv` matches no case. This includes a row for a label that drops to background, which is never a missing-concept case. The error says `stale acknowledgement`.
- A row of `acknowledged.csv` has a `kind` that is not a case kind, or an empty `reason`.
- `--out` is missing and a `.<name>.*.old` backup sits beside it. A kill during an earlier write leaves this state. Rename the backup to `--out`, then run the build again.

The build writes to a temporary directory and swaps it onto `--out`, so a failed build leaves the previous files. Other files in `--out`, such as `acknowledged.csv`, stay.

Commit the whole directory, `acknowledged.csv` included.

## After a rebuild

1. Commit the input change first (for example `configs/class_to_concepts.csv`), then build from that clean tree. Each builder refuses a tree where tracked files differ from `HEAD` before it writes anything, and it refuses a `--data-config` or `--schema-config` that is not a git-tracked file of the repo (`DirtyTree`). The sidecar's `git_sha` is the input commit. If you rebuild both configs, commit the first one's output before you build the second, or write it outside the repo; otherwise the second build sees a dirty tree.
2. Review the diff of each CSV and its `.meta.yaml`.
3. Commit the derived CSVs and their sidecars.
4. For SageMaker, merge and build the image again. A job reads `configs/` from its image. See [Build the training image](train-on-sagemaker.md#build-the-training-image).

A change to the derived configs changes model outputs. Run the evaluation and put the metrics in the PR. See [Compare two models](evaluate-a-model.md#compare-two-models).

## When something fails

| Symptom | Cause | Fix |
|---|---|---|
| `NotSynced` from a builder | `--cache-dir` has no sync marker for a version that the data config selects. | Run `scripts/sync_data.py` with the same data config and `--cache-dir`. |
| `FileNotFoundError` for a manifest or table | The cache is empty, or was synced from a different data config. | Run `scripts/sync_data.py` with the same data config and `--cache-dir`. |
| `DirtyTree` from a builder | Tracked files differ from `HEAD`, or a `--data-config` or `--schema-config` is not a git-tracked file. | Commit or stash the changes, then build again. If the dirt is the first config's output, commit it before you build the second. For an untracked input, commit the input or pass a tracked file. |
| `tests/configs/test_derived_configs.py` fails in CI | `configs/class_to_concepts.csv` changed after a sidecar was built, so its `concepts_csv_sha1` no longer matches. | Rebuild both derived configs, in the order of this page. |
| At training start, `ValueError: label space ... does not match this run; rebuild it with scripts/build_label_space.py` | The pins, label keys, concepts CSV, removal list or source mask differ from those the label space was built from. The error lists each mismatch. | Rebuild the label space. |
