# Evaluate a model

This runbook scores an exported bundle with `scripts/evaluate.py`, the validation suite. The model comes only from `--bundle`. The input size, normalization, label maps and run settings come from the bundle's `model.json`.

- [Before you start](#before-you-start)
- [Run the evaluation](#run-the-evaluation)
- [Choose what to evaluate](#choose-what-to-evaluate)
- [Read the results](#read-the-results)
- [Compare two models](#compare-two-models)
- [Rebuild the probe point sets](#rebuild-the-probe-point-sets)

## Before you start

- Export the checkpoint to a bundle. See [Export a model](export-a-model.md).
- Fill the cache. `scripts/evaluate.py` needs `--cache-dir`, a directory that `scripts/sync_data.py` filled from the same `--data-config`. The script reads only that cache and makes no data call to S3:

  ```bash
  uv run python scripts/sync_data.py --data-config configs/data_config.yaml --cache-dir /path/to/cache
  ```

- The benthic taxonomy comes from the taxonomy version that the data config pins, not from the MERMAID API. The CoralscapesV2 probes read the synced `coralscapes_v2` version.

## Run the evaluation

```bash
uv run python scripts/evaluate.py \
    --bundle /path/to/bundle \
    --output-dir /path/to/results \
    --cache-dir /path/to/cache \
    --batch-size 8 --num-workers 16 --amp
```

`--amp` runs the forward pass in the run's `training.mixed_precision_dtype`.

Without `--strict`, a failed evaluation is only logged, and the script exits 0. To fail on any error, for example in a script or a job, pass `--strict`. The script then writes the summary and exits 1.

## Choose what to evaluate

`--evals` selects a subset of the evaluations. The default is all five.

| Eval | What it scores |
|---|---|
| `coralnet` | The CoralNet `--split` sources, per source and pooled |
| `pacific` | The Pacific Labeled Corals validation regions, per region and pooled |
| `coralscapes` | A CoralscapesV2 few-shot linear probe, 95 classes, 10 pixels per class |
| `coralscapes39` | The same probe on the published 39-class map, 50 pixels per class |
| `val_loss` | The concept-bottleneck loss components on the val splits of `--data-config`, per dataset and overall |

Other options:

- `--split` selects the CoralNet groups and the CoralscapesV2 split that the probes score. The default is `val`. The CoralNet groups are the rows with that `split` in the data config's split file (`configs/splits/coralnet.csv`). `--split train` allows only `coralnet` and `val_loss`, and reports the loss as `train_loss`.
- `--images-per-source N` scores a seeded random sample of at most N images per CoralNet source. `--sample-seed` sets the seed. The sample is written to `coralnet_<split>_sample.parquet`.
- `--data-config` defaults to `configs/data_config.yaml`. It sets the sources and splits of every evaluation.
- `--schema-config` sets the data config whose datasets freeze the concept channels, as in training. The default is `configs/data_config.yaml`.
- `--max-images-per-unit` and `--max-test-images` limit the data for a quick debug run.

Run the script with `--help` for the rest.

## Read the results

The script writes each section to `summary.json` and `summary.md` in `--output-dir`, and to its own subdirectory, for example `coralnet_val/`.

`summary.json` records `coralnet_split_in_training`. It is true when the data config trains on a CoralNet group that `--split` scores, and the script then logs a warning that names the groups. A score on such a split does not measure generalization.

CoralNet and Pacific report:

- class metrics
- taxonomic accuracy and macro F1 (all, living and coral)
- binary-concept metrics, with `bleached` and `anthropogenic` called out
- hierarchical metrics (consistency, mistake severity, ECE)

## Compare two models

Two scores compare only when they come from the same data. Run both bundles with the same `--data-config`, `--split`, `--evals` and, if you use it, the same `--images-per-source` and `--sample-seed`. Then compare the `summary.md` files.

Evaluation counts a taxonomic value with no model channel as `not_given` at its rank, as in training. Scores from before that rule do not compare with scores computed under it.

A change that alters model outputs must include metrics and the baseline they compare against in the PR. See item 7 of the working agreement in [AGENTS.md](../../AGENTS.md#2-working-agreement).

## Rebuild the probe point sets

The CoralscapesV2 probe point sets are committed in `configs/eval/`. Scores from different point sets do not compare, so a rebuild breaks comparison with earlier results.

`scripts/build_coralscapes_v2_probe_points.py` rebuilds them from the `coralscapes_v2` version that the data config pins, read from the synced `--cache-dir`. Its docstring has the command for each set:

| File | Set |
|---|---|
| `coralscapes_v2_probe_points.json` | 95 classes, 10 points per class (`--per-class 10`) |
| `coralscapes_v2_39_probe_k50.json` | 39 classes, 50 points per class (`--per-class 50 --label-space legacy39`) |
