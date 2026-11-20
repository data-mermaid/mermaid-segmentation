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
- The evaluation reads the same data as training. To read it from the local S3 cache, pass `--cache-dir`. Without `--cache-dir`, the script reads S3 and needs AWS credentials.
- The benthic hierarchy comes from the MERMAID API, and the script caches it in `--output-dir`. If the machine cannot reach the API, pass a saved copy with `--benthic-hierarchy-json`.
- A CoralNet evaluation also reads the CoralNet label mapping: from the API online, or from the snapshot that `scripts/prefetch_cache.py` writes offline. `--benthic-hierarchy-json` does not cover it. See [Map new labels](coralnet-data.md#map-new-labels).

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

- `--split` selects the CoralNet sources and the CoralscapesV2 split that the probes score. The default is `val`. `--split train` allows only `coralnet` and `val_loss`, and reports the loss as `train_loss`.
- `--images-per-source N` scores a seeded random sample of at most N images per CoralNet source. `--sample-seed` sets the seed. The sample is written to `coralnet_<split>_sample.parquet`.
- `--data-config` defaults to `configs/data_config.yaml`. It sets the sources and splits of every evaluation.
- `--coralscapes-root` reads CoralscapesV2 from a local mirror.
- `--max-images-per-unit` and `--max-test-images` limit the data for a quick debug run.

Run the script with `--help` for the rest.

## Read the results

The script writes each section to `summary.json` and `summary.md` in `--output-dir`, and to its own subdirectory, for example `coralnet_val/`.

CoralNet and Pacific report:

- class metrics
- taxonomic accuracy and macro F1 (all, living and coral)
- binary-concept metrics, with `bleached` and `anthropogenic` called out
- hierarchical metrics (consistency, mistake severity, ECE)

## Compare two models

Two scores compare only when they come from the same data. Run both bundles with the same `--data-config`, `--split`, `--evals` and, if you use it, the same `--images-per-source` and `--sample-seed`. Then compare the `summary.md` files.

A change that alters model outputs must include metrics and the baseline they compare against in the PR. See item 7 of the working agreement in [AGENTS.md](../../AGENTS.md#2-working-agreement).

## Rebuild the probe point sets

The CoralscapesV2 probe point sets are committed in `configs/eval/`. Scores from different point sets do not compare, so a rebuild breaks comparison with earlier results.

`scripts/build_coralscapes_v2_probe_points.py` rebuilds them from a local mirror of `josauder/coralscapesV2`. Its docstring has the command for each set:

| File | Set |
|---|---|
| `coralscapes_v2_probe_points.json` | 95 classes, 10 points per class (`--per-class 10`) |
| `coralscapes_v2_39_probe_k50.json` | 39 classes, 50 points per class (`--per-class 50 --label-space legacy39`) |
