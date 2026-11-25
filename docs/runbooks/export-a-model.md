# Export a model

This runbook turns a training checkpoint into a bundle: `model.json` and `weights.safetensors`. A bundle describes itself, so evaluation and inference need nothing else. To score the bundle, follow [Evaluate a model](evaluate-a-model.md).

- [Before you start](#before-you-start)
- [Export the checkpoint](#export-the-checkpoint)
- [When the export refuses a checkpoint](#when-the-export-refuses-a-checkpoint)
- [Checkpoints and bundles](#checkpoints-and-bundles)

## Before you start

- You need a checkpoint that `scripts/train.py` wrote, with the run's `model.json` in the same directory.
  - Slurm or local run: `outputs/model_checkpoints/<run>/`.
  - SageMaker run: download from `s3://$SM_STAGING_BUCKET/runs/<job>/checkpoints/model_checkpoints/<run>/`. See [Get the checkpoints](train-on-sagemaker.md#get-the-checkpoints).
- Have AWS credentials that read `mermaid-ic-datasets`, and run the command from a git checkout. The export checks the run's data pins against the bucket and its commit against your checkout.
- Run `git fetch origin` first. The export reads the local `origin/*` refs and does not fetch.
- Choose an output directory that is empty or does not exist yet. Its parent directory must exist.

## Export the checkpoint

```bash
uv run python scripts/export.py \
    --checkpoint outputs/model_checkpoints/<run>/<checkpoint> \
    --output-dir /path/to/bundle
```

The script prints the bundle path. The bundle holds `model.json` and `weights.safetensors`, the two files that `load_bundle` in `mermaidseg/inference/bundle.py` reads. `model.json` records the sha256 of `weights.safetensors`, and `load_bundle` refuses weights that do not match it.

Bundles exported before the `weights_sha256` key existed have no key, and `load_bundle` refuses them. Re-export each one from its checkpoint with `scripts/export.py`.

## When the export refuses a checkpoint

`scripts/export.py` refuses a checkpoint when it cannot prove what produced it.

| Refusal | What to do |
|---|---|
| No `model.json` beside the checkpoint | Copy the run's `model.json` into the checkpoint's directory. For SageMaker, download it with the checkpoint. |
| The checkpoint cannot be loaded, or lacks a required key | Use another checkpoint of the run. A `*.partial` file is incomplete. |
| No recorded lineage | Export a checkpoint that `scripts/train.py` wrote. Each checkpoint records the lineage of the session that wrote it. |
| The lineage records a dirty or unknown git tree | The session that wrote the checkpoint ran on uncommitted code. A run that resumed or warm-started from a checkpoint inherits that checkpoint's git state, so a source that was dirty or had no lineage gives the same refusal. `lineage.parent` in the checkpoint shows the source's git SHA and dirty state. You cannot export it. Train again from a clean commit, from scratch or from a clean checkpoint. The resumed or warm-started session must itself run on a clean tree; a clean checkpoint does not help if the code is dirty. |
| The lineage has no `data_version`, no datasets, or no taxonomy pin | The checkpoint was written by a run that did not pin its data. Train again with a data config that pins every dataset and the taxonomy. |
| A pinned manifest is missing from `mermaid-ic-datasets`, or its sha256 differs from the pin | The export checks the manifest of each dataset and the taxonomy in `lineage.data_version`. Check the AWS credentials and that the version was published, then export again. |
| `git_sha` is not a 40-character lowercase hex id, or no `origin/*` ref contains it | Run `git fetch origin`. If the commit is still missing, push it. |
| The config differs from the recorded lineage | The checkpoint and its lineage do not match. Use a checkpoint from one unchanged run. |
| The output directory is not empty | Choose an empty or new directory. |
| The parent of the output directory does not exist | Create the parent directory. |

## Data lineage

`model.json` records the data of the run under `lineage.data_version`:

```text
{"datasets": {<name>: {"version", "manifest_sha256", "resize", "images_dropped"}},
 "taxonomy": {"version", "manifest_sha256"}}
```

`resize` is the resize spec of the dataset, or null. `images_dropped` counts the images that the run skipped for that dataset.

## Checkpoints and bundles

A checkpoint is a `torch.save` pickle with the full training state. A bundle is SafeTensors plus `model.json`. See [Model Serialization Strategy](../../README.md#model-serialization-strategy) for how each is loaded.
