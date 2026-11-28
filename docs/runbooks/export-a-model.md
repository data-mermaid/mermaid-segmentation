# Export a model

This runbook turns a training checkpoint into a bundle: `model.json`, `weights.safetensors` and `label_space/`. A bundle describes itself, so evaluation and inference need nothing else. To score the bundle, follow [Evaluate a model](evaluate-a-model.md).

- [Before you start](#before-you-start)
- [Export the checkpoint](#export-the-checkpoint)
- [When the export refuses a checkpoint](#when-the-export-refuses-a-checkpoint)
- [Checkpoints and bundles](#checkpoints-and-bundles)

## Before you start

- You need a checkpoint that `scripts/train.py` wrote, with the run's `model.json` and `label_space/` directory in the same directory.
  - Slurm or local run: `outputs/model_checkpoints/<run>/`.
  - SageMaker run: download from `s3://$SM_STAGING_BUCKET/runs/<job>/checkpoints/model_checkpoints/<run>/`. See [Get the checkpoints](train-on-sagemaker.md#get-the-checkpoints).
- Have AWS credentials that read `mermaid-ic-datasets`, and run the command from a git checkout. The export checks the run's data pins against the bucket and its commit against your checkout.
- Run `git fetch` first. The export reads the local remote-tracking branches of every remote and does not fetch.
- Choose an output directory that is empty or does not exist yet. Its parent directory must exist.

## Export the checkpoint

```bash
uv run python scripts/export.py \
    --checkpoint outputs/model_checkpoints/<run>/<checkpoint> \
    --output-dir /path/to/bundle
```

The script prints the bundle path. The bundle holds `model.json`, `weights.safetensors` and a `label_space/` directory. `load_bundle` in `mermaidseg/inference/bundle.py` reads the first two and ignores `label_space/`. The export copies each file that `model.json` lists under `label_space` from the directory beside the checkpoint into the bundle's `label_space/`. It does not compare the files' sha256 with the values that `model.json` lists. A listed file missing from that directory fails the copy with `FileNotFoundError`. The evaluation that reads the bundle checks each file against its sha256 (`mermaidseg/evaluation/val_loss.py`). `model.json` records the sha256 of `weights.safetensors`, and `load_bundle` refuses weights that do not match it.

Bundles exported before the `weights_sha256` key existed have no key, and `load_bundle` refuses them. Re-export each one from its checkpoint with `scripts/export.py`.

## When the export refuses a checkpoint

`scripts/export.py` refuses a checkpoint when it cannot prove what produced it.

| Refusal | What to do |
|---|---|
| No `model.json` beside the checkpoint | Copy the run's `model.json` into the checkpoint's directory. For SageMaker, download it with the checkpoint. |
| The checkpoint cannot be loaded, or lacks a required key | Use another checkpoint of the run. A `*.partial` file is incomplete. |
| No recorded lineage | Export a checkpoint that `scripts/train.py` wrote. Each checkpoint records the lineage of the session that wrote it. |
| The lineage records a dirty git tree | The session that wrote the checkpoint was dirty: a tracked file differed from `HEAD`, git could not be read, or a run input was not git-tracked. Export refuses a dirty run. A run that resumed or warm-started from a checkpoint inherits that checkpoint's git state, so a source that was dirty gives the same refusal. A checkpoint without lineage cannot be resumed or warm-started at all. `lineage.parent` in the checkpoint shows the source's git SHA, dirty state and data version. You cannot export it. Train again from a clean commit, from scratch or from a clean checkpoint. The resumed or warm-started session must itself run on a clean tree; a clean checkpoint does not help if the code is dirty. |
| A pinned manifest is missing from `mermaid-ic-datasets`, or its sha256 differs from the pin | The export checks the manifest of each dataset and the taxonomy in `lineage.data_version`. Check the AWS credentials and that the version was published, then export again. |
| No remote branch contains `git_sha` | Run `git fetch`. If the commit is still missing, push it. |
| The output directory is not empty | Choose an empty or new directory. |
| The parent of the output directory does not exist | Create the parent directory. |

## Data lineage

`model.json` records the data of the run under `lineage.data_version`:

```text
{"datasets": {<name>: {"version", "manifest_sha256", "resize"}},
 "taxonomy": {"version", "manifest_sha256"}}
```

`resize` is the resize spec of the dataset, or null.

## Checkpoints and bundles

A checkpoint is a `torch.save` pickle with the full training state. A bundle is SafeTensors plus `model.json` and the `label_space/` files. See [Model Serialization Strategy](../../README.md#model-serialization-strategy) for how each is loaded.
