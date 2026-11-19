# Runbooks

Each runbook gives the steps for one task, in order, from the prerequisites to a check that the task worked. The package READMEs, script docstrings and `--help` output are the reference for flags. A runbook links to them and does not repeat them.

## Set up

| Runbook | Use it to |
|---|---|
| [Set up a development environment](set-up-dev-environment.md) | Install the project, get access to S3 and DINOv3, and choose a tracker |
| [Set up the Slurm cluster](set-up-slurm.md) | Prepare a checkout and an env file on the cluster, once per account |

## Data

| Runbook | Use it to |
|---|---|
| [Get the training datasets](training-datasets.md) | Find where each dataset comes from, load one into S3, or add a new dataset |
| [Recreate and update the CoralNet data](coralnet-data.md) | Scrape CoralNet and build the dataset that training reads |
| [Regenerate the derived configs](derived-configs.md) | Rebuild the train + val data config, the concept removal list and the concept source mask |

## Train, export and evaluate

| Runbook | Use it to |
|---|---|
| [Train on Slurm](train-on-slurm.md) | Fill the offline cache and train the final model on the cluster (the default lane) |
| [Train on SageMaker](train-on-sagemaker.md) | Build the image and train the final model as SageMaker jobs (the backup lane) |
| [Export a model](export-a-model.md) | Turn a checkpoint into a deployable bundle |
| [Evaluate a model](evaluate-a-model.md) | Score a bundle with the validation suite and compare two models |

## Not written yet

These tasks wait on open issues. Write the runbook when the issue settles the process.

| Task | Waits on |
|---|---|
| Update the concept mappings (`configs/class_to_concepts.csv`). CoralNet label mappings come from the MERMAID API; see [Map new labels](coralnet-data.md#map-new-labels) | [#201](https://github.com/data-mermaid/mermaid-segmentation/issues/201), [#60](https://github.com/data-mermaid/mermaid-segmentation/issues/60) |
| Hand a model off for deployment | [#21](https://github.com/data-mermaid/mermaid-segmentation/issues/21) |

## Write a runbook

Use the shape of the existing runbooks:

1. One or two sentences on the outcome, then a table of contents.
2. **Before you start**: everything the reader needs before step 1.
3. Task-named sections with numbered steps, one action per step.
4. A check that the task worked.
5. **When something fails**, as a table of failure and action, where the task has known failures.
6. **Known gaps**, if any.

Add the runbook to this index and to the Documentation table in the root [README](../../README.md#documentation).
