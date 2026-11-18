# docker/jobs/

Container definition for SageMaker training jobs launched by
`python -m scripts.sagemaker.launch`. SageMaker is the backup to Slurm; both run
`scripts/train.py`.

## Files

| File | What |
| ---- | ---- |
| `Dockerfile` | GPU image: CUDA 12.6 base, Python 3.12 and packages from `uv.lock` via `uv sync --locked` |
| `entrypoint.sh` | Shim: runs `python -m scripts.sagemaker.train_entrypoint` from `/opt/ml/code` |
| `local_smoke.sh` | Local build and offline check (`make sm-smoke`) |

## Build

Images are built only by the `sagemaker-image` workflow, from a committed tree: Actions,
`sagemaker-image`, Run workflow. It pushes `training-<sha>`, and also `training-latest` when run on `main`. The image has
no `.git`; the workflow passes the commit as the `GIT_SHA` build arg, which becomes
`MERMAIDSEG_GIT_SHA`. A build without it yields an image whose entrypoint refuses to run.

Run the workflow on `main`. A job runs the code and `configs/` baked into its image; only the run YAML travels with the job. `training-latest` is `main` at the last workflow run on `main`. Merge config and code changes, and rerun the workflow, before you launch. `make sm-validate` checks only the local checkout. The `git_sha` in `model.json` and the saved `config.yaml` record what ran.

## Local smoke

`make sm-smoke` builds `mermaid-segmentation-jobs:training-smoke-local` (no `GIT_SHA`, never
pushed), checks the trainer imports, and runs `launch --check` on each `sagemaker/runs/*.yaml`
inside the image.

## Prerequisites

- The job execution role needs `secretsmanager:GetSecretValue` on the secrets named by
  `HF_TOKEN_SECRET_ID` and `WANDB_API_KEY_SECRET_ID`. The mermaid-api IaC grants it.
- The job execution role needs S3 read on the training-data buckets the trainer streams from (there is no prewarmed cache) and on the staging bucket (the config prefix, and the source job's checkpoints for the 768 px init). It needs S3 write on its own job prefix (`runs/<job>/output/` and `runs/<job>/checkpoints/`).
- Both run YAMLs exceed the 5-day default of the account quota "Longest run time for a training job": `cbm_final` 480 h, `cbm_final_res768` 168 h. Raise the quota to at least 480 h.
