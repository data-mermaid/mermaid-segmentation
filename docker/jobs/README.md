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

## Local smoke

`make sm-smoke` builds `mermaid-segmentation-jobs:training-smoke-local` (no `GIT_SHA`, never
pushed), checks the trainer imports, and runs `launch --check` on each `sagemaker/runs/*.yaml`
inside the image.

## Prerequisites

- The job execution role needs `secretsmanager:GetSecretValue` on the secrets named by
  `HF_TOKEN_SECRET_ID` and `WANDB_API_KEY_SECRET_ID`. The mermaid-api IaC grants it.
- The account quota "Longest run time for a training job" must be raised for `cbm_final` (480 h).
