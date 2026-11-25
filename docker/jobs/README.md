# docker/jobs/

Container definition for SageMaker training jobs launched by
`make sm-launch` (see [Train on SageMaker](../../docs/runbooks/train-on-sagemaker.md)). SageMaker is the backup to Slurm; both run
`scripts/train.py`.

## Files

| File | What |
| ---- | ---- |
| `Dockerfile` | GPU image: CUDA 12.6 base, Python 3.12 and packages from `uv.lock` via `uv sync --locked --no-dev --extra mlflow`; runs the entrypoint under `tini` (PID 1) so StopTrainingJob's SIGTERM reaches the trainer |
| `entrypoint.sh` | Shim: runs `python -m scripts.sagemaker.train_entrypoint` from `/opt/ml/code` |
| `local_smoke.sh` | Local build and offline check (`make sm-smoke`) |

On StopTrainingJob or MaxRuntime, SageMaker sends SIGTERM to the trainer. A single-GPU job ends its MLflow run as failed and flushes queued metrics before it exits. A multi-GPU job (torchrun) usually does not: rank 0 is typically blocked in a collective when the other ranks die, so torchrun kills it after 30 s and the MLflow run can be left RUNNING.

## Build

Images that jobs run are built and pushed only by the `sagemaker-image` workflow, from a committed tree on `main`: Actions,
`sagemaker-image`, Run workflow. The workflow runs in the `sagemaker-image` GitHub environment, whose deployment-branch policy (set in repo settings) allows only `main`; a dispatch from another branch fails. Each run pushes `training-<sha>` and moves `training-latest`. The image has
no `.git`; the workflow passes the commit as the `GIT_SHA` build arg, which becomes
`MERMAIDSEG_GIT_SHA`. A build without it yields an image whose entrypoint refuses to run.

A job runs the code and `configs/` baked into its image; only the run YAML, and for an init run the init checkpoint, travels with the job. `training-latest` is `main` at the last workflow run. Merge config and code changes, and rerun the workflow, before you launch. `make sm-validate` checks only the local checkout. The `git_sha` in `model.json` and the saved `config.yaml` record what ran.

## Local smoke

`make sm-smoke` builds `mermaid-segmentation-jobs:training-smoke-local` (no `GIT_SHA`, never
pushed), checks the trainer imports, checks that `tini` exists at its ENTRYPOINT path, confirms
the real ENTRYPOINT refuses a run with no config channel, and runs `launch --check` on each
`sagemaker/runs/*.yaml` inside the image.

## Prerequisites

- The job execution role must be granted `secretsmanager:GetSecretValue` on the secret named by
  `HF_TOKEN_SECRET_ID`. Grant it in the job role. It is the only Secrets Manager secret the job reads.
- Store the secret as plain-text `SecretString`, not JSON key/value: the entrypoint exports the raw string.
- The job execution role needs `sagemaker:CallMlflowAppApi` on the MLflow App ARN named by `SM_MLFLOW_TRACKING_ARN` (`arn:aws:sagemaker:<region>:<account-id>:mlflow-app/<app-id>`). The `sagemaker-mlflow:*` actions govern only the older MLflow tracking servers; granting them alone returns 403. The tracker also writes artifacts (`config/config.json`, `metadata/*.json`, `metadata/model.json`) directly to the app's S3 artifact store (the `ArtifactStoreUri` that `aws sagemaker describe-mlflow-app` returns), so the role needs `s3:PutObject` and `s3:GetObject` on that bucket and prefix. Without the S3 grant those writes fail with only a warning, and the run finishes without its config in MLflow. The launcher passes the ARN to the job as `MLFLOW_TRACKING_URI`.
- The repo variable `AWS_ECR_PUSH_ROLE_ARN` holds the OIDC push role defined in the mermaid-api IaC. With it unset, the credentials step fails and no image is pushed. The job runs in the `sagemaker-image` environment, so the role's trust policy must match `sub` = `repo:data-mermaid/mermaid-segmentation:environment:sagemaker-image`. A trust policy that matches a ref-based `sub` (`...:ref:refs/heads/main`) fails with AccessDenied at the credentials step.
- The `sagemaker-image` GitHub environment exists, with a deployment-branch policy that allows only `main`.
- The launcher caller needs `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`, `iam:PassRole` on the job role, `ecr:DescribeImages` on `mermaid-segmentation-jobs`, `s3:PutObject` on `runs/*/config/` and `s3:GetObject` on `runs/*/checkpoints/*` in the staging bucket.
- The job execution role needs S3 read on `mermaid-ic-datasets` (the image has no prewarmed cache; the entrypoint runs `scripts/sync_data.py` into `/opt/ml/cache` before training) and on the staging bucket (the config prefix, and the source job's checkpoints for the 768 px init). It needs S3 write on its own job prefix (`runs/<job>/output/` and `runs/<job>/checkpoints/`).
- Both run YAMLs exceed the 5-day default of the account quota "Longest run time for a training job": `cbm_final` 480 h, `cbm_final_res768` 168 h. Raise the quota to at least 480 h.
