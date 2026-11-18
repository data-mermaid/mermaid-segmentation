# docker/jobs/

Container definition for SageMaker training jobs launched by
`python -m scripts.sagemaker.launch`. SageMaker is the backup to Slurm; both run
`scripts/train.py`.

## Files

| File | What |
| ---- | ---- |
| `Dockerfile` | GPU image: CUDA 12.6 base, Python 3.12 and packages from `uv.lock` via `uv sync --locked --extra sagemaker`; runs the entrypoint under `tini` (PID 1) so StopTrainingJob's SIGTERM reaches the trainer |
| `entrypoint.sh` | Shim: runs `python -m scripts.sagemaker.train_entrypoint` from `/opt/ml/code` |
| `local_smoke.sh` | Local build and offline check (`make sm-smoke`) |

## Build

Images are built only by the `sagemaker-image` workflow, from a committed tree on `main`: Actions,
`sagemaker-image`, Run workflow. The workflow runs in the `sagemaker-image` GitHub environment, whose deployment-branch policy (set in repo settings) allows only `main`; a dispatch from another branch fails. Each run pushes `training-<sha>` and moves `training-latest`. The image has
no `.git`; the workflow passes the commit as the `GIT_SHA` build arg, which becomes
`MERMAIDSEG_GIT_SHA`. A build without it yields an image whose entrypoint refuses to run.

A job runs the code and `configs/` baked into its image; only the run YAML travels with the job. `training-latest` is `main` at the last workflow run. Merge config and code changes, and rerun the workflow, before you launch. `make sm-validate` checks only the local checkout. The `git_sha` in `model.json` and the saved `config.yaml` record what ran.

## Local smoke

`make sm-smoke` builds `mermaid-segmentation-jobs:training-smoke-local` (no `GIT_SHA`, never
pushed), checks the trainer imports, and runs `launch --check` on each `sagemaker/runs/*.yaml`
inside the image.

## Prerequisites

- The job execution role must be granted `secretsmanager:GetSecretValue` on the secret named by
  `HF_TOKEN_SECRET_ID`; the mermaid-api IaC does not grant it today. It is the only Secrets Manager secret the job reads.
- Store the secret as plain-text `SecretString`, not JSON key/value: the entrypoint exports the raw string.
- The job execution role needs `sagemaker-mlflow` access to the MLflow tracking server named by `SM_MLFLOW_TRACKING_ARN`. The launcher passes that ARN to the job as `MLFLOW_TRACKING_URI`.
- The repo variable `AWS_ECR_PUSH_ROLE_ARN` holds the OIDC push role defined in the mermaid-api IaC. With it unset, the credentials step fails and no image is pushed. The job runs in the `sagemaker-image` environment, so the role's trust policy must match `sub` = `repo:data-mermaid/mermaid-segmentation:environment:sagemaker-image`. A trust policy that matches a ref-based `sub` (`...:ref:refs/heads/main`) fails with AccessDenied at the credentials step.
- The `sagemaker-image` GitHub environment exists, with a deployment-branch policy that allows only `main`.
- The launcher caller needs `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`, `iam:PassRole` on the job role, `ecr:DescribeImages` on `mermaid-segmentation-jobs`, `s3:PutObject` on `runs/*/config/` and `s3:GetObject` on `runs/*/checkpoints/*` in the staging bucket.
- The job execution role needs S3 read on the training-data buckets the trainer streams from (there is no prewarmed cache) and on the staging bucket (the config prefix, and the source job's checkpoints for the 768 px init). It needs S3 write on its own job prefix (`runs/<job>/output/` and `runs/<job>/checkpoints/`).
- Both run YAMLs exceed the 5-day default of the account quota "Longest run time for a training job": `cbm_final` 480 h, `cbm_final_res768` 168 h. Raise the quota to at least 480 h.
