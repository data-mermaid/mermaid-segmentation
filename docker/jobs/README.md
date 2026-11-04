# docker/jobs/

Container definition for SageMaker jobs launched by
`scripts/launch_training.py` (the backup to Slurm; both run
`scripts/train.py` from the repo root). The launcher chooses the
in-container entrypoint via `CONTAINER_ENTRYPOINT_SCRIPT`.

## Files

| File | What |
| ---- | ---- |
| `Dockerfile` | GPU image (CUDA 12.6 base; Python 3.12 and packages from `uv.lock` via `uv sync --locked`) with mermaidseg installed editable |
| `entrypoint.sh` | Shim: `exec`s the script named by `CONTAINER_ENTRYPOINT_SCRIPT` |
| `local_smoke.sh` | Build + run-locally test (`bash local_smoke.sh`) |

## Prerequisites

- The job execution role needs `secretsmanager:GetSecretValue` on the secrets named by
  `HF_TOKEN_SECRET_ID` / `WANDB_API_KEY_SECRET_ID` (granted in the mermaid-api IaC, not here).
- With `use_spot`, checkpoints sync to `s3://<staging>/runs/<run-id>/checkpoints/`; that prefix
  holds the run's artifacts (`model_checkpoints/<run-name>/`), not `model.tar.gz`.

## Tag convention

- `training-latest`, `training-smoke`, `training-YYYY-MM-DD`
- `user-<name>-<purpose>-YYYY-MM-DD`

See `mermaid-api/iac/sagemaker-launcher-convention.md` for authoritative rules.

## Build/push

ECR push requires the launcher role (`dev-mermaid-sagemaker-launcher-role`, profile
`wcs-launcher`). The default SageMaker SSO role (`mermaid-core`) can pull but NOT push —
pushing with it fails `ecr:InitiateLayerUpload ... no identity-based policy allows`. The
`docker login` credential is what's checked on push, so log in with `--profile wcs-launcher`.

```bash
cd mermaid-segmentation
ACCT=554812291621
IMG=$ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs

# --profile wcs-launcher: the role with ECR push permission (NOT mermaid-core)
aws ecr get-login-password --region us-east-1 --profile wcs-launcher \
    | docker login --username AWS --password-stdin \
        $ACCT.dkr.ecr.us-east-1.amazonaws.com

docker buildx build --platform linux/amd64 \
    --build-arg GIT_SHA=$(git rev-parse HEAD) \
    -t $IMG:training-latest -f docker/jobs/Dockerfile .
docker push $IMG:training-latest
```

`make sm-build` runs the same build with the `GIT_SHA` build arg.
