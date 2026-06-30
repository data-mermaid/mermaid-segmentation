# docker/jobs/

Container definition for SageMaker jobs launched by
`scripts/launch_training.py` and `scripts/launch_processing.py`. One
image, both job types — the launcher chooses which in-container
entrypoint runs via `CONTAINER_ENTRYPOINT_SCRIPT`.

## Files

| File | What |
| ---- | ---- |
| `Dockerfile` | GPU image (PyTorch 2.4.1 CUDA 12.4, Python 3.11) with mermaidseg installed editable |
| `entrypoint.sh` | Shim: `exec`s the script named by `CONTAINER_ENTRYPOINT_SCRIPT` |
| `local_smoke.sh` | Build + run-locally test (`bash local_smoke.sh [training\|processing]`) |

## Tag convention

- `training-latest`, `training-smoke`, `training-YYYY-MM-DD`
- `processing-latest`, `processing-smoke`, `processing-YYYY-MM-DD`
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
    -t $IMG:training-latest -f docker/jobs/Dockerfile .
docker push $IMG:training-latest
```
