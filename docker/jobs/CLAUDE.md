# docker/jobs/

Container definition for SageMaker jobs launched by
`scripts/launch_training.py` and `scripts/launch_processing.py`. One
image, both job types — the launcher chooses which in-container
entrypoint runs via `CONTAINER_ENTRYPOINT_SCRIPT`.

## Files

| File | What |
| ---- | ---- |
| `Dockerfile` | GPU image (PyTorch 2.3 CUDA 12.1, Python 3.11) with mermaidseg installed editable |
| `entrypoint.sh` | Shim: `exec`s the script named by `CONTAINER_ENTRYPOINT_SCRIPT` |
| `local_smoke.sh` | Build + run-locally test (`bash local_smoke.sh [training\|processing]`) |

## Tag convention

- `training-latest`, `training-smoke`, `training-YYYY-MM-DD`
- `processing-latest`, `processing-smoke`, `processing-YYYY-MM-DD`
- `user-<name>-<purpose>-YYYY-MM-DD`

See `mermaid-api/iac/sagemaker-launcher-convention.md` for authoritative rules.

## Build/push

```bash
cd mermaid-segmentation
ACCT=554812291621
IMG=$ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs

aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin \
        $ACCT.dkr.ecr.us-east-1.amazonaws.com

docker buildx build --platform linux/amd64 \
    -t $IMG:training-latest -f docker/jobs/Dockerfile .
docker push $IMG:training-latest
```
