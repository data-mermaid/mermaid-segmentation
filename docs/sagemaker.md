# Running mermaid-segmentation on SageMaker

For dev-account TrainingJob and ProcessingJob support. The launcher
convention (role ARN, bucket, schema, ECR tagging) is defined in
[mermaid-api/iac/sagemaker-launcher-convention.md](../../mermaid-api/iac/sagemaker-launcher-convention.md).
This doc is the seg-specific runbook on top of that.

## Prerequisites

- AWS SSO with the `SageMaker` Identity Center permission set on
  account `554812291621`. See convention doc for `~/.aws/config`.
- Docker installed locally.
- `pip install -e .` succeeds, and the `sagemaker` optional extra is
  installed: `uv sync --extra sagemaker`.
- An MLflow App is provisioned at
  `arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2`.

### Preflight

Account-specific values (`SM_ROLE_ARN`, `MLFLOW_TRACKING_URI`, etc.) go in
**`.env`** (gitignored), loaded by direnv — see `.env.example`. The Makefile
does not embed ARNs.

Run before your first job (or after credential changes):

```bash
uv sync --extra sagemaker
aws sso login --profile wcs-sso
export MERMAID_AWS_MODE=launcher && direnv reload

make sm-check
```

The check validates SDK version, region, credentials, execution role
trust (`sagemaker.amazonaws.com`), and warns about staging-bucket IAM and
`HF_TOKEN`.

## One-time: build and push the image

```bash
export AWS_PROFILE=wcs-launcher
ACCT=554812291621
IMG=$ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs

aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin \
        $ACCT.dkr.ecr.us-east-1.amazonaws.com

docker buildx build --platform linux/amd64 \
    -t $IMG:training-latest -f docker/jobs/Dockerfile .
docker push $IMG:training-latest

# Tag the same image for processing use:
docker tag $IMG:training-latest $IMG:processing-latest
docker push $IMG:processing-latest
```

Smoke-test locally before pushing:

```bash
bash docker/jobs/local_smoke.sh training
bash docker/jobs/local_smoke.sh processing
```

## Run a training job

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_training.py \
    --run-config sagemaker/runs/example-training.yaml \
    --config-dir sagemaker/configs/example/ \
    --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2
```

Outputs:
- Run ID and CloudWatch URL printed at submission time.
- MLflow run appears at the `mermaidseg` MLflow App under the
  `--run-name` from the YAML's `config.overrides`.
- Final model artifact saved to
  `s3://dev-datamermaid-sm-data/runs/<run-id>/output/`.

## Run a processing job (eval/inference)

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_processing.py \
    --run-config sagemaker/runs/example-processing.yaml \
    --config-dir sagemaker/configs/example/
```

Add new processing tasks by adding a `--task=<name>` branch to
`scripts/sagemaker_processing_entrypoint.py` and a corresponding
routine in `mermaidseg/`.

## Tuning

| Instance | GPU | $/hr | Use case |
|---|---|---|---|
| `ml.g5.xlarge` | A10G (24GB) | ~$1.41 | Eval / inference; small training runs |
| `ml.g5.2xlarge` | A10G | ~$1.52 | Standard training run (DINOv3-base, batch 4-8) |
| `ml.g5.4xlarge` | A10G | ~$2.03 | Larger batch, more CPU for data loading |
| `ml.p3.2xlarge` | V100 (16GB) | ~$3.83 | When A10G's 24GB isn't enough memory |

## Debug a failed job

CloudWatch log group: `/aws/sagemaker/TrainingJobs` (or `/ProcessingJobs`).

```bash
export AWS_PROFILE=wcs-launcher
aws logs tail /aws/sagemaker/TrainingJobs \
    --log-stream-name-prefix <run-id>/ \
    --follow
```

Or reproduce locally:

```bash
docker run --rm -it \
    -v $(pwd)/sagemaker/configs/example:/opt/ml/input/data/config:ro \
    -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_train_entrypoint.py \
    mermaid-segmentation-jobs:training-smoke-local
```
