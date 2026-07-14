# Running code in SageMaker jobs

How to run `mermaid-segmentation` code as SageMaker **TrainingJob**s and **ProcessingJob**s in the
dev account. One Docker image backs both; the launcher selects the in-container entrypoint.

The launcher convention (role ARN, bucket, schema, ECR tagging) is defined in
`mermaid-api/iac/sagemaker-launcher-convention.md`. This page is the seg-specific runbook on top of
that. Dataset-specific tasks (e.g. CoralNet ETL) are documented on their own pages.

## Prerequisites

- AWS SSO with the `SageMaker` Identity Center permission set on the project's dev account.
- Docker installed locally.
- `pip install -e .` succeeds and the `sagemaker` extra is installed: `uv sync --extra sagemaker`.
- An MLflow App provisioned in the dev account; its ARN is your `MLFLOW_TRACKING_URI` (set in `.env`).

### Preflight

Account-specific values (`SM_ROLE_ARN`, `MLFLOW_TRACKING_URI`, …) go in **`.env`** (gitignored,
loaded by direnv — see `.env.example`); the Makefile embeds no ARNs. Run before your first job (or
after credential changes):

```bash
uv sync --extra sagemaker
aws sso login --profile wcs-sso
export MERMAID_AWS_MODE=launcher && direnv reload
make sm-check
```

`sm-check` validates SDK version, region, credentials, execution-role trust
(`sagemaker.amazonaws.com`), and warns about staging-bucket IAM and `HF_TOKEN`.

## One-time: build and push the image

ECR push requires the **`wcs-launcher`** profile (the launcher role); the read-only SSO role cannot
push.

```bash
export AWS_PROFILE=wcs-launcher
ACCT=<your-aws-account-id>
IMG=$ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs

aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin $ACCT.dkr.ecr.us-east-1.amazonaws.com

docker buildx build --platform linux/amd64 -t $IMG:training-latest -f docker/jobs/Dockerfile .
docker push $IMG:training-latest

# Same image, tagged for processing use:
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
    --mlflow-tracking-uri $MLFLOW_TRACKING_URI   # from .env
```

Outputs:
- Run ID and CloudWatch URL printed at submission.
- An MLflow run under the `--run-name` from the YAML's `config.overrides`.
- Final model artifact at `s3://dev-datamermaid-sm-data/runs/<run-id>/output/`.

A run YAML's `config:` block names the model/training/data/logger configs and any `overrides:`
(forwarded to `scripts/train.py` as CLI flags). See `sagemaker/configs/example/` for the shape.

## Run a processing job

Processing jobs share the training image but route to different code via `--task`. The entrypoint is
`scripts/sagemaker_processing_entrypoint.py`.

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_processing.py \
    --run-config sagemaker/runs/<your-run>.yaml \
    --config-dir sagemaker/configs/<your-config-dir>/
```

Built-in tasks are `eval` and `inference`. Dataset/ETL tasks are registered by their owning modules
and documented on their own pages.

### Adding a new processing task

1. Add the task name to `choices=` in `scripts/sagemaker_processing_entrypoint.py`.
2. Add an `elif args.task == "<name>":` block that imports from `mermaidseg/` and calls its
   `main(extra)`.
3. If the implementation lives under `scripts/` rather than `mermaidseg/`, move it to a package path
   under `mermaidseg/` and register a console script in `pyproject.toml` so the in-container import
   resolves.

### Credentials in run YAMLs

Any `${VAR}` in a run YAML `env:` block is expanded from the shell environment at launch via
`os.path.expandvars()`. Set sensitive values in `.env` and reference them as `${MY_VAR}` — they are
never committed.

### Sharding

For a task that fans out over a list of items, add a `shard:` block to the run YAML:

```yaml
processing:
  container_args:
    - --task=<task>
  shard:
    items_from: items.csv      # CSV in --config-dir; must have a header row
    items_column: id           # which column to read
    workers: 5                 # number of parallel ProcessingJobs
    per_worker_arg: --ids      # CLI arg each worker receives its slice as
```

The launcher splits `items_from` across `workers` jobs and submits them in parallel. Each job's logs
land in `/aws/sagemaker/ProcessingJobs` under stream `<job-name>-N/algo-1`.

## Instance sizing

| Instance | GPU | $/hr | Use case |
|---|---|---|---|
| `ml.g5.xlarge` | A10G (24GB) | ~$1.41 | Eval / inference; small training runs |
| `ml.g5.2xlarge` | A10G | ~$1.52 | Standard training run (DINOv3-base, batch 4-8) |
| `ml.g5.4xlarge` | A10G | ~$2.03 | Larger batch, more CPU for data loading |
| `ml.p3.2xlarge` | V100 (16GB) | ~$3.83 | When A10G's 24GB isn't enough |

CPU-only processing jobs (ETL, resize) typically use `ml.m5.*` — size by the task.

## Debug a failed job

CloudWatch log group: `/aws/sagemaker/TrainingJobs` (or `/ProcessingJobs`).

```bash
export AWS_PROFILE=wcs-launcher
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <run-id>/ --follow
```

Or reproduce locally:

```bash
docker run --rm -it \
    -v $(pwd)/sagemaker/configs/example:/opt/ml/input/data/config:ro \
    -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_train_entrypoint.py \
    mermaid-segmentation-jobs:training-smoke-local
```
