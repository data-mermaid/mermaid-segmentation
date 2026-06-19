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

### Quick Start: LinearDINOv3 Baseline (Issue #129)

For the mermaid + coralnet baseline (no CBM), see **[BASELINE_SUBMISSION.md](./BASELINE_SUBMISSION.md)** for:
- Pre-configured data/model/training configs
- One-command submission: `make sm-baseline-launch`
- Detailed troubleshooting

### Custom Training Job

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

## Run a processing job

Processing jobs share the same Docker image as training but dispatch on `--task`.
The entrypoint is `scripts/sagemaker_processing_entrypoint.py`.

### CoralNet ETL audit (`--task=coralnet-etl`)

Runs the full ETL pipeline (`audit` → `build-annotations` → `build-images`) and
uploads versioned parquets to `s3://${MERMAID_CORALNET_BUCKET}/${MERMAID_CORALNET_OUTPUT_PREFIX}/<date>_<git-sha>/`.

Run YAML: [`sagemaker/runs/issue_130_coralnet_etl_audit.yaml`](../sagemaker/runs/issue_130_coralnet_etl_audit.yaml)
(ml.m5.4xlarge · 32 workers · 4-hour ceiling)

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_processing.py \
    --run-config sagemaker/runs/issue_130_coralnet_etl_audit.yaml \
    --config-dir sagemaker/runs/
```

After the job completes, the audit parquet is at:

```
s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/<version>/coralnet_audit_<version>.parquet
```

Download and inspect `is_complete` and `image_list_covers_annotations` counts to
decide whether a refresh run is needed.

### CoralNet image-list refresh (`--task=coralnet-refresh`)

Re-downloads truncated `image_list.csv` files for sources where
`image_list_covers_annotations=False` in the audit parquet (fix for
[#130](https://github.com/data-mermaid/mermaid-segmentation/issues/130)).

**Before launching:**

1. Extract the flagged source IDs from the audit parquet and write them (one per
   line, column header `source_id`) to
   `sagemaker/configs/coralnet-refresh/coralnet_refresh_sources.csv`.
2. Ensure `CORALNET_USERNAME` and `CORALNET_PASSWORD` are set in `.env` (loaded by
   direnv). The YAML references `${CORALNET_USERNAME}` / `${CORALNET_PASSWORD}`;
   `launcher_config.py` expands these via `os.path.expandvars()` at launch time —
   credentials are never committed.

Run YAML: [`sagemaker/runs/issue_130_coralnet_refresh.yaml`](../sagemaker/runs/issue_130_coralnet_refresh.yaml)
(ml.m5.xlarge · 5 parallel shards · 3-hour ceiling per shard)

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_processing.py \
    --run-config sagemaker/runs/issue_130_coralnet_refresh.yaml \
    --config-dir sagemaker/configs/coralnet-refresh/
```

The launcher reads `coralnet_refresh_sources.csv`, splits the source IDs across 5
workers, and submits 5 parallel ProcessingJobs. Each job receives a
`--source-ids=id1,id2,...` slice and logs independently.

**Monitoring:** CloudWatch log group `/aws/sagemaker/ProcessingJobs`, one stream per
shard — `<job-name>-N/algo-1` (where N is the zero-based shard index).

```bash
# Tail all 5 streams (replace <prefix> with the job name prefix printed at launch):
for i in 0 1 2 3 4; do
  aws logs tail /aws/sagemaker/ProcessingJobs \
      --log-stream-name-prefix "<prefix>-${i}/" --follow &
done
wait
```

### Adding new processing tasks

1. Add the task name to `choices=` in `scripts/sagemaker_processing_entrypoint.py`.
2. Add an `elif args.task == "<name>":` block that imports from `mermaidseg/` and
   calls `main(extra)`.
3. If the implementation currently lives under `scripts/` rather than `mermaidseg/`:
   move it to a package path under `mermaidseg/`, then register it as a console
   script in `pyproject.toml` so the in-container import resolves correctly.

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
