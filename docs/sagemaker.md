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

Processing jobs share the same Docker image as training but route to different code
via `--task`. The entrypoint is `scripts/sagemaker_processing_entrypoint.py`.

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_processing.py \
    --run-config sagemaker/runs/<your-run>.yaml \
    --config-dir sagemaker/configs/<your-config-dir>/
```

### Available tasks

| Task | What it does |
|---|---|
| `coralnet-etl` | Full ETL pipeline: audit → build-annotations → build-images, uploads versioned parquets to S3 |
| `coralnet-refresh` | Re-downloads image_list CSVs for a list of source IDs (supports sharding across N workers) |
| `eval` | Model evaluation |
| `inference` | Batch inference |

### Example: CoralNet ETL audit

```bash
export AWS_PROFILE=wcs-launcher
uv run --extra sagemaker python scripts/launch_processing.py \
    --run-config sagemaker/runs/issue_130_coralnet_etl_audit.yaml \
    --config-dir sagemaker/runs/
```

See [`sagemaker/runs/issue_130_coralnet_etl_audit.yaml`](../sagemaker/runs/issue_130_coralnet_etl_audit.yaml)
for the full job spec (ml.m5.4xlarge · 32 workers · 4-hour ceiling).

### CoralNet website behavior: image size & load (read before tuning `coralnet-refresh`)

The `coralnet-refresh` task scrapes the browse listing at
`/source/<id>/browse/images`. CoralNet renders this **server-side**, and its
responsiveness is the dominant constraint on refresh jobs. Hard-won observations
from the issue #130 remediation runs:

- **First, check CoralNet is even up — its availability dominates everything
  else.** CoralNet periodically becomes unresponsive site-wide: TCP connects in
  milliseconds but no HTTP response ever returns. When that happens every job
  dies at login (which is *not* retry-wrapped), regardless of worker count. A
  low-concurrency (2-shard) test during #130 failed at login for exactly this
  reason — confirmed by a local `curl` that also hung. Probe before launching:

  ```bash
  curl -s -o /dev/null \
    -w "http=%{http_code} ttfb=%{time_starttransfer}s total=%{time_total}s\n" \
    --max-time 100 https://coralnet.ucsd.edu/accounts/login/
  ```

  `ttfb=0.000 … total=100s … http=000` means CoralNet is down — wait, don't launch.

- **Page 1 is the bottleneck, and it's load-dependent — not a clean size cutoff.**
  `get_images` must fetch page 1 first (to learn the page size) before it can fan
  out the remaining `?page=N` requests in parallel. If that first GET exceeds the
  client read timeout (`_PAGE_GET_TIMEOUT`, 90 s), the **entire source fails**
  before the fan-out even starts. Empirically a 30 k-image source succeeded while
  a 29 k one failed, and a 9 k failed while a 16 k succeeded — page-1 render time
  varies with CoralNet's current load, not just the source's image count.
- **Many parallel shards trigger a thundering herd.** With 7 shards, ~7 heavy
  browse requests hit CoralNet simultaneously at startup. Failures fail fast
  (~6 min/source) in lockstep, so the shards stay synchronized and keep colliding
  on each subsequent source. A refresh batch composed only of previously-failed
  (i.e. already timeout-prone) sources can fail close to 100 % this way.
- **Retries alone don't rescue a slow page 1.** The `_with_retry` wrapper
  (3 attempts; 30 s / 60 s backoff; covers `Timeout`, `ConnectionError`,
  `ChunkedEncodingError`, 5xx) handles *transient* blips, but when page-1 latency
  consistently exceeds 90 s all three attempts hit the same wall — and fixed
  backoff means concurrent shards retry in lockstep and re-collide.

**Guidance for `coralnet-refresh`:**

- Prefer **low concurrency: 2–3 `workers`**, not 7+. Fewer simultaneous browse
  requests is the single biggest lever against page-1 timeouts.
- Budget runtime by source size: per-source ≈ `pages × ~5 s` (20 images/page,
  `--browse-workers=10`), so ~79 k images ≈ 5.5 h. Set `max_runtime_hours`
  accordingly (12 h is a safe ceiling for the largest sources) and keep each huge
  source on its own shard.
- Sources **> ~20 k images** are the most timeout-prone; expect to iterate.
- **Shipped robustness:** login is retry-wrapped; retry backoff is jittered;
  `--startup-jitter-seconds` (default 30) desyncs shards; and `main()` runs a
  pre-flight `is_coralnet_reachable()` probe that **exits 2 without scraping when
  CoralNet is down** (exit 2 = "site was down, retry later", not a code bug).
- **Still pending:** a **size-aware page-1 timeout** (scale with
  `total_images_hint`, mirroring `export_prep_timeout_seconds`) for when CoralNet
  is merely slow rather than down.

### Credentials in run YAMLs

Any `${VAR}` in a run YAML `env:` block is expanded from the shell environment at
launch time via `os.path.expandvars()`. Set sensitive values in `.env` (gitignored,
loaded by direnv) and reference them as `${MY_VAR}` — credentials are never committed.

### Sharding

For tasks that fan out over a list of items, set a `shard:` block in the run YAML:

```yaml
processing:
  container_args:
    - --task=coralnet-refresh
  shard:
    items_from: sources.csv   # CSV in --config-dir; must have a header row
    items_column: source_id   # which column to read
    workers: 5                # number of parallel ProcessingJobs
    per_worker_arg: --source-ids  # CLI arg each worker receives its slice as
```

The launcher splits `items_from` across `workers` jobs and submits them in parallel.
Each job's logs land in `/aws/sagemaker/ProcessingJobs` under a stream named
`<job-name>-N/algo-1`.

### Adding new processing tasks

1. Add the task name to `choices=` in `scripts/sagemaker_processing_entrypoint.py`.
2. Add an `elif args.task == "<name>":` block that imports from `mermaidseg/` and
   calls `main(extra)`.
3. If the implementation lives under `scripts/` rather than `mermaidseg/`: move it
   to a package path under `mermaidseg/` and register a console script in
   `pyproject.toml` so the in-container import resolves correctly.

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
