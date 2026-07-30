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
    --mlflow-tracking-uri $MLFLOW_TRACKING_URI   # from .env
```

The run YAML is the single source of truth: the launcher reads its `job:` block locally and
uploads the same file as the container's `config` channel, so there's no separate config-dir copy
to keep in sync.

**Validate offline before submitting** (no AWS creds needed):

```bash
make sm-validate SM_RUN_CONFIG=sagemaker/runs/<your-run>.yaml
# or: uv run --extra training python -m mermaidseg.experiment validate sagemaker/runs/<your-run>.yaml
```

This schema- and value-checks the whole run — the `job:`/`config:`/`overrides:` blocks **and** the
four referenced split configs (`data`/`model`/`training`/`logger`). It catches a misspelled key, a
missing required field, a bad `optimizer`/`scheduler`/`loss.type` or `model.name`, an invalid
`training_mode`, and mode↔model / mode↔loss mismatches — the failures that otherwise surface only
after a Docker push, inside `MetaModel.__init__`. `--dry-run` runs the same validation as part of
assembling (but un-submitting) a job.

Outputs:
- Run ID and CloudWatch URL printed at submission.
- An MLflow run under the `--run-name` from the YAML's `config.overrides`.
- Final model artifact at `s3://dev-datamermaid-sm-data/runs/<run-id>/output/`.

A run YAML's `config:` block names the model/training/data/logger configs and any `overrides:`
(forwarded to `scripts/train.py` as CLI flags). See `sagemaker/runs/example-training.yaml` for the
shape.

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

| Instance | GPU | vCPU | Host RAM | rec. `num_workers` | $/hr | Use case |
|---|---|---|---|---|---|---|
| `ml.g5.xlarge` | A10G (24GB) | 4 | 16 GiB | ≤3 | ~$1.41 | Eval / inference; small training runs |
| `ml.g5.2xlarge` | A10G | 8 | 32 GiB | ≤7 | ~$1.52 | Standard training run (DINOv3-base, batch 4-8) |
| `ml.g5.4xlarge` | A10G | 16 | 64 GiB | ≤15 | ~$2.03 | Larger batch, more CPU for data loading |
| `ml.g6.2xlarge` | L4 (24GB) | 8 | 32 GiB | ≤7 | ~$1.20 | LoRA / frozen-encoder runs (cheaper L4) |
| `ml.g6.4xlarge` | L4 (24GB) | 16 | 64 GiB | ≤15 | ~$1.74 | Same, more host RAM / CPU for data loading |
| `ml.p3.2xlarge` | V100 (16GB) | 8 | 61 GiB | ≤7 | ~$3.83 | When A10G's 24GB isn't enough |

CPU-only processing jobs (ETL, resize) typically use `ml.m5.*` — size by the task.

**DataLoader worker sizing.** Keep `num_workers ≤ vCPU − 1` (leave a core for the main process).
`persistent_workers` defaults **on** (a throughput win) and is safe with the numpy-native dataset
hot path; `Experiment.validate` emits an advisory if you pair it with `num_workers ≥ vCPU` on a
≤32 GiB instance. If host RAM is ever tight, set `persistent-workers: false` in the run YAML
`overrides:` (reclaims worker RAM every epoch) or move to a `4xlarge`.

## Debug a failed job

The Makefile wraps the common CloudWatch/SageMaker queries. All are read-only and take
`JOB=<run-id>` (the training-job name printed at launch). They use `SM_AWS_PROFILE` when set
(as the launch targets do), otherwise they inherit your ambient AWS credentials.

```bash
make sm-jobs                    # 10 most recent jobs + status
make sm-status  JOB=<run-id>    # status + FailureReason (the container exit code)
make sm-errors  JOB=<run-id>    # grep logs for tracebacks / OOM / disk-full / signal kills
make sm-logs    JOB=<run-id>    # tail -f the CloudWatch stream
make sm-metrics JOB=<run-id>    # disk / RAM / GPU-mem / CPU over the job's lifetime
```

CloudWatch log group: `/aws/sagemaker/TrainingJobs` (or `/ProcessingJobs`). The raw equivalents:

```bash
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <run-id>/ --follow
```

### Diagnosing a *silent* death

A job whose logs just **stop** (no Python traceback) and whose `FailureReason` is a bare
`AlgorithmError: exit code 1` was almost always killed *outside* Python — an OS OOM-kill, a
full disk, or a native (C-extension) crash — so the traceback never flushed to CloudWatch.
`make sm-metrics` is the fastest triage; the `Host` dimension is `<run-id>/algo-1`:

| Metric | Reads ~100% before death → | Notes |
|---|---|---|
| `MemoryUtilization` | **system-RAM OOM** — a worker was OOM-killed | % of instance RAM. A spike between the 1-min samples can be missed — narrow `--period` to 60. |
| `DiskUtilization` | **volume full** — checkpoints/cache filled `volume_gb` | % of the EBS ML volume. Steady ~0 means nothing is writing there (e.g. an image cache that never populated). |
| `GPUMemoryUtilization` | **CUDA OOM** — but this usually *does* raise a Python traceback | |
| all flat, GPU-mem pinned | **hang/deadlock** (DataLoader worker, CUDA sync) killed later | GPU memory staying allocated with zero log progress is the tell. |

For finer resolution than the Makefile's 5-min buckets, query one metric directly with
`--period 60`:

```bash
aws cloudwatch get-metric-statistics --namespace /aws/sagemaker/TrainingJobs \
    --metric-name MemoryUtilization --dimensions Name=Host,Value=<run-id>/algo-1 \
    --start-time <ISO8601> --end-time <ISO8601> --period 60 --statistics Maximum \
    --query 'Datapoints|sort_by(@,&Timestamp)[].[Timestamp,Maximum]' --output text
```

> Do not `describe-training-job` without a narrow `--query`: the full response includes the
> job's `Environment`, which carries `HF_TOKEN`. The `make sm-*` targets already scope their
> queries to avoid printing it.

### Known gotchas (data-loading path)

- **A silent `exit code 1` with no traceback is usually a native crash** (segfault in
  libjpeg/PIL/boto3/CUDA), which Python can't print by default. The container sets
  `PYTHONUNBUFFERED=1` but **not** `PYTHONFAULTHANDLER` — enabling faulthandler is the way to get
  a C-level stack into CloudWatch for the next occurrence. Until then, silent deaths can only be
  triaged by `make sm-metrics` (see the table above), not by reading logs.
- **Dataset S3 clients are fork/pickle-unsafe.** Each dataset holds `self.s3 = boto3.client(...)`
  created before the DataLoader forks its workers. On Linux (SageMaker) `fork` tolerates this;
  running the same code locally on macOS (default `spawn`) with `num_workers>0` fails immediately
  with `PicklingError: Can't pickle botocore.client.S3`. Keep local runs at `num_workers=0` unless
  the client is made per-process (lazy property + dropped in `__getstate__`).
- **Persistent DataLoader workers + host-RAM growth (the dinov3-lora-qv-r8 OOM).** Under Linux
  `fork`, a persistent worker that touched pandas object columns / str-keyed dicts every sample
  copy-on-write-privatized their pages and never reclaimed them — host RAM climbed ~4 GB/day to a
  32 GiB OOM at epoch 76 while GPU stayed flat. The dataset hot path is now numpy-native
  (`BaseCoralDataset._build_annotation_index`), which keeps forked workers CoW-clean, so
  `persistent_workers=True` is safe. To diagnose any recurrence, launch with
  `MERMAIDSEG_LOG_WORKER_RSS=1` (logs per-worker RSS) and reproduce/quantify with
  `scripts/diagnostics/dataloader_rss_repro.py`. Full write-up: `docs/investigating-training-runs.md`
  and `scripts/diagnostics/dataloader_rss_findings.md`.

Or reproduce locally:

```bash
docker run --rm -it \
    -v $(pwd)/sagemaker/runs/example-training.yaml:/opt/ml/input/data/config/run.yaml:ro \
    -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_train_entrypoint.py \
    mermaid-segmentation-jobs:training-smoke-local
```

(Mount the one run YAML you want to test, not the whole `sagemaker/runs/` directory — the
entrypoint expects exactly one file with a `job:` block under `config/` and errors on more than one.)
