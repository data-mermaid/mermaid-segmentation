# Submitting the Issue #129 Baseline to SageMaker

This guide walks you through submitting the LinearDINOv3 baseline (mermaid + coralnet, no CBM) to SageMaker for training.

**TL;DR:**
```bash
# Check environment
make sm-check

# Dry-run (validation)
make sm-baseline-dry-run

# Launch job
make sm-baseline-launch

# Watch it
make sm-baseline-logs
```

---

## 1. Preflight Setup

### Required AWS Credentials
Ensure you have the `wcs-launcher` profile configured for SageMaker launcher access.

**Check it works:**
```bash
make sm-check
```

Output should show:
- ✓ SageMaker SDK version
- ✓ AWS region (us-east-1)
- ✓ Boto3/credentials valid
- ✓ IAM role trust for sagemaker.amazonaws.com
- ✓ MLflow tracking URI

### Required Environment Variables (in `.env`)
```bash
# Example .env (replace with your values)
SM_ROLE_ARN=arn:aws:iam::554812291621:role/wcs-sagemaker-launcher-role
MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2
SM_AWS_PROFILE=wcs-launcher
AWS_DEFAULT_REGION=us-east-1
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx  # Optional, for HuggingFace access
```

Load with direnv:
```bash
direnv reload
```

---

## 2. Build & Push the Docker Image

The baseline runs inside a Docker container. Build it once, push it to ECR:

```bash
# Login to ECR
export AWS_PROFILE=wcs-launcher
ACCT=554812291621
docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) \
    $ACCT.dkr.ecr.us-east-1.amazonaws.com

# Build
docker buildx build --platform linux/amd64 \
    -t $ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-latest \
    -f docker/jobs/Dockerfile .

# Push
docker push $ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-latest

# Verify locally (smoke test)
bash docker/jobs/local_smoke.sh training
```

**Or via Makefile:**
```bash
make sm-sync  # Install SageMaker dependencies
```

---

## 3. Baseline Configuration

The baseline is pre-configured with:

| Component | Config |
|-----------|--------|
| **Data** | `configs/data_config_dinov3_base.yaml` — mermaid + coralnet only |
| **Model** | `configs/model_config_dinov3_base.yaml` — LinearDINOv3, ViT-B/16 |
| **Training** | `configs/training_config_dinov3_base.yaml` — 200 epochs, batch=8, standard mode (no CBM) |
| **Logger** | `configs/logger_config.yaml` — MLflow tracking |

**Job config:** `sagemaker/configs/baseline/run.yaml`
- Instance: `ml.g5.2xlarge` (A10G GPU, ~$1.52/hr)
- Volume: 200GB (for model & data cache)
- Runtime limit: 8 hours
- Entrypoint: `scripts/sagemaker_train_entrypoint.py`

### Key Parameters
```yaml
config:
  config_model: configs/model_config_dinov3_base.yaml
  config_training: configs/training_config_dinov3_base.yaml
  config_data: configs/data_config_dinov3_base.yaml
  config_logger: configs/logger_config.yaml
  overrides:
    run-name: issue-129-dinov3-baseline-mermaid-coralnet
    epochs: 200
    batch-size: 8
    seed: 42
```

---

## 4. Validate Locally (Dry-Run)

Before launching to SageMaker, validate the pipeline locally:

```bash
# Synthetic smoke test (mocked datasets, ~5s)
make smoke-standard

# Real data dry-run (1 epoch, 1 batch, ~40s, requires AWS creds)
uv run python scripts/train_baseline.py --dry-run
```

Both should complete **without errors**. You'll see:
- Dataset summary (15.8k mermaid + 52.7k coralnet train samples)
- Epoch 0 metrics (train/val loss & accuracy)
- Checkpoint saved locally
- MLflow run logged

---

## 5. SageMaker Dry-Run (Validation Job)

Simulate the SageMaker job without actually queuing it:

```bash
make sm-baseline-dry-run
```

This:
- ✓ Validates launcher config syntax
- ✓ Checks AWS credentials & IAM role
- ✓ Validates Docker image exists in ECR
- ✓ Prints the job spec it **would** submit
- ✗ Does NOT queue the job

**Expected output:**
```
[launcher] Validating run config: sagemaker/runs/issue_129_dinov3_baseline.yaml
[launcher] Config dir: sagemaker/configs/baseline/
[launcher] Image: 554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-latest
[launcher] IAM role: arn:aws:iam::554812291621:role/wcs-sagemaker-launcher-role
[launcher] MLflow tracking: arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2

TrainingJob spec:
  Name: mermaidseg-issue-129-baseline-20260616T123456Z
  Image: ...training-latest
  InstanceType: ml.g5.2xlarge
  VolumeSizeInGB: 200
  ... (full spec)

[launcher] Dry-run OK. Run 'make sm-baseline-launch' to submit.
```

---

## 6. Launch the Baseline Job

Once you're confident, submit the job:

```bash
make sm-baseline-launch
```

This:
- ✓ Queues the job to SageMaker
- ✓ Prints the job name (e.g., `mermaidseg-issue-129-baseline-20260616T123456Z`)
- ✓ Prints CloudWatch logs URL
- ✓ Prints MLflow run link

**Expected output:**
```
[launcher] Submitting TrainingJob: mermaidseg-issue-129-baseline-20260616T123456Z
[launcher] Job status: InProgress
[launcher] CloudWatch: https://console.aws.amazon.com/cloudwatch/...
[launcher] MLflow: https://mlflow.sagemaker.us-east-1.app.aws/#/experiments/2/runs/...

Job submitted. Check status with:
  make sm-baseline-logs JOB_ID=mermaidseg-issue-129-baseline-20260616T123456Z
```

---

## 7. Monitor the Job

### Watch CloudWatch Logs (Real-Time)
```bash
make sm-baseline-logs
```

Or manually:
```bash
export AWS_PROFILE=wcs-launcher
JOB_ID=mermaidseg-issue-129-baseline-20260616T123456Z
aws logs tail /aws/sagemaker/TrainingJobs \
    --log-stream-name-prefix $JOB_ID/ \
    --follow
```

### Check Job Status
```bash
export AWS_PROFILE=wcs-launcher
aws sagemaker describe-training-job \
    --training-job-name mermaidseg-issue-129-baseline-20260616T123456Z
```

### View in MLflow
Open the MLflow App URL (printed on job submission):
```
https://mlflow.sagemaker.us-east-1.app.aws/#/experiments/2
```

Look for run: `issue-129-dinov3-baseline-mermaid-coralnet`
- Epoch metrics logged per epoch
- Model checkpoint artifact links
- Config versioning

---

## 8. Troubleshooting

### Job stuck in "Downloading" state
The instance is downloading the Docker image and datasets. Check CloudWatch logs; this can take 5-10 minutes on first run.

### "Unable to pull image" error
The Docker image doesn't exist in ECR or the ARN is wrong. Verify:
```bash
export AWS_PROFILE=wcs-launcher
aws ecr describe-repositories --repository-names mermaid-segmentation-jobs
aws ecr list-images --repository-name mermaid-segmentation-jobs
```

### "Access Denied: IAM role not trusted"
The SageMaker execution role doesn't trust `sagemaker.amazonaws.com`. Check:
```bash
export AWS_PROFILE=wcs-launcher
aws iam get-role --role-name wcs-sagemaker-launcher-role
```

Trust policy should include:
```json
{
  "Effect": "Allow",
  "Principal": {"Service": "sagemaker.amazonaws.com"},
  "Action": "sts:AssumeRole"
}
```

### NaN loss or crashes during training
Check CloudWatch logs for:
- Concept schema assertion errors (should not happen — ConceptSchema is skipped in standard mode)
- Out-of-memory errors (batch size too large for GPU)
- Dataset loading errors (check S3 paths in data_config.yaml)

Reproduce locally:
```bash
uv run python scripts/train_baseline.py --config-data ... --config-training ...
```

---

## 9. Makefile Commands Summary

All available targets for baseline work:

```bash
# Local validation
make smoke-standard              # Synthetic smoke test (~5s)
uv run python scripts/train_baseline.py --dry-run  # Real dry-run (~40s)

# SageMaker submission
make sm-check                    # Validate environment
make sm-baseline-dry-run         # Dry-run (no job queued)
make sm-baseline-launch          # Submit job
make sm-baseline-logs            # Watch CloudWatch logs
make sm-baseline-status          # Check job status

# Docker
docker buildx build ... -f docker/jobs/Dockerfile .  # Build image
docker push ...training-latest                       # Push to ECR
bash docker/jobs/local_smoke.sh training            # Local smoke test
```

---

## 10. What's Happening Behind the Scenes

**Local execution flow:**
```
scripts/train_baseline.py
  ├── Load baseline configs (data/model/training)
  ├── Load mermaid + coralnet datasets (no other datasets)
  ├── Build SourceLabelRegistry (compute_concepts=False, no CBM)
  ├── Initialize MetaModel (LinearDINOv3, no concept bottleneck)
  ├── Log to MLflow (remote: SageMaker MLflow App)
  ├── Train for 200 epochs, batch_size=8
  ├── Save checkpoints locally
  └── Return final metrics
```

**SageMaker execution flow:**
```
sagemaker_train_entrypoint.py (in Docker container)
  ├── Download Docker image
  ├── Mount config dir: sagemaker/configs/baseline/ → /opt/ml/input/data/config/
  ├── Run: python -u scripts/train.py (or train_baseline.py)
  │   └── (same as local execution)
  ├── Upload model artifacts → S3: dev-datamermaid-sm-data/runs/<job-id>/output/
  └── Log to CloudWatch + MLflow
```

---

## 11. Next Steps After Job Completes

Once the job finishes (InProgress → Completed or Failed):

### If successful:
1. **Check metrics** in MLflow (mIoU, accuracy, loss over 200 epochs)
2. **Download model** from S3: `s3://dev-datamermaid-sm-data/runs/<job-id>/output/`
3. **Launch evaluation job** against test set (separate processing job)
4. **Create PR** documenting baseline results

### If failed:
1. **Check CloudWatch logs** for error messages
2. **Inspect config** (data paths, GPU memory, batch size)
3. **Reproduce locally** with same config
4. **File issue** with logs + config + local repro steps

---

## Reference

- **Issue:** #129 (LinearDINOv3 baseline, no CBM)
- **SageMaker launcher convention:** [mermaid-api/iac/sagemaker-launcher-convention.md](../../mermaid-api/iac/sagemaker-launcher-convention.md)
- **Baseline configs:** `sagemaker/configs/baseline/`
- **Local baseline script:** `scripts/train_baseline.py`
- **Training script:** `scripts/train.py` or `scripts/sagemaker_train_entrypoint.py`
