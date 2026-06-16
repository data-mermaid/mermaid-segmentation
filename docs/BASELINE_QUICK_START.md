# Baseline Submission — Quick Start

**Submit the LinearDINOv3 baseline (mermaid + coralnet) to SageMaker in 3 steps.**

---

## Step 1: Preflight (Do this once)

```bash
# Load environment
direnv reload

# Validate setup
make sm-check

# Build Docker image
docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) \
    554812291621.dkr.ecr.us-east-1.amazonaws.com

docker buildx build --platform linux/amd64 \
    -t 554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-latest \
    -f docker/jobs/Dockerfile .

docker push 554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-latest

# Verify locally
bash docker/jobs/local_smoke.sh training
```

---

## Step 2: Validate Locally

```bash
# Quick smoke test (synthetic data, ~5s)
make smoke-standard

# Real data dry-run (1 epoch, 1 batch, ~40s)
uv run python scripts/train_baseline.py --dry-run

# Expected: "Training complete" ✓
```

---

## Step 3: Submit to SageMaker

```bash
# Dry-run (validates config, doesn't queue job)
make sm-baseline-dry-run

# Expected output:
#   [launcher] TrainingJob spec: mermaidseg-issue-129-baseline-20260616T...
#   [launcher] Dry-run OK. Run 'make sm-baseline-launch' to submit.

# Launch!
make sm-baseline-launch

# Expected output:
#   [launcher] Job submitted: mermaidseg-issue-129-baseline-20260616T123456Z
#   [launcher] CloudWatch: https://console.aws.amazon.com/cloudwatch/...
#   [launcher] MLflow: https://mlflow.sagemaker.us-east-1.app.aws/#/...
```

---

## Step 4: Monitor

```bash
# Watch CloudWatch logs in real-time
make sm-baseline-logs

# Or check status
make sm-baseline-status

# Or view in MLflow
# Open the link from Step 3 → look for "issue-129-dinov3-baseline-mermaid-coralnet"
```

---

## What's Running

| Component | Value |
|-----------|-------|
| **Datasets** | mermaid (15.8k train) + coralnet (52.7k train, 22.9k val) |
| **Model** | LinearDINOv3 with ViT-B/16 backbone |
| **Training mode** | Standard (no concept bottleneck) |
| **Epochs** | 200 |
| **Batch size** | 8 |
| **Instance** | ml.g5.2xlarge (A10G GPU, ~$1.52/hr) |
| **Runtime limit** | 8 hours |
| **Estimated duration** | 3-4 hours (depends on GPU utilization) |

---

## Estimated Cost

- **Instance**: ml.g5.2xlarge @ $1.52/hr × 4 hrs = **~$6**
- **EBS volume**: 200GB @ $0.10/GB/month (prorated) = **<$1** (shared)
- **Data transfer**: Minimal (S3 to VPC)
- **Total**: **~$7-10 per baseline run**

---

## If Something Goes Wrong

1. **Check CloudWatch logs:**
   ```bash
   make sm-baseline-logs
   ```

2. **Check job status:**
   ```bash
   make sm-baseline-status
   ```

3. **Reproduce locally:**
   ```bash
   uv run python scripts/train_baseline.py --config-data ... --config-training ...
   ```

4. **Read the full guide:**
   - [docs/BASELINE_SUBMISSION.md](./BASELINE_SUBMISSION.md) — Full troubleshooting section

---

## Success Criteria

Job is done when:
- ✓ CloudWatch logs show "Training complete"
- ✓ MLflow run logged with 200 epochs of metrics
- ✓ Final model artifact in S3: `s3://dev-datamermaid-sm-data/runs/<job-id>/output/`
- ✓ Training time < 8 hours (job timeout)

---

## Next Steps

After job completes:
1. View metrics in MLflow (mIoU, accuracy, loss curves)
2. Download model artifact from S3
3. Create PR documenting baseline results (config, metrics, duration)
4. (Optional) Launch evaluation job against test set

---

**For more details, see [docs/BASELINE_SUBMISSION.md](./BASELINE_SUBMISSION.md)**
