#!/bin/bash
# =============================================================================
# BASELINE SUBMISSION - READY-TO-RUN COMMAND SHEET
# =============================================================================
# This script is a reference for submitting the LinearDINOv3 baseline.
# DO NOT run this directly - commands are here for copy-paste.
# See .docs/BASELINE_QUICK_START.md for the full walkthrough.
# =============================================================================

# ============================================================================
# STEP 1: PREFLIGHT SETUP (one-time)
# ============================================================================
# Load environment and validate setup

direnv reload

# Check AWS credentials & SageMaker setup
make sm-check

# Expected output:
#   ✓ AWS region: us-east-1
#   ✓ Credentials valid
#   ✓ SageMaker execution role trusted
#   ✓ MLflow tracking URI reachable


# ============================================================================
# STEP 1B: BUILD & PUSH DOCKER IMAGE
# ============================================================================
# Build the training container image and push to ECR

export AWS_PROFILE=wcs-launcher
ACCT=554812291621
IMG=$ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs

# Login to ECR
aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin \
        $ACCT.dkr.ecr.us-east-1.amazonaws.com

# Build image (amd64 for SageMaker)
docker buildx build --platform linux/amd64 \
    -t $IMG:training-latest \
    -f docker/jobs/Dockerfile .

# Push to ECR
docker push $IMG:training-latest

# Verify locally (smoke test)
bash docker/jobs/local_smoke.sh training

# Expected output:
#   ✓ Image built successfully
#   ✓ Pushed to ECR
#   ✓ Local smoke test passed


# ============================================================================
# STEP 2: VALIDATE LOCALLY
# ============================================================================
# Test the training pipeline locally before SageMaker submission

# Option A: Synthetic smoke test (fast, ~5 seconds)
make smoke-standard

# Option B: Real data dry-run (slower, ~40 seconds, requires AWS access)
uv run python scripts/train_baseline.py --dry-run

# Expected output from both:
#   ✓ Dataset loaded (mermaid 15.8k + coralnet 75.7k samples)
#   ✓ Model initialized (LinearDINOv3)
#   ✓ Training completed (epoch 0 metrics)
#   ✓ "Training complete" message


# ============================================================================
# STEP 3: SAGEMAKER DRY-RUN (validation without queuing)
# ============================================================================
# Validate the SageMaker job spec without actually queuing the job

make sm-baseline-dry-run

# Expected output:
#   [launcher] Validating run config: sagemaker/runs/issue_129_dinov3_baseline.yaml
#   [launcher] Image: 554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-latest
#   [launcher] InstanceType: ml.g5.2xlarge
#   ...
#   [launcher] Dry-run OK. Run 'make sm-baseline-launch' to submit.


# ============================================================================
# STEP 4: SUBMIT THE JOB (LIVE)
# ============================================================================
# Actually queue the job to SageMaker

make sm-baseline-launch

# Expected output:
#   [launcher] Submitting TrainingJob: mermaidseg-issue-129-baseline-20260616T123456Z
#   [launcher] Job status: InProgress
#   [launcher] CloudWatch: https://console.aws.amazon.com/cloudwatch/...
#   [launcher] MLflow: https://mlflow.sagemaker.us-east-1.app.aws/#/experiments/2/runs/...
#
#   Job submitted! 🚀


# ============================================================================
# STEP 5: MONITOR THE JOB
# ============================================================================
# Watch the job run

# Option A: Tail CloudWatch logs (real-time)
make sm-baseline-logs

# Option B: Check job status
make sm-baseline-status

# Option C: View in MLflow
# Open the link from Step 4 and look for:
#   Run name: "issue-129-dinov3-baseline-mermaid-coralnet"
#   Experiment: "mermaid"
# Metrics will be logged per epoch

# Expected duration:
#   - Dataset loading & model download: ~5 min
#   - Training (200 epochs): ~3-4 hours
#   - Total: ~3.5-4.5 hours


# ============================================================================
# ESTIMATED COST
# ============================================================================
# Instance:  ml.g5.2xlarge (A10G GPU) @ $1.52/hr
# Duration:  ~4 hours
# Cost:      ~$6 compute + <$1 storage = ~$7-10 per run


# ============================================================================
# SUCCESS CRITERIA
# ============================================================================
# Job is complete when:
#   ✓ CloudWatch logs show "Training complete"
#   ✓ Job status: "Completed"
#   ✓ MLflow run has 200 epochs of metrics
#   ✓ Model artifact in S3: s3://dev-datamermaid-sm-data/runs/<job-id>/output/


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Job stuck in "Downloading" (5-10 min wait is normal)
#   → Check CloudWatch logs: make sm-baseline-logs
#   → Look for "Downloading image" messages

# "Unable to pull image" error
#   → Verify image exists: aws ecr list-images --repository-name mermaid-segmentation-jobs
#   → Rebuild & push: docker push $IMG:training-latest

# NaN loss or training errors
#   → Reproduce locally: uv run python scripts/train_baseline.py ...
#   → Check CloudWatch logs for specific error
#   → See .docs/BASELINE_SUBMISSION.md section 8 for full troubleshooting

# =============================================================================
# REFERENCE
# =============================================================================
# Quick start guide:       .docs/BASELINE_QUICK_START.md (1 page)
# Full guide:              .docs/BASELINE_SUBMISSION.md (11 sections)
# Configuration:           sagemaker/configs/baseline/
# Training script:         scripts/train_baseline.py
# Main training script:    scripts/train.py

# =============================================================================
