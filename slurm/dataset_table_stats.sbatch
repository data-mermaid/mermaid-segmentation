#!/bin/bash
#SBATCH --job-name=mermaid-dataset-stats
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-dataset-stats-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-dataset-stats-%j.err

set -euo pipefail

# CPU-only dataset-table counts on the warmed training cache.
# The L40S partition is requested with zero GPUs.
#
#   sbatch slurm/dataset_table_stats.sbatch

# ---------- relocate $HOME so nothing touches the real home dir ----------
export SCRATCH=/data/vision/beery/scratch/sauder
export HOME=$SCRATCH/home
export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
export XDG_DATA_HOME=$HOME/.local/share
export UV_CACHE_DIR=$SCRATCH/uv-cache
export UV_PYTHON_INSTALL_DIR=$SCRATCH/uv-python
export UV_LINK_MODE=copy
export HF_HOME=$SCRATCH/hf-cache
export PIP_CACHE_DIR=$SCRATCH/pip-cache
export TMPDIR=$SCRATCH/tmp
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$TMPDIR" "$HOME/.aws" "$SCRATCH/mermaid-segmentation/logs"

# Read the cache the train jobs already warmed. A miss must fail, not download.
export MERMAIDSEG_S3_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ---------- credentials ----------
# Dataset constructors still build a boto3 client. Keys are sourced so that
# client setup matches the training jobs; offline mode never calls S3 on a hit.
# 1. Non-secret repo defaults (MLFLOW_TRACKING_URI, HF_TOKEN, bucket overrides, ...)
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a
# 2. Long-lived IAM keys for batch jobs.
#    This file sets AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
#    and ends with `unset AWS_PROFILE` so SSO is never consulted.
#    Must be sourced AFTER .env so it overrides any AWS_PROFILE in there.
source "$HOME/.aws/mermaid-training.env"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK  gpus=${CUDA_VISIBLE_DEVICES:-none}"
aws sts get-caller-identity \
    || { echo "AWS credentials invalid — check $HOME/.aws/mermaid-training.env (perms 600, keys not rotated/disabled)"; exit 2; }

cd $SCRATCH/mermaid-segmentation
uv sync --locked
srun uv run python scripts/dataset_table_stats.py --workers "${SLURM_CPUS_PER_TASK:-16}"
echo "[batch] dataset table written to dataset_stats/dataset_table.csv"
