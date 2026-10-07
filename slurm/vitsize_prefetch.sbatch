#!/bin/bash
#SBATCH --job-name=mermaid-vitsize-prefetch
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/vitsize-prefetch-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/vitsize-prefetch-%j.err

set -euo pipefail

# Cache the four DINOv3 encoders (and the training image cache) so the ViT-size
# jobs can run with MERMAIDSEG_S3_OFFLINE=1. ViT-L may already be cached; S/B/H+
# will not be. Safe to resubmit.

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

# ---------- credentials ----------
# This is the ONLY job in the sweep that needs AWS. It warms the local cache so
# training can run fully offline. It reads S3 with the long-lived IAM keys.
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a
source "$HOME/.aws/mermaid-training.env"

# Prefetch must be able to reach S3 — never run it in offline mode.
unset MERMAIDSEG_S3_OFFLINE || true

# ---------- diagnostics ----------
echo "[prefetch] node=$(hostname)  job=$SLURM_JOB_ID"
aws sts get-caller-identity \
    || { echo "AWS credentials invalid — check $HOME/.aws/mermaid-training.env (perms 600, keys not rotated/disabled)"; exit 2; }

# ---------- run ----------
cd $SCRATCH/mermaid-segmentation
uv sync --locked

srun uv run python scripts/prefetch_cache.py \
    --data-config configs/data_config_all.yaml \
    --training-config configs/training_config_cbm_vitsize.yaml \
    --workers 32 \
    --hf-models \
        facebook/dinov3-vits16-pretrain-lvd1689m \
        facebook/dinov3-vitb16-pretrain-lvd1689m \
        facebook/dinov3-vitl16-pretrain-lvd1689m \
        facebook/dinov3-vith16plus-pretrain-lvd1689m
