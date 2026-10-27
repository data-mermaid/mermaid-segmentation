#!/bin/bash
#SBATCH --job-name=mermaid-dq-build
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-dq-build-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-dq-build-%j.err

set -euo pipefail

# Build one named data-quality profile: its degraded training data and the data
# config that trains on it, checked by loading the degraded train splits back
# before any GPU time is spent (scripts/build_data_quality_profile.py).
#
#   sbatch slurm/data_quality_build.sbatch sparse                                   # -> sparse
#   sbatch slurm/data_quality_build.sbatch phylum --quantiles 0 0 0 0 0 0.1         # -> phylum_0_0_0_0_0_0.1
#   sbatch slurm/data_quality_build.sbatch sparse phylum --quantiles 0 0 0 0 0 0.1  # -> sparse_phylum_0_0_0_0_0_0.1
#
# sparse: Coralscapes V2 train + UCSD mosaics, 25 points per image as 7x7 squares.
# phylum: CoralNet training sources flagged by --quantiles (in order n_images n_pixels
#         n_labels n_genera n_coral_genera avg_pixels_per_image) keep phylum-level labels.
#
# The degraded data are shared between profiles under $DQ_ROOT/components and only
# built when missing, so the combined profile above reuses the other two. The
# profile lands in $DQ_ROOT/profiles/<name>; train it with
# slurm/data_quality_launch_train.sh <name>. All arguments go to the build script
# (e.g. --overwrite to rewrite an existing profile, --stats-csv for phylum).

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch slurm/data_quality_build.sbatch {sparse|phylum --quantiles Q*6|sparse phylum --quantiles Q*6} [args...]" >&2
    exit 2
fi

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
# phylum and the check read data local-cache-first and fall back to S3.
# 1. Non-secret repo defaults (HF_TOKEN, bucket overrides, ...)
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a
# 2. Long-lived IAM keys for batch jobs. Must be sourced AFTER .env so it
#    overrides any AWS_PROFILE in there.
source "$HOME/.aws/mermaid-training.env"
unset MERMAIDSEG_S3_OFFLINE || true

DQ_ROOT=$SCRATCH/mermaid-segmentation/data_quality
mkdir -p "$DQ_ROOT"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=$SLURM_JOB_ID  args=$*  cpus=$SLURM_CPUS_PER_TASK  out=$DQ_ROOT"
aws sts get-caller-identity \
    || { echo "AWS credentials invalid — check $HOME/.aws/mermaid-training.env (perms 600, keys not rotated/disabled)"; exit 2; }

cd $SCRATCH/mermaid-segmentation
uv sync --locked

srun uv run python scripts/build_data_quality_profile.py \
    --dq-root "$DQ_ROOT" \
    --num-proc "${SLURM_CPUS_PER_TASK:-32}" \
    "$@"
