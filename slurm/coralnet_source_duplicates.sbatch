#!/bin/bash
#SBATCH --job-name=mermaid-coralnet-dupes
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-coralnet-dupes-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-coralnet-dupes-%j.err

set -euo pipefail

# Same-count visual-hash duplicate check for every CoralNet source.
#
#   sbatch slurm/coralnet_source_duplicates.sbatch
#   sbatch slurm/coralnet_source_duplicates.sbatch --limit-sources 20
#
# Extra arguments are forwarded to scripts/coralnet_source_duplicates.py.
# Re-running the same output directory resumes from hashed images. To copy the
# report back:
#
#   rsync -av <cluster>:/data/vision/beery/scratch/sauder/mermaid-segmentation/coralnet_source_duplicates ./

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

# Not offline: image bytes for sources outside the warmed cache are read from S3.
unset MERMAIDSEG_S3_OFFLINE

# ---------- credentials ----------
# 1. Non-secret repo defaults (MLFLOW_TRACKING_URI, HF_TOKEN, bucket overrides, ...)
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a
# 2. Long-lived IAM keys for batch jobs. Must be sourced AFTER .env so it
#    overrides any AWS_PROFILE in there.
source "$HOME/.aws/mermaid-training.env"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK"
aws sts get-caller-identity \
    || { echo "AWS credentials invalid — check $HOME/.aws/mermaid-training.env (perms 600, keys not rotated/disabled)"; exit 2; }

OUT=$SCRATCH/mermaid-segmentation/coralnet_source_duplicates

cd $SCRATCH/mermaid-segmentation
uv sync --locked
srun uv run python scripts/coralnet_source_duplicates.py \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --output-dir "$OUT" \
    "$@"
echo "[batch] report written to $OUT (later --output-dir arguments override this)"
