#!/bin/bash
#SBATCH --job-name=mermaid-coralnet-report
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-coralnet-report-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-coralnet-report-%j.err

set -euo pipefail

# Static HTML report on every CoralNet source (stats, histograms, 20 overlays each).
#
#   sbatch slurm/coralnet_source_report.sbatch
#   sbatch slurm/coralnet_source_report.sbatch --sources 23 57 --output-dir /tmp/report
#
# Extra arguments are forwarded to scripts/coralnet_source_report.py. To view on a laptop:
#
#   rsync -av <cluster>:/data/vision/beery/scratch/sauder/mermaid-segmentation/coralnet_source_report ./
#   open coralnet_source_report/index.html

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

# Not offline: the warmed cache only holds whitelisted sources, so images for
# every other source are fetched from S3. The script never writes to the cache.
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

OUT=$SCRATCH/mermaid-segmentation/coralnet_source_report

cd $SCRATCH/mermaid-segmentation
uv sync --locked
srun uv run python scripts/coralnet_source_report.py \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --output-dir "$OUT" \
    "$@"
echo "[batch] report written to $OUT (later --output-dir arguments override this)"
