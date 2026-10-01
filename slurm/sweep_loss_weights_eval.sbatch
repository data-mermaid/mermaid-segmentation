#!/bin/bash
#SBATCH --job-name=mermaid-seg-eval
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=80G
#SBATCH --time=71:59:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-eval-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-eval-%j.err

set -euo pipefail

# ---------- parameters (passed as positional args from the launcher) ----------
# Usage: sbatch [--job-name=...] slurm/sweep_loss_weights_eval.sbatch CKPT RES OUT
#   CKPT : path to the checkpoint (relative to the repo root or absolute)
#   RES  : eval resolution, 256 or 512
#   OUT  : output directory for this (checkpoint, resolution) eval
# NOTE: do NOT pass these via --export=ALL,VAR=...; on this cluster such jobs are
# cancelled by a site prolog hook ~1 s after start with no logs (CANCELLED by 0).
CKPT="${1:-${CKPT:-}}"
RES="${2:-${RES:-}}"
OUT="${3:-${OUT:-}}"
: "${CKPT:?usage: sbatch sweep_loss_weights_eval.sbatch CKPT RES OUT}"
: "${RES:?usage: sbatch sweep_loss_weights_eval.sbatch CKPT RES OUT}"
: "${OUT:?usage: sbatch sweep_loss_weights_eval.sbatch CKPT RES OUT}"

case "$RES" in
    256) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl_256.yaml ;;
    512) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl.yaml ;;
    *) echo "RES must be 256 or 512, got: $RES" >&2; exit 2 ;;
esac

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
# 1. Non-secret repo defaults (MLFLOW_TRACKING_URI, HF_TOKEN, bucket overrides, ...)
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a
# 2. Long-lived IAM keys for batch jobs.
#    This file sets AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
#    and ends with `unset AWS_PROFILE` so SSO is never consulted.
#    Must be sourced AFTER .env so it overrides any AWS_PROFILE in there.
source "$HOME/.aws/mermaid-training.env"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=$SLURM_JOB_ID  gpus=$CUDA_VISIBLE_DEVICES  ckpt=$CKPT  res=$RES  out=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
aws sts get-caller-identity \
    || { echo "AWS credentials invalid — check $HOME/.aws/mermaid-training.env (perms 600, keys not rotated/disabled)"; exit 2; }

cd $SCRATCH/mermaid-segmentation
uv sync --locked
# Drop the previous success marker before python overwrites summary.json.
# It is recreated only after a clean exit, so a failed re-run is not "done".
rm -f "$OUT/.done"
srun uv run python scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --model-config "$MODEL_CONFIG" \
    --data-config configs/data_config_all.yaml \
    --id2label demo/id2label.json \
    --concept-id2name nbs/concept_id2name.json \
    --output-dir "$OUT" \
    --evals all \
    --batch-size 8 \
    --num-workers 16 \
    --amp

# Completion marker: summary.json is written incrementally during the run, so it
# is not a reliable "finished" signal. This file is only created on clean exit
# and is what the launcher checks to avoid re-running a completed eval.
touch "$OUT/.done"
echo "[batch] eval complete -> $OUT/.done"
