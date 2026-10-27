#!/bin/bash
#SBATCH --job-name=mermaid-hires-eval
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-free
#SBATCH --requeue
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-hires-eval-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-hires-eval-%j.err

set -euo pipefail

# Eval one hi-res fine-tune checkpoint at one prediction resolution.
# Usage: sbatch [--job-name=...] slurm/hires_eval.sbatch CKPT RES OUT [EVAL_ARGS]
#   CKPT      : path to model_step* (relative to the repo root or absolute)
#   RES       : prediction resolution, one of 256 384 512 768 1024
#   OUT       : output directory for this (checkpoint, resolution) eval
#   EVAL_ARGS : optional evaluate.py eval selection (default: --evals all)
# NOTE: do NOT pass these via --export=ALL,VAR=...; on this cluster such jobs are
# cancelled by a site prolog hook ~1 s after start with no logs (CANCELLED by 0).
CKPT="${1:-${CKPT:-}}"
RES="${2:-${RES:-}}"
OUT="${3:-${OUT:-}}"
EVAL_ARGS="${4:---evals all}"
: "${CKPT:?usage: sbatch hires_eval.sbatch CKPT RES OUT}"
: "${RES:?usage: sbatch hires_eval.sbatch CKPT RES OUT}"
: "${OUT:?usage: sbatch hires_eval.sbatch CKPT RES OUT}"

case "$RES" in
    256) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl_256.yaml ;;
    512) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl.yaml ;;
    384|768|1024) MODEL_CONFIG="configs/model_config_cbm_dpt_lora_vitl_${RES}.yaml" ;;
    *) echo "RES must be one of 256 384 512 768 1024, got: $RES" >&2; exit 2 ;;
esac
# Point-eval batch. 512 at batch 8 is the known ViT-L fit on an L40S; larger
# prediction resolutions step the batch down with the token grid.
case "$RES" in
    256|384|512) EVAL_BATCH=8 ;;
    768) EVAL_BATCH=2 ;;
    1024) EVAL_BATCH=1 ;;
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
echo "[batch] node=$(hostname)  job=$SLURM_JOB_ID  gpus=$CUDA_VISIBLE_DEVICES  ckpt=$CKPT  res=$RES  out=$OUT  model=$MODEL_CONFIG  batch=$EVAL_BATCH  evals=$EVAL_ARGS"
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
    --data-config configs/data_config_dedup.yaml \
    --id2label demo/id2label.json \
    --concept-id2name nbs/concept_id2name.json \
    --output-dir "$OUT" \
    $EVAL_ARGS \
    --batch-size "$EVAL_BATCH" \
    --num-workers 16 \
    --amp

# Completion marker: summary.json is written incrementally during the run, so it
# is not a reliable "finished" signal. This file is only created on clean exit
# and is what the launcher checks to avoid re-running a completed eval.
touch "$OUT/.done"
echo "[batch] eval complete -> $OUT/.done"
