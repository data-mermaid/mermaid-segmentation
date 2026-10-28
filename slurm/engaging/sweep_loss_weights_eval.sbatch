#!/bin/bash
#SBATCH --job-name=mermaid-lossw-eval
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -G h100:1
#SBATCH --output=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.out
#SBATCH --error=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.err

# Eval one loss-weight checkpoint at one resolution.
# Usage: sbatch [-G a100:1|h100:1|h200:1] slurm/engaging/sweep_loss_weights_eval.sbatch CKPT RES OUT [EVAL_ARGS]
# A preempted eval is requeued and starts over; .done is removed at start and
# rewritten only after a clean exit.

set -euo pipefail

CKPT="${1:-${CKPT:-}}"
RES="${2:-${RES:-}}"
OUT="${3:-${OUT:-}}"
EVAL_ARGS="${4:---evals all}"
: "${CKPT:?usage: sbatch sweep_loss_weights_eval.sbatch CKPT RES OUT}"
: "${RES:?usage: sbatch sweep_loss_weights_eval.sbatch CKPT RES OUT}"
: "${OUT:?usage: sbatch sweep_loss_weights_eval.sbatch CKPT RES OUT}"

case "$RES" in
    256) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl_256.yaml ;;
    512) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl.yaml ;;
    *) echo "RES must be 256 or 512, got: $RES" >&2; exit 2 ;;
esac
EVAL_BATCH=8

REPO="${MERMAIDSEG_REPO:-$HOME/mermaid-segmentation}"
# shellcheck source=env.sh
source "$REPO/slurm/engaging/env.sh"
# shellcheck source=eval_body.sh
source "$REPO/slurm/engaging/eval_body.sh"
