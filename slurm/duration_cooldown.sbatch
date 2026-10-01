#!/bin/bash
#SBATCH --job-name=mermaid-durcd
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-durcd-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-durcd-%j.err

# 25k-step linear cooldown from a constant-LR checkpoint down to LR 0.
# Normal qos (not the 14-day reservation). 32 CPUs, default workers/prefetch.
#
# Usage: sbatch [--job-name=...] slurm/duration_cooldown.sbatch SRC_CKPT RUN_NAME
#   SRC_CKPT : checkpoint to initialize from (model + optimizer; scheduler is rebuilt).
#              Relative to nbs/ (where this script cds) or absolute.
#   RUN_NAME : new run name; the cooled model is model_checkpoints/<RUN_NAME>/model_epoch0
# NOTE: do NOT pass these via --export=ALL,VAR=...; on this cluster such jobs are
# cancelled by a site prolog hook ~1 s after start with no logs (CANCELLED by 0).

set -euo pipefail

SRC_CKPT="${1:?usage: sbatch duration_cooldown.sbatch SRC_CKPT RUN_NAME}"
RUN_NAME="${2:?usage: sbatch duration_cooldown.sbatch SRC_CKPT RUN_NAME}"

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
mkdir -p "$TMPDIR" "$SCRATCH/mermaid-segmentation/logs" "$SCRATCH/mermaid-segmentation/wandb"

# ---------- configuration (wandb + AWS-free) ----------
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a

unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION || true

export MERMAIDSEG_S3_OFFLINE=1
export WANDB_DIR=$SCRATCH/mermaid-segmentation/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-mermaidseg}
export WANDB_RUN_GROUP=cbm_duration

: "${WANDB_API_KEY:?set WANDB_API_KEY in $SCRATCH/mermaid-segmentation/.env}"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=${SLURM_JOB_ID}  run=${RUN_NAME}  src=${SRC_CKPT}  gpus=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ---------- run ----------
cd $SCRATCH/mermaid-segmentation/nbs
uv sync --locked

# Data seed 1004 is outside the main run's resume offsets (4..43) so the 25k
# cooldown batches are not a replay of training. Shared across cooldown jobs.
srun uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py \
    --run-name "$RUN_NAME" \
    --training-config ../configs/training_config_cbm_cooldown.yaml \
    --per-pixel-loss-weight 0.5 \
    --per-image-loss-weight 0.5 \
    --init-checkpoint "$SRC_CKPT" \
    --data-seed 1004
