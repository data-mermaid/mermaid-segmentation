#!/bin/bash
#SBATCH --job-name=mermaid-vitsize
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@240
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -G h100:1
#SBATCH --array=0-2
#SBATCH --output=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.out
#SBATCH --error=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.err

# ViT-S / ViT-B / ViT-L on an 80GB-class GPU (default H100). Array indices match
# the CSAIL sweep: 0 vits, 1 vitb, 2 vitl. ViT-H+ is vitsize_train_hplus.sbatch
# so it can request an H200. One epoch of 150k updates, model_step every 25k,
# plus a rolling model_preempt file every 1000 updates.
#
# --requeue restarts this script after preemption. USR1 fires a few minutes
# before the 48h wall and the body submits a continuation; --requeue alone does
# not restart a job that runs out of time. The training script offsets the
# shuffle seed by global_step when the loaded checkpoint is mid-epoch.

set -euo pipefail

REPO="${MERMAIDSEG_REPO:-$HOME/mermaid-segmentation}"
# shellcheck source=env.sh
source "$REPO/slurm/engaging/env.sh"

# Indices match slurm/vitsize_train.sbatch. H+ is not in this array.
SIZES=(vits vitb vitl)
i=${SLURM_ARRAY_TASK_ID}
SIZE=${SIZES[$i]}
RUN="cbm_vitsize_${SIZE}"
EPOCH_ITERS=150000
TOTAL_STEPS=150000
PREEMPT_EVERY=1000
export WANDB_RUN_GROUP=cbm_vitsize_engaging
SBATCH_PATH="$REPO/slurm/engaging/vitsize_train.sbatch"

TRAIN_CMD=(
    uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py
    --run-name "$RUN"
    --training-config ../configs/training_config_cbm_vitsize.yaml
    --model-config "../configs/model_config_cbm_dpt_lora_${SIZE}_256.yaml"
    --per-pixel-loss-weight 0.5
    --per-image-loss-weight 0.5
    --preempt-checkpoint-every-steps "$PREEMPT_EVERY"
)

# shellcheck source=train_body.sh
source "$REPO/slurm/engaging/train_body.sh"
