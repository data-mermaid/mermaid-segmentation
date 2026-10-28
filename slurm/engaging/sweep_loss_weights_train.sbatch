#!/bin/bash
#SBATCH --job-name=mermaid-lossw
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
#SBATCH --array=0-6
#SBATCH --output=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.out
#SBATCH --error=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.err

# Loss-weight sweep. Array order matches slurm/sweep_loss_weights_train.sbatch.
# 10 epochs x 15000 updates. model_epoch* stays the eval cadence; model_preempt*
# is the rolling preemption checkpoint and is not evaluated.
# Default model config is the training script's ViT-L config, same as CSAIL.

set -euo pipefail

REPO="${MERMAIDSEG_REPO:-$HOME/mermaid-segmentation}"
# shellcheck source=env.sh
source "$REPO/slurm/engaging/env.sh"

PIX=(1.0 0.9 0.75 0.5 0.25 0.1 0.0)
IMG=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)
i=${SLURM_ARRAY_TASK_ID}
PPW=${PIX[$i]}
PIW=${IMG[$i]}
RUN="cbm_lossw_pix${PPW}_img${PIW}"
EPOCH_ITERS=15000
TOTAL_STEPS=150000
PREEMPT_EVERY=1000
export WANDB_RUN_GROUP=cbm_lossw_engaging
SBATCH_PATH="$REPO/slurm/engaging/sweep_loss_weights_train.sbatch"

TRAIN_CMD=(
    uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py
    --run-name "$RUN"
    --training-config ../configs/training_config_cbm_sweep.yaml
    --per-pixel-loss-weight "$PPW"
    --per-image-loss-weight "$PIW"
    --preempt-checkpoint-every-steps "$PREEMPT_EVERY"
)

# shellcheck source=train_body.sh
source "$REPO/slurm/engaging/train_body.sh"
