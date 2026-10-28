#!/bin/bash
#SBATCH --job-name=mermaid-vitsize-hplus
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@240
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -G h200:1
#SBATCH --output=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.out
#SBATCH --error=/home/%u/orcd/scratch/mermaidseg/logs/%x-%j.err

# ViT-H+ on an H200 (140GB). Same 150k-update schedule as the other sizes,
# batch 64 and no gradient accumulation. CSAIL array index 3. Preempt
# checkpoints are model_preempt*; evals still use model_step every 25k.

set -euo pipefail

REPO="${MERMAIDSEG_REPO:-$HOME/mermaid-segmentation}"
# shellcheck source=env.sh
source "$REPO/slurm/engaging/env.sh"

SIZE=vithplus
RUN="cbm_vitsize_${SIZE}"
EPOCH_ITERS=150000
TOTAL_STEPS=150000
PREEMPT_EVERY=1000
export WANDB_RUN_GROUP=cbm_vitsize_engaging
SBATCH_PATH="$REPO/slurm/engaging/vitsize_train_hplus.sbatch"

TRAIN_CMD=(
    uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py
    --run-name "$RUN"
    --training-config ../configs/training_config_cbm_vitsize.yaml
    --model-config "../configs/model_config_cbm_dpt_lora_${SIZE}_256.yaml"
    --per-pixel-loss-weight 0.5
    --per-image-loss-weight 0.5
    --batch-size 64
    --preempt-checkpoint-every-steps "$PREEMPT_EVERY"
)

# shellcheck source=train_body.sh
source "$REPO/slurm/engaging/train_body.sh"
