#!/bin/bash
#SBATCH --job-name=mermaid-lossw
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=71:59:00
#SBATCH --array=0-6
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-sweep-%A_%a.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-sweep-%A_%a.err

set -euo pipefail

# ---------- loss-weight grid (indexed by SLURM_ARRAY_TASK_ID) ----------
PIX=(1.0 0.9 0.75 0.5 0.25 0.1 0.0)
IMG=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)
i=${SLURM_ARRAY_TASK_ID}
PPW=${PIX[$i]}
PIW=${IMG[$i]}
RUN="cbm_lossw_pix${PPW}_img${PIW}"

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
# Non-secret repo defaults plus WANDB_API_KEY / WANDB_ENTITY live in .env.
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a

# This job needs NO AWS. Data is served from the pre-warmed local cache
# (see slurm/prefetch_cache.sbatch). Actively strip any AWS config so neither
# SSO nor long-lived keys are ever consulted.
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION || true

# Fail loudly on any cache miss instead of silently reaching for S3.
export MERMAIDSEG_S3_OFFLINE=1
export WANDB_DIR=$SCRATCH/mermaid-segmentation/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-mermaidseg}
# Group all 7 sweep runs together in the wandb UI.
export WANDB_RUN_GROUP=cbm_lossw_sweep

# Pre-flight: wandb must be authenticated via the env file.
: "${WANDB_API_KEY:?set WANDB_API_KEY in $SCRATCH/mermaid-segmentation/.env}"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=${SLURM_JOB_ID}  task=${SLURM_ARRAY_TASK_ID}  run=${RUN}  pix=${PPW}  img=${PIW}  gpus=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ---------- run ----------
cd $SCRATCH/mermaid-segmentation/nbs
uv sync --locked                                 # cheap no-op if cache is warm

# ---------- resume from latest checkpoint (if any) ----------
# Checkpoints are written to ./model_checkpoints/<run>/model_epochN. Pick the
# highest-epoch checkpoint so a restarted job continues instead of starting over.
CKPT_DIR="model_checkpoints/${RUN}"
LATEST_CKPT=$(ls -1v "${CKPT_DIR}/model_epoch"* 2>/dev/null | tail -1)
CKPT_ARGS=""
if [ -n "$LATEST_CKPT" ]; then
    echo "[batch] resuming ${RUN} from ${LATEST_CKPT}"
    CKPT_ARGS="--checkpoint ${LATEST_CKPT}"
else
    echo "[batch] no checkpoint found for ${RUN}; starting from scratch"
fi

srun uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py \
    --run-name "$RUN" \
    --training-config ../configs/training_config_cbm_sweep.yaml \
    --per-pixel-loss-weight "$PPW" \
    --per-image-loss-weight "$PIW" \
    $CKPT_ARGS
