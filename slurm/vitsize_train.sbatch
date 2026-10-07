#!/bin/bash
#SBATCH --job-name=mermaid-vitsize
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=71:59:00
#SBATCH --array=0-3
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-vitsize-%A_%a.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-vitsize-%A_%a.err

# ViT-size sweep: one 150k-update epoch (4k warmup, then linear decay to 0).
# Array order MUST match scripts/compile_vitsize_evals.py VIT_SIZES.
#   0 vits, 1 vitb, 2 vitl: batch 32
#   3 vithplus: microbatch 16 x 2 grad-accum steps (effective batch 32)
#
# Checkpoints are local only: nbs/model_checkpoints/cbm_vitsize_<size>/model_step*
# every 25k optimizer updates. Resubmit this file to continue a timed-out task
# (sbatch slurm/vitsize_train.sbatch, or --array=3 for H+ only). A finished run
# (model_step150000, epoch complete) loads that checkpoint and exits 0.

set -euo pipefail

# ---------- ViT grid (indexed by SLURM_ARRAY_TASK_ID) ----------
SIZES=(vits vitb vitl vithplus)
i=${SLURM_ARRAY_TASK_ID}
SIZE=${SIZES[$i]}
RUN="cbm_vitsize_${SIZE}"
MODEL_CONFIG="../configs/model_config_cbm_dpt_lora_${SIZE}_256.yaml"

# 32 CPUs, 28 workers, leave the rest for the main process and pin-memory thread.
export MERMAIDSEG_NUM_WORKERS=28

EXTRA=()
if [[ "$SIZE" == "vithplus" ]]; then
    EXTRA+=(--batch-size 16 --grad-accum-steps 2)
fi

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
export WANDB_RUN_GROUP=cbm_vitsize

# Pre-flight: wandb must be authenticated via the env file.
: "${WANDB_API_KEY:?set WANDB_API_KEY in $SCRATCH/mermaid-segmentation/.env}"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=${SLURM_JOB_ID}  task=${SLURM_ARRAY_TASK_ID}  run=${RUN}  size=${SIZE}  gpus=${CUDA_VISIBLE_DEVICES}  workers=${MERMAIDSEG_NUM_WORKERS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ---------- run ----------
cd $SCRATCH/mermaid-segmentation/nbs
uv sync --locked                                 # cheap no-op if cache is warm

# ---------- resume from latest step checkpoint (if any) ----------
# model_step{N} is the checkpoint after N optimizer updates. Pick the highest
# so a restarted job continues instead of starting over. A completed run's
# model_step150000 has epoch_complete set, so training does not start again.
CKPT_DIR="model_checkpoints/${RUN}"
shopt -s nullglob
ckpts=("${CKPT_DIR}"/model_step*)
shopt -u nullglob
CKPT_ARGS=()
if ((${#ckpts[@]})); then
    LATEST_CKPT=$(printf '%s\n' "${ckpts[@]}" | sort -V | tail -1)
    echo "[batch] resuming ${RUN} from ${LATEST_CKPT}"
    CKPT_ARGS=(--checkpoint "${LATEST_CKPT}")
else
    echo "[batch] no checkpoint found for ${RUN}; starting from scratch"
fi

srun uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py \
    --run-name "$RUN" \
    --training-config ../configs/training_config_cbm_vitsize.yaml \
    --model-config "$MODEL_CONFIG" \
    --per-pixel-loss-weight 0.5 \
    --per-image-loss-weight 0.5 \
    "${EXTRA[@]}" \
    "${CKPT_ARGS[@]}"
