#!/bin/bash
#SBATCH --job-name=mermaid-hires
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=71:59:00
#SBATCH --array=0-3
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-hires-%A_%a.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-hires-%A_%a.err

# Higher-resolution fine-tune: 50k updates (4k warmup, then linear decay to 0).
# Array order MUST match scripts/compile_hires_evals.py TRAIN_RESOLUTIONS.
#   0  256  microbatch 32  accum 1
#   1  384  microbatch 16  accum 2
#   2  512  microbatch  8  accum 4
#   3  768  microbatch  4  accum 8
# Effective batch is 32. Checkpoints are local only:
# nbs/model_checkpoints/cbm_hires_res<R>_mb<B>/model_step* every 10k updates.
#
# The first start loads weights only from SRC_CKPT (fresh AdamW, data seed
# 3004). A resubmit resumes that run's latest model_step* with data seed 4004
# and does not read SRC_CKPT again. The training script adds global_step, so a
# resume from step 30000 shuffles with 43004. A finished run (model_step50000,
# epoch complete) loads that checkpoint and exits 0. To fine-tune a different
# source, change SRC_CKPT and RUN_PREFIX together so these directories are not
# reused.
#
#   sbatch slurm/hires_train.sbatch
#   sbatch --array=3 slurm/hires_train.sbatch   # 768 only

set -euo pipefail

# Rank-16 ViT-L, loss weights 0.5/0.5, cooled after 675k (700k total). Read-only.
SRC_CKPT="model_checkpoints/cbm_duration_pix0.5_img0.5_cd675000/model_epoch0"
RUN_PREFIX="cbm_hires"
# Outside the duration seeds (4..43), cooldown (1004), and duration restart (2004).
# A first start keeps DATA_SEED. A resume switches to RESTART_DATA_SEED so the
# shuffle is not the original stream; the entrypoint then adds global_step.
DATA_SEED=3004
RESTART_DATA_SEED=4004

# ---------- resolution grid (indexed by SLURM_ARRAY_TASK_ID) ----------
RESOLUTIONS=(256 384 512 768)
MICROBATCHES=(32 16 8 4)
ACCUM_STEPS=(1 2 4 8)
i=${SLURM_ARRAY_TASK_ID}
RES=${RESOLUTIONS[$i]}
MICROBATCH=${MICROBATCHES[$i]}
ACCUM=${ACCUM_STEPS[$i]}
RUN="${RUN_PREFIX}_res${RES}_mb${MICROBATCH}"
DATA_CONFIG="../configs/data_config_hires_${RES}.yaml"
case "$RES" in
    256) MODEL_CONFIG="../configs/model_config_cbm_dpt_lora_vitl_256.yaml" ;;
    512) MODEL_CONFIG="../configs/model_config_cbm_dpt_lora_vitl.yaml" ;;
    *) MODEL_CONFIG="../configs/model_config_cbm_dpt_lora_vitl_${RES}.yaml" ;;
esac

# 32 CPUs, 28 workers, leave the rest for the main process and pin-memory thread.
export MERMAIDSEG_NUM_WORKERS=28

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
export WANDB_RUN_GROUP=cbm_hires

# Pre-flight: wandb must be authenticated via the env file.
: "${WANDB_API_KEY:?set WANDB_API_KEY in $SCRATCH/mermaid-segmentation/.env}"

# ---------- resume seed (checkpoint files are local; torch.load is later) ----------
# model_step{N} is the checkpoint after N optimizer updates of THIS run.
NBS_DIR="$SCRATCH/mermaid-segmentation/nbs"
CKPT_DIR="$NBS_DIR/model_checkpoints/${RUN}"
shopt -s nullglob
ckpts=("${CKPT_DIR}"/model_step*)
shopt -u nullglob
LATEST_CKPT=""
if ((${#ckpts[@]})); then
    LATEST_CKPT=$(printf '%s\n' "${ckpts[@]}" | sort -V | tail -1)
    step_file="$(basename "$LATEST_CKPT")"
    step="${step_file#model_step}"
    if [[ ! "$step" =~ ^[0-9]+$ ]]; then
        echo "[batch] unrecognized checkpoint name: ${LATEST_CKPT}" >&2
        exit 1
    fi
    DATA_SEED=$RESTART_DATA_SEED
fi

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=${SLURM_JOB_ID}  task=${SLURM_ARRAY_TASK_ID}  run=${RUN}  res=${RES}  microbatch=${MICROBATCH}  accum=${ACCUM}  gpus=${CUDA_VISIBLE_DEVICES}  workers=${MERMAIDSEG_NUM_WORKERS}  data_seed=${DATA_SEED}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ---------- run ----------
cd $SCRATCH/mermaid-segmentation/nbs
uv sync --locked                                 # cheap no-op if cache is warm

# Confirm a checkpoint file has model weights. When EXPECTED_STEP is set, the
# saved global_step must match the model_step{N} filename.
verify_ckpt() {
    local path="$1"
    local expected_step="${2:-}"
    uv run python - "$path" "$expected_step" <<'PY'
import sys

import torch

path, expected = sys.argv[1], sys.argv[2]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict")
if not isinstance(state, dict):
    raise SystemExit(f"{path} has no model_state_dict")
step = ckpt.get("global_step")
print(
    f"[batch] checkpoint ok path={path} global_step={step} "
    f"epoch_complete={ckpt.get('epoch_complete')}"
)
if expected:
    if step is None:
        raise SystemExit(f"{path} has no global_step")
    if int(step) != int(expected):
        raise SystemExit(f"{path} global_step {step} != {expected}")
PY
}

# ---------- resume from latest step checkpoint, else init weights ----------
# A completed run's model_step50000 has epoch_complete set, so training exits 0.
WEIGHT_ARGS=()
if [[ -n "$LATEST_CKPT" ]]; then
    echo "[batch] resuming ${RUN} from ${LATEST_CKPT} data_seed=${DATA_SEED}"
    verify_ckpt "$LATEST_CKPT" "$step"
    WEIGHT_ARGS=(--checkpoint "$LATEST_CKPT")
else
    if [[ ! -f "$SRC_CKPT" ]]; then
        echo "[batch] source checkpoint missing: ${SRC_CKPT}" >&2
        exit 1
    fi
    echo "[batch] no checkpoint found for ${RUN}; initializing weights from ${SRC_CKPT}"
    verify_ckpt "$SRC_CKPT"
    WEIGHT_ARGS=(--init-weights "$SRC_CKPT")
fi

srun uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py \
    --run-name "$RUN" \
    --training-config ../configs/training_config_cbm_hires.yaml \
    --model-config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --per-pixel-loss-weight 0.5 \
    --per-image-loss-weight 0.5 \
    --batch-size "$MICROBATCH" \
    --grad-accum-steps "$ACCUM" \
    --data-seed "$DATA_SEED" \
    "${WEIGHT_ARGS[@]}"
