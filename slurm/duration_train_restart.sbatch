#!/bin/bash
#SBATCH --job-name=mermaid-duration-restart
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=14-00:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-duration-restart-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-duration-restart-%j.err

# Continue cbm_duration_pix0.5_img0.5 from the 350k checkpoint (model_epoch13)
# at peak LR, with no warmup, through the original 1M-step budget.
#
# Submit against a 14-day A100 reservation (the name is not hardcoded):
#   sbatch --reservation=RESERVATION_NAME slurm/duration_train_restart.sbatch
#
# Do not resubmit slurm/duration_train.sbatch for this continuation. That script
# uses the 4k-warmup config and data seed 4, so a resume would shuffle with
# 4 + start_epoch (the original stream). This script always passes --data-seed
# 2004; resume adds start_epoch, so epoch 14 is shuffled with 2018.
#
# Resubmitting THIS script after a later preemption continues from the newest
# model_epoch* at or after epoch 13. It will not rewind to 350k.

set -euo pipefail

RUN="cbm_duration_pix0.5_img0.5"
# model_epoch13 is the save after 350k steps. Never resume an earlier file.
MIN_EPOCH=13
ITERS_PER_EPOCH=25000
# Outside the original resume offsets (4..43) and the cooldown seed (1004).
DATA_SEED=2004

# 64 CPUs, but leave a few for the main process and the pin-memory thread.
# Prefetch 2 keeps the queued-batch count near the 32-CPU sweep (32*3=96).
export MERMAIDSEG_NUM_WORKERS=56
export MERMAIDSEG_PREFETCH_FACTOR=2

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
export WANDB_RUN_GROUP=cbm_duration

# Pre-flight: wandb must be authenticated via the env file.
: "${WANDB_API_KEY:?set WANDB_API_KEY in $SCRATCH/mermaid-segmentation/.env}"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=${SLURM_JOB_ID}  run=${RUN}  gpus=${CUDA_VISIBLE_DEVICES}  cpus=${SLURM_CPUS_PER_TASK}  workers=${MERMAIDSEG_NUM_WORKERS}  data_seed=${DATA_SEED}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ---------- run ----------
cd $SCRATCH/mermaid-segmentation/nbs
uv sync --locked                                 # cheap no-op if cache is warm

# ---------- resume from the newest checkpoint at or after 350k ----------
# Checkpoints are ./model_checkpoints/<run>/model_epochN. Newest index >= 13
# so a later preemption does not rewind and overwrite newer files. Abort when
# nothing has reached model_epoch13, so a 325k save cannot be treated as 350k.
CKPT_DIR="model_checkpoints/${RUN}"
best_epoch=-1
best_ckpt=""
shopt -s nullglob
for ckpt in "${CKPT_DIR}"/model_epoch*; do
    [[ -f "$ckpt" ]] || continue
    name="$(basename "$ckpt")"
    epoch="${name#model_epoch}"
    [[ "$epoch" =~ ^[0-9]+$ ]] || continue
    if (( epoch > best_epoch )); then
        best_epoch=$epoch
        best_ckpt=$ckpt
    fi
done
shopt -u nullglob

if (( best_epoch < MIN_EPOCH )); then
    echo "[batch] no checkpoint at or after model_epoch${MIN_EPOCH} in ${CKPT_DIR} (newest epoch=${best_epoch}); aborting" >&2
    exit 1
fi
echo "[batch] resuming ${RUN} from ${best_ckpt} (epoch ${best_epoch})"

# Abort unless the file is the save at the end of that epoch.
uv run python - "$best_ckpt" "$ITERS_PER_EPOCH" <<'PY'
import sys

import torch

path, iters = sys.argv[1], int(sys.argv[2])
ckpt = torch.load(path, map_location="cpu", weights_only=False)
if "global_step" not in ckpt:
    raise SystemExit(f"{path} has no global_step")
epoch = int(ckpt["epoch"])
step = int(ckpt["global_step"])
expected = (epoch + 1) * iters
groups = ckpt.get("optimizer_state_dict", {}).get("param_groups", [])
lrs = [group.get("lr") for group in groups]
print(f"[batch] checkpoint epoch={epoch} global_step={step} expected_step={expected} lrs={lrs}")
if step != expected:
    raise SystemExit(f"global_step {step} != (epoch+1)*{iters} = {expected}")
PY

srun uv run python Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py \
    --run-name "$RUN" \
    --training-config ../configs/training_config_cbm_duration_restart.yaml \
    --per-pixel-loss-weight 0.5 \
    --per-image-loss-weight 0.5 \
    --data-seed "$DATA_SEED" \
    --checkpoint "$best_ckpt"
