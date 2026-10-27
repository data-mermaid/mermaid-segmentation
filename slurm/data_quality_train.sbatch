#!/bin/bash
#SBATCH --job-name=mermaid-dq
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=71:59:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.err

# Data-quality sweep: the ViT-size sweep's ViT-L recipe (DINOv3 ViT-L, 256 crops,
# batch 32, one 150k-update epoch: 4k warmup, then linear decay to 0, loss
# weights 0.5/0.5, default data seed) trained on one data-quality profile. Its
# baseline is cbm_vitsize_vitl from slurm/vitsize_train.sbatch, trained on the
# unmodified data. Same dataset lengths and seed, so every run sees the same
# sample order.
#
#   sbatch --job-name=mermaid-dq-<profile> slurm/data_quality_train.sbatch <profile>
#
# Usually submitted by slurm/data_quality_launch_train.sh, which finds queued runs
# by that job name. Build the profile first with slurm/data_quality_build.sbatch;
# it reads $DQ_ROOT/profiles/<profile>/data_config.yaml.
# Checkpoints are local only: nbs/model_checkpoints/cbm_dq_<profile>/model_step*
# every 25k optimizer updates. Resubmit the command above to continue a timed-out
# run. A finished run (model_step150000, epoch complete) loads that checkpoint
# and exits 0.

set -euo pipefail

PROFILE="${1:-}"
: "${PROFILE:?usage: sbatch --job-name=mermaid-dq-<profile> slurm/data_quality_train.sbatch <profile>}"
RUN="cbm_dq_${PROFILE}"
MODEL_CONFIG="../configs/model_config_cbm_dpt_lora_vitl_256.yaml"
TRAINING_CONFIG="../configs/training_config_cbm_vitsize.yaml"

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

# ---------- data-quality profile (scripts/build_data_quality_profile.py) ----------
DQ_ROOT=$SCRATCH/mermaid-segmentation/data_quality
DATA_CONFIG="$DQ_ROOT/profiles/${PROFILE}/data_config.yaml"
if [[ ! -f "$DATA_CONFIG" ]]; then
    BUILD_ARGS="$(sed 's/_/ /g; s/phylum /phylum --quantiles /' <<<"$PROFILE")"
    echo "[batch] profile $PROFILE is not built; run: sbatch slurm/data_quality_build.sbatch $BUILD_ARGS" >&2
    exit 1
fi

# ---------- configuration (wandb + AWS-free) ----------
# Non-secret repo defaults plus WANDB_API_KEY / WANDB_ENTITY live in .env.
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a

# This job needs NO AWS. Data is served from the pre-warmed local cache
# (see slurm/prefetch_cache.sbatch) and the static copies under $DQ_ROOT.
# Actively strip any AWS config so neither SSO nor long-lived keys are ever consulted.
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION || true

# Fail loudly on any cache miss instead of silently reaching for S3.
export MERMAIDSEG_S3_OFFLINE=1
export WANDB_DIR=$SCRATCH/mermaid-segmentation/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-mermaidseg}
export WANDB_RUN_GROUP=cbm_data_quality

# Pre-flight: wandb must be authenticated via the env file.
: "${WANDB_API_KEY:?set WANDB_API_KEY in $SCRATCH/mermaid-segmentation/.env}"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=${SLURM_JOB_ID}  run=${RUN}  profile=${PROFILE}  data=${DATA_CONFIG}  gpus=${CUDA_VISIBLE_DEVICES}  workers=${MERMAIDSEG_NUM_WORKERS}"
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
    --data-config "$DATA_CONFIG" \
    --training-config "$TRAINING_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --per-pixel-loss-weight 0.5 \
    --per-image-loss-weight 0.5 \
    "${CKPT_ARGS[@]}"
