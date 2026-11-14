#!/bin/bash
#SBATCH --job-name=mermaid-train
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=71:59:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.err

# One training job of a run: nbs/Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py.
#
#   sbatch [sbatch options] slurm/train.sbatch RUN [--init-weights SRC | --init-checkpoint SRC]
#          [--group GROUP] [-- PIPELINE_ARGS...]
#
# RUN            run name; checkpoints go to nbs/model_checkpoints/RUN.
# --init-weights / --init-checkpoint
#                starting point for a fresh run (path relative to the repo root or
#                absolute). Ignored once RUN has its own checkpoint.
# --group        wandb run group (default: RUN).
# PIPELINE_ARGS  passed through to the pipeline (configs, loss weights, resolution...).
#
# The job resumes from RUN's most recently written checkpoint (model_preempt*,
# model_step* or model_epoch*), so resubmitting the same command continues the
# run. A clean exit means training finished and writes nbs/model_checkpoints/RUN/.done;
# a job submitted after that exits immediately. With more than one GPU the job
# runs DDP under torchrun; --batch-size stays the global batch.
#
# Pass parameters as arguments, not --export=ALL,VAR=...: a site prolog cancels
# such jobs about a second after they start, without logs.

set -euo pipefail

usage="usage: sbatch slurm/train.sbatch RUN [--init-weights SRC | --init-checkpoint SRC] [--group GROUP] [-- PIPELINE_ARGS...]"
RUN="${1:?$usage}"
shift
INIT_ARGS=()
GROUP="$RUN"
while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --init-weights | --init-checkpoint)
            INIT_ARGS=("$1" "${2:?$usage}")
            shift 2
            ;;
        --group)
            GROUP="${2:?$usage}"
            shift 2
            ;;
        *)
            echo "$usage" >&2
            exit 2
            ;;
    esac
done
[[ "${1:-}" == "--" ]] && shift
PIPELINE_ARGS=("$@")

source /data/vision/beery/scratch/sauder/mermaid-segmentation/slurm/common.sh

: "${WANDB_API_KEY:?set WANDB_API_KEY in the secured env file (override path via MERMAIDSEG_ENV_FILE)}"

CKPT_DIR="$REPO/nbs/model_checkpoints/$RUN"
if [[ -e "$CKPT_DIR/.done" ]]; then
    echo "[batch] $RUN already finished ($CKPT_DIR/.done); nothing to do"
    exit 0
fi

IFS=',' read -r -a GPU_IDS <<<"${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#GPU_IDS[@]}
# Leave a few cores for the main processes; the rest are split over the ranks.
export MERMAIDSEG_NUM_WORKERS=$(((${SLURM_CPUS_PER_TASK:-32} - 4) / NGPU))

# Every resubmitted job of RUN appends to one wandb run. Bump MERMAIDSEG_WANDB_ID_PREFIX
# in the env file to start fresh wandb runs under the same run names.
export WANDB_RUN_ID="${MERMAIDSEG_WANDB_ID_PREFIX:-r1}-${RUN}"
export WANDB_RUN_GROUP="$GROUP"

# Newest checkpoint by write time. Training only moves forward, and older
# model_preempt* files are pruned, so the newest file is the furthest along.
shopt -s nullglob
ckpts=("$CKPT_DIR"/model_preempt* "$CKPT_DIR"/model_step* "$CKPT_DIR"/model_epoch*)
shopt -u nullglob
START_ARGS=()
latest=""
for ckpt in "${ckpts[@]}"; do
    [[ "$ckpt" == *.partial ]] && continue
    if [[ -z "$latest" || "$ckpt" -nt "$latest" ]]; then
        latest="$ckpt"
    fi
done
if [[ -n "$latest" ]]; then
    echo "[batch] resuming $RUN from $latest"
    START_ARGS=(--checkpoint "$latest")
elif ((${#INIT_ARGS[@]})); then
    src="${INIT_ARGS[1]}"
    [[ "$src" == /* ]] || src="$REPO/$src"
    if [[ ! -e "$src" ]]; then
        echo "[batch] initial checkpoint $src does not exist" >&2
        exit 1
    fi
    echo "[batch] starting $RUN from $src (${INIT_ARGS[0]})"
    START_ARGS=("${INIT_ARGS[0]}" "$src")
else
    echo "[batch] starting $RUN from scratch"
fi

echo "[batch] node=$(hostname) job=${SLURM_JOB_ID:-} run=$RUN gpus=${CUDA_VISIBLE_DEVICES:-} workers/rank=$MERMAIDSEG_NUM_WORKERS args=${PIPELINE_ARGS[*]:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

cd "$REPO/nbs"
PIPELINE=(Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py
    --run-name "$RUN"
    --preempt-checkpoint-every-steps "${MERMAIDSEG_PREEMPT_EVERY:-2000}"
    ${START_ARGS[@]+"${START_ARGS[@]}"}
    ${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"})
if ((NGPU > 1)); then
    export OMP_NUM_THREADS=1
    # On a timeout, the NCCL flight recorder logs each rank's last collectives.
    export MERMAIDSEG_NCCL_TIMEOUT_MIN="${MERMAIDSEG_NCCL_TIMEOUT_MIN:-30}"
    export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
    export TORCH_NCCL_DUMP_ON_TIMEOUT=1
    export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$REPO/logs/nccl_trace_${RUN}_${SLURM_JOB_ID:-$$}_rank_"
    srun "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "${PIPELINE[@]}"
else
    srun "$PY" "${PIPELINE[@]}"
fi

touch "$CKPT_DIR/.done"
echo "[batch] $RUN finished -> $CKPT_DIR/.done"
