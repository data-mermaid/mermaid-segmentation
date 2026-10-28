#!/bin/bash
# Idempotent train launcher for the Engaging ViT-size sweep.
#
# Submits ViT-S/B/L (array indices 0-2, default H100) only when that size has
# no checkpoint and no queued job. ViT-H+ is a separate H200 job:
#
#   slurm/engaging/vitsize_launch_train.sh [--dry-run] [--gpu a100|h100|h200]
#   slurm/engaging/vitsize_launch_train.sh --hplus [--dry-run]
#
# A run that already has a checkpoint is left alone. Preemption (--requeue)
# and the 48h handoff resume it. To resume a crashed run, sbatch the train
# script again.

set -euo pipefail

DRY_RUN=0
HPLUS=0
GPU=h100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --hplus) HPLUS=1 ;;
        --gpu)
            GPU="${2:?--gpu needs a100, h100, or h200}"
            shift
            ;;
        --gpu=*) GPU="${1#--gpu=}" ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: slurm/engaging/vitsize_launch_train.sh [--dry-run] [--gpu a100|h100|h200] [--hplus]" >&2
            exit 2
            ;;
    esac
    shift
done
case "$GPU" in
    a100|h100|h200) ;;
    *) echo "gpu must be a100, h100, or h200 (got $GPU)" >&2; exit 2 ;;
esac
if [[ "$HPLUS" -eq 1 && "$GPU" != h100 ]]; then
    echo "--hplus always uses an H200; do not pass --gpu" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CKPT_ROOT="nbs/model_checkpoints"

has_ckpt() {
    local run="$1"
    local path
    shopt -s nullglob
    local ckpts=(
        "$CKPT_ROOT/$run"/model_step*
        "$CKPT_ROOT/$run"/model_epoch*
        "$CKPT_ROOT/$run"/model_preempt*
    )
    shopt -u nullglob
    # Bash 3.2 (macOS) errors on "${arr[@]}" when the array is empty and set -u.
    if ((${#ckpts[@]} == 0)); then
        return 1
    fi
    for path in "${ckpts[@]}"; do
        [[ "$(basename "$path")" == *.partial ]] && continue
        [[ -f "$path" ]] || continue
        return 0
    done
    return 1
}

if [[ "$HPLUS" -eq 1 ]]; then
    RUN="cbm_vitsize_vithplus"
    JOB_NAME="mermaid-vitsize-hplus"
    SBATCH="slurm/engaging/vitsize_train_hplus.sbatch"
    if has_ckpt "$RUN"; then
        echo "skip    : $RUN (has a checkpoint)"
        exit 0
    fi
    if squeue --me -h -n "$JOB_NAME" 2>/dev/null | grep -q .; then
        echo "skip    : $RUN (already queued/running)"
        exit 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "would submit : sbatch -G h200:1 $SBATCH (dry-run)"
    else
        mkdir -p "$HOME/orcd/scratch/mermaidseg/logs"
        sbatch -G h200:1 "$SBATCH"
        echo "submitted    : sbatch -G h200:1 $SBATCH"
    fi
    exit 0
fi

TRAIN_SBATCH="slurm/engaging/vitsize_train.sbatch"
JOB_NAME="mermaid-vitsize"
SIZES=(vits vitb vitl)

BUSY_IDS=""
if command -v squeue >/dev/null 2>&1; then
    BUSY_IDS="$(squeue --me -h -r -n "$JOB_NAME" -o '%K' 2>/dev/null || true)"
fi

to_launch=()
n_ckpt=0
n_running=0
for i in "${!SIZES[@]}"; do
    run="cbm_vitsize_${SIZES[$i]}"
    if has_ckpt "$run"; then
        echo "skip    : task $i  $run  (has a checkpoint)"
        n_ckpt=$((n_ckpt + 1))
        continue
    fi
    if [[ -n "$BUSY_IDS" ]] && grep -qxF "$i" <<<"$BUSY_IDS"; then
        echo "skip    : task $i  $run  (already queued/running)"
        n_running=$((n_running + 1))
        continue
    fi
    echo "launch  : task $i  $run  (no checkpoint, not in queue)"
    to_launch+=("$i")
done

echo
echo "===================== train launcher summary ===================="
echo "grid points          : ${#SIZES[@]} (H+ is --hplus)"
echo "skipped (checkpoint)  : $n_ckpt"
echo "skipped (in queue)    : $n_running"
echo "to launch             : ${#to_launch[@]} on ${GPU}"

if [[ ${#to_launch[@]} -eq 0 ]]; then
    echo "nothing to launch — every S/B/L run is checkpointed or already queued."
    echo "================================================================="
    exit 0
fi

array_spec="$(IFS=,; echo "${to_launch[*]}")"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "would submit         : sbatch -G ${GPU}:1 --array=$array_spec $TRAIN_SBATCH (dry-run)"
else
    mkdir -p "$HOME/orcd/scratch/mermaidseg/logs"
    sbatch -G "${GPU}:1" --array="$array_spec" "$TRAIN_SBATCH"
    echo "submitted            : sbatch -G ${GPU}:1 --array=$array_spec $TRAIN_SBATCH"
fi
echo "================================================================="
