#!/bin/bash
# Idempotent train launcher for the Engaging loss-weight sweep.
#
# Submits array tasks only for grid points with no checkpoint and no queued
# job. Default GPU is an H100. Pass --gpu a100|h200 to override.
#
#   slurm/engaging/sweep_loss_weights_launch_train.sh [--dry-run] [--gpu a100|h100|h200]
#
# A run that already has a checkpoint is left alone. Preemption and the 48h
# handoff resume it. To resume a crashed run, sbatch the train script again.

set -euo pipefail

DRY_RUN=0
GPU=h100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --gpu)
            GPU="${2:?--gpu needs a100, h100, or h200}"
            shift
            ;;
        --gpu=*) GPU="${1#--gpu=}" ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: slurm/engaging/sweep_loss_weights_launch_train.sh [--dry-run] [--gpu a100|h100|h200]" >&2
            exit 2
            ;;
    esac
    shift
done
case "$GPU" in
    a100|h100|h200) ;;
    *) echo "gpu must be a100, h100, or h200 (got $GPU)" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TRAIN_SBATCH="slurm/engaging/sweep_loss_weights_train.sbatch"
JOB_NAME="mermaid-lossw"
CKPT_ROOT="nbs/model_checkpoints"

PIX=(1.0 0.9 0.75 0.5 0.25 0.1 0.0)
IMG=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)

BUSY_IDS=""
if command -v squeue >/dev/null 2>&1; then
    BUSY_IDS="$(squeue --me -h -r -n "$JOB_NAME" -o '%K' 2>/dev/null || true)"
fi

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

to_launch=()
n_ckpt=0
n_running=0
for i in "${!PIX[@]}"; do
    run="cbm_lossw_pix${PIX[$i]}_img${IMG[$i]}"
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
echo "grid points          : ${#PIX[@]}"
echo "skipped (checkpoint)  : $n_ckpt"
echo "skipped (in queue)    : $n_running"
echo "to launch             : ${#to_launch[@]} on ${GPU}"

if [[ ${#to_launch[@]} -eq 0 ]]; then
    echo "nothing to launch — every run is checkpointed or already queued."
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
