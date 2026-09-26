#!/bin/bash
# Idempotent train (re)launcher for the loss-weight sweep.
#
# Submits training array tasks ONLY for grid points that have NOT started yet,
# i.e. those with no saved checkpoint AND no currently queued/running job. Use
# this to fill in sweep runs that never launched without disturbing runs that
# are already training (those keep running and resume from their own checkpoints
# via slurm/sweep_loss_weights_train.sbatch).
#
# Safe to run repeatedly: it skips runs that already have a checkpoint or are
# already in the queue, and only submits the missing ones as a sparse array.
#
# Usage:
#   slurm/sweep_loss_weights_launch_train.sh [--dry-run]
#
# Run from the login node (from anywhere; paths resolve relative to the repo).

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# Resolve repo root from this script's location (slurm/<this>).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_SBATCH="slurm/sweep_loss_weights_train.sbatch"
JOB_NAME="mermaid-lossw"
CKPT_ROOT="nbs/model_checkpoints"

# ---------- loss-weight grid (MUST match sweep_loss_weights_train.sbatch) ----------
PIX=(1.0 0.9 0.75 0.5 0.25 0.1 0.0)
IMG=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)

# Snapshot the array indices already queued/running for this sweep. `-r` expands
# array jobs to one task per line so pending ranges (e.g. 12345_[4-6]) are
# enumerated; `%K` prints each element's array index.
BUSY_IDS=""
if command -v squeue >/dev/null 2>&1; then
    BUSY_IDS="$(squeue --me -h -r -n "$JOB_NAME" -o '%K' 2>/dev/null || true)"
fi

to_launch=()
n_ckpt=0
n_running=0

for i in "${!PIX[@]}"; do
    run="cbm_lossw_pix${PIX[$i]}_img${IMG[$i]}"

    # Already has a checkpoint => it has started training; leave it alone.
    shopt -s nullglob
    ckpts=("$CKPT_ROOT/$run/model_epoch"*)
    shopt -u nullglob
    if [[ ${#ckpts[@]} -gt 0 ]]; then
        echo "skip    : task $i  $run  (has ${#ckpts[@]} checkpoint(s))"
        n_ckpt=$((n_ckpt + 1))
        continue
    fi

    # Queued/running but no checkpoint yet (started, just hasn't saved) => leave alone.
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
echo "to launch             : ${#to_launch[@]}"

if [[ ${#to_launch[@]} -eq 0 ]]; then
    echo "nothing to launch — every run is checkpointed or already queued."
    echo "================================================================="
    exit 0
fi

# Comma-separated sparse array spec (e.g. "2,5,6") overrides the sbatch's
# #SBATCH --array=0-6 directive so only the missing runs are submitted.
array_spec="$(IFS=,; echo "${to_launch[*]}")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "would submit         : sbatch --array=$array_spec $TRAIN_SBATCH (dry-run)"
else
    sbatch --array="$array_spec" "$TRAIN_SBATCH"
    echo "submitted            : sbatch --array=$array_spec $TRAIN_SBATCH"
fi
echo "================================================================="
