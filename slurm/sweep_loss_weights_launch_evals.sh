#!/bin/bash
# One-shot, idempotent eval launcher for the loss-weight sweep.
#
# Scans for training checkpoints and submits an eval sbatch for every
# (checkpoint, resolution) pair that has not been evaluated yet. Safe to run
# repeatedly / sporadically: it skips evals that are already done or currently
# queued, and re-submits ones that failed (no .done marker, not in the queue).
#
# Usage:
#   slurm/sweep_loss_weights_launch_evals.sh [--dry-run]
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

CKPT_GLOB="nbs/model_checkpoints/cbm_lossw_*/model_epoch*"
EVAL_SBATCH="slurm/sweep_loss_weights_eval.sbatch"
OUT_BASE="eval_out/sweep_loss_weights"
RESOLUTIONS=(256 512)
# Skip checkpoints touched within this many seconds (still being written).
MIN_AGE_SEC=180

# Snapshot currently queued/running eval job names once (avoids one squeue per job).
QUEUED_NAMES=""
if command -v squeue >/dev/null 2>&1; then
    QUEUED_NAMES="$(squeue --me -h -o '%j' 2>/dev/null || true)"
fi

now=$(date +%s)
n_done=0
n_running=0
n_fresh=0
n_submitted=0
n_missing_ckpt=0

shopt -s nullglob
checkpoints=($CKPT_GLOB)
shopt -u nullglob

if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "No checkpoints found matching: $CKPT_GLOB"
    echo "(training may not have produced any checkpoints yet)"
    exit 0
fi

for ckpt in "${checkpoints[@]}"; do
    [[ -f "$ckpt" ]] || { n_missing_ckpt=$((n_missing_ckpt + 1)); continue; }

    run="$(basename "$(dirname "$ckpt")")"          # e.g. cbm_lossw_pix0.75_img0.25
    epoch_file="$(basename "$ckpt")"                 # e.g. model_epoch3
    epoch="${epoch_file#model_epoch}"               # e.g. 3

    # Skip freshly written checkpoints (torch.save may still be in progress).
    mtime=$(stat -c %Y "$ckpt" 2>/dev/null || stat -f %m "$ckpt")
    age=$((now - mtime))
    if [[ $age -lt $MIN_AGE_SEC ]]; then
        echo "fresh   : $run epoch$epoch (age ${age}s < ${MIN_AGE_SEC}s) — skipping for now"
        n_fresh=$((n_fresh + 1))
        continue
    fi

    for res in "${RESOLUTIONS[@]}"; do
        out="$OUT_BASE/$run/epoch${epoch}_res${res}"
        job_name="ev-${run}-e${epoch}-r${res}"

        if [[ -f "$out/.done" ]]; then
            n_done=$((n_done + 1))
            continue
        fi

        if grep -qxF "$job_name" <<<"$QUEUED_NAMES"; then
            echo "running : $job_name (already queued)"
            n_running=$((n_running + 1))
            continue
        fi

        mkdir -p "$out"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "SUBMIT  : $job_name  CKPT=$ckpt  RES=$res  OUT=$out  (dry-run)"
        else
            sbatch \
                --job-name="$job_name" \
                "$EVAL_SBATCH" "$ckpt" "$res" "$out"
            echo "submitted: $job_name"
        fi
        n_submitted=$((n_submitted + 1))
    done
done

echo
echo "===================== eval launcher summary ====================="
echo "checkpoints scanned : ${#checkpoints[@]}"
echo "already done        : $n_done"
echo "already running     : $n_running"
echo "fresh (skipped)     : $n_fresh"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "would submit        : $n_submitted (dry-run, nothing submitted)"
else
    echo "submitted           : $n_submitted"
fi
[[ $n_missing_ckpt -gt 0 ]] && echo "missing (skipped)   : $n_missing_ckpt"
echo "================================================================="
