#!/bin/bash
# One-shot, idempotent eval launcher for constant-peak-LR duration checkpoints.
#
# Scans the main training run (no cooldown) and submits an eval sbatch for
# every (checkpoint, resolution) pair that has not been evaluated yet. Safe to
# run repeatedly as model_epoch* files appear: it skips evals that are already
# done or currently queued, and re-submits ones that failed (no .done marker,
# not in the queue). Pass --force to ignore existing .done markers and re-run
# every checkpoint.
#
# Usage:
#   slurm/duration_launch_peak_evals.sh [--dry-run] [--force]
#
# --force resubmits evals that already have a .done marker, so a protocol
# change can overwrite summary.json / metrics.json in place. Jobs that are
# already queued are still skipped. The eval sbatch deletes .done before
# python starts and writes it again only after a clean exit.
#
# Run from the login node (from anywhere; paths resolve relative to the repo).

set -euo pipefail

DRY_RUN=0
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: slurm/duration_launch_peak_evals.sh [--dry-run] [--force]" >&2
            exit 2
            ;;
    esac
    shift
done

# Resolve repo root from this script's location (slurm/<this>).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Exact run directory, so cbm_duration_pix0.5_img0.5_cd* cooldown checkpoints
# are left to slurm/duration_launch_evals.sh.
CKPT_GLOB="nbs/model_checkpoints/cbm_duration_pix0.5_img0.5/model_epoch*"
EVAL_SBATCH="slurm/sweep_loss_weights_eval.sbatch"
OUT_BASE="eval_out/duration_peak"
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
    echo "(the constant-LR training job may not have written a checkpoint yet)"
    exit 0
fi

for ckpt in "${checkpoints[@]}"; do
    [[ -f "$ckpt" ]] || { n_missing_ckpt=$((n_missing_ckpt + 1)); continue; }

    run="$(basename "$(dirname "$ckpt")")"          # cbm_duration_pix0.5_img0.5
    epoch_file="$(basename "$ckpt")"                 # model_epochN
    epoch="${epoch_file#model_epoch}"

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

        if [[ -f "$out/.done" && $FORCE -eq 0 ]]; then
            n_done=$((n_done + 1))
            continue
        fi
        if [[ -f "$out/.done" && $FORCE -eq 1 ]]; then
            echo "force   : $job_name (.done exists, resubmitting)"
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
