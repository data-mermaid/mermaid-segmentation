#!/bin/bash
# Idempotent cooldown launcher for the training-duration ablation.
#
# Scans for constant-LR checkpoints at the duration points below and submits a
# 25k linear cooldown (slurm/duration_cooldown.sbatch) for each one that does
# not already have a cooled checkpoint and is not already queued. Safe to run
# repeatedly over the 14-day training job.
#
# Usage:
#   slurm/duration_launch_cooldowns.sh [--dry-run]
#
# Run from the login node (from anywhere; paths resolve relative to the repo).

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ -n "${1:-}" ]]; then
    echo "unknown argument: $1" >&2
    echo "usage: slurm/duration_launch_cooldowns.sh [--dry-run]" >&2
    exit 2
fi

# Resolve repo root from this script's location (slurm/<this>).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

COOLDOWN_SBATCH="slurm/duration_cooldown.sbatch"
CKPT_ROOT="nbs/model_checkpoints"
MAIN_RUN="cbm_duration_pix0.5_img0.5"
# Must match iterations_per_train_epoch in configs/training_config_cbm_duration.yaml.
# model_epoch{N} is written after (N+1)*ITERS_PER_EPOCH steps.
ITERS_PER_EPOCH=25000
# Skip checkpoints touched within this many seconds (still being written).
MIN_AGE_SEC=180

# 25k, 75k, 125k, 175k, then every 100k from 275k through 975k. 225k is omitted.
STEPS=(25000 75000 125000 175000 275000 375000 475000 575000 675000 775000 875000 975000)

QUEUED_NAMES=""
if command -v squeue >/dev/null 2>&1; then
    QUEUED_NAMES="$(squeue --me -h -o '%j' 2>/dev/null || true)"
fi

now=$(date +%s)
n_done=0
n_running=0
n_fresh=0
n_missing=0
n_submitted=0

for step in "${STEPS[@]}"; do
    epoch=$((step / ITERS_PER_EPOCH - 1))
    # Relative to nbs/: duration_cooldown.sbatch cds there before training.
    src_from_nbs="model_checkpoints/$MAIN_RUN/model_epoch${epoch}"
    src="$CKPT_ROOT/$MAIN_RUN/model_epoch${epoch}"
    dst_run="${MAIN_RUN}_cd${step}"
    dst="$CKPT_ROOT/$dst_run/model_epoch0"
    job_name="durcd-${step}"

    if [[ -f "$dst" ]]; then
        n_done=$((n_done + 1))
        continue
    fi

    if grep -qxF "$job_name" <<<"$QUEUED_NAMES"; then
        echo "running : $job_name (already queued)"
        n_running=$((n_running + 1))
        continue
    fi

    if [[ ! -f "$src" ]]; then
        n_missing=$((n_missing + 1))
        continue
    fi

    mtime=$(stat -c %Y "$src" 2>/dev/null || stat -f %m "$src")
    age=$((now - mtime))
    if [[ $age -lt $MIN_AGE_SEC ]]; then
        echo "fresh   : $src (age ${age}s < ${MIN_AGE_SEC}s) — skipping for now"
        n_fresh=$((n_fresh + 1))
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "SUBMIT  : $job_name  SRC=$src_from_nbs  RUN=$dst_run  (dry-run)"
    else
        sbatch --job-name="$job_name" "$COOLDOWN_SBATCH" "$src_from_nbs" "$dst_run"
        echo "submitted: $job_name"
    fi
    n_submitted=$((n_submitted + 1))
done

echo
echo "===================== cooldown launcher summary ================="
echo "duration points     : ${#STEPS[@]}"
echo "already cooled      : $n_done"
echo "already running     : $n_running"
echo "source missing      : $n_missing"
echo "fresh (skipped)     : $n_fresh"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "would submit        : $n_submitted (dry-run, nothing submitted)"
else
    echo "submitted           : $n_submitted"
fi
echo "================================================================="
