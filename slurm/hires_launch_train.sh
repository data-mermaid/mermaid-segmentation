#!/bin/bash
# Idempotent train launcher for the higher-resolution fine-tune.
#
# Submits array tasks for resolutions that are not finished and not already
# queued. A run is finished when its latest model_step{N} is at least
# TARGET_STEP (50000). An unfinished run is resubmitted; hires_train.sbatch
# resumes from that checkpoint with a different data seed.
#
#   slurm/hires_launch_train.sh
#   slurm/hires_launch_train.sh --dry-run
#
# Usage:
#   slurm/hires_launch_train.sh [--dry-run]
#
# Run from the login node (from anywhere; paths resolve relative to the repo).

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ -n "${1:-}" ]]; then
    echo "unknown argument: $1" >&2
    echo "usage: slurm/hires_launch_train.sh [--dry-run]" >&2
    exit 2
fi

# Resolve repo root from this script's location (slurm/<this>).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_SBATCH="slurm/hires_train.sbatch"
JOB_NAME="mermaid-hires"
CKPT_ROOT="nbs/model_checkpoints"

# ---------- resolution grid (MUST match hires_train.sbatch) ----------
RESOLUTIONS=(256 384 512 768)
MICROBATCHES=(32 16 8 4)
# iterations_per_train_epoch. model_step50000 is a finished run.
TARGET_STEP=50000

# Snapshot the array indices already queued/running for this sweep. `-r` expands
# array jobs to one task per line so pending ranges (e.g. 12345_[4-6]) are
# enumerated; `%K` prints each element's array index.
BUSY_IDS=""
if command -v squeue >/dev/null 2>&1; then
    BUSY_IDS="$(squeue --me -h -r -n "$JOB_NAME" -o '%K' 2>/dev/null || true)"
fi

to_launch=()
n_complete=0
n_running=0

for i in "${!RESOLUTIONS[@]}"; do
    run="cbm_hires_res${RESOLUTIONS[$i]}_mb${MICROBATCHES[$i]}"

    # Finished only at model_step50000. Earlier step files are a timed-out run.
    shopt -s nullglob
    ckpts=("$CKPT_ROOT/$run"/model_step*)
    shopt -u nullglob
    latest=0
    for f in "${ckpts[@]}"; do
        step="${f##*/model_step}"
        if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest )); then
            latest=$step
        fi
    done
    if (( latest >= TARGET_STEP )); then
        echo "skip    : task $i  $run  (complete at step $latest)"
        n_complete=$((n_complete + 1))
        continue
    fi

    # Queued/running => leave alone, whether or not a checkpoint exists yet.
    if [[ -n "$BUSY_IDS" ]] && grep -qxF "$i" <<<"$BUSY_IDS"; then
        echo "skip    : task $i  $run  (already queued/running)"
        n_running=$((n_running + 1))
        continue
    fi

    if (( latest > 0 )); then
        echo "launch  : task $i  $run  (resume from step $latest)"
    else
        echo "launch  : task $i  $run  (no checkpoint, not in queue)"
    fi
    to_launch+=("$i")
done

echo
echo "===================== train launcher summary ===================="
echo "grid points           : ${#RESOLUTIONS[@]}"
echo "skipped (complete)    : $n_complete"
echo "skipped (in queue)    : $n_running"
echo "to launch             : ${#to_launch[@]}"

if [[ ${#to_launch[@]} -eq 0 ]]; then
    echo "nothing to launch — every run is complete or already queued."
    echo "================================================================="
    exit 0
fi

# Comma-separated sparse array spec (e.g. "2,3") overrides the sbatch's
# #SBATCH --array=0-3 directive so only the missing runs are submitted.
array_spec="$(IFS=,; echo "${to_launch[*]}")"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "would submit         : sbatch --array=$array_spec $TRAIN_SBATCH (dry-run)"
else
    sbatch --array="$array_spec" "$TRAIN_SBATCH"
    echo "submitted            : sbatch --array=$array_spec $TRAIN_SBATCH"
fi
echo "================================================================="
