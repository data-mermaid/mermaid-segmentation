#!/bin/bash
# Idempotent train launcher for the ViT-size sweep.
#
# Submits array tasks ONLY for sizes that have NOT started yet, i.e. those with
# no saved checkpoint AND no currently queued/running job. A timed-out ViT-H+
# (or any other size) is restarted by resubmitting the sbatch itself, which
# resumes from the latest model_step*:
#
#   sbatch slurm/vitsize_train.sbatch
#   sbatch --array=3 slurm/vitsize_train.sbatch   # H+ only
#
# This launcher will not do that restart: a run that already has a checkpoint
# is left alone, same as slurm/sweep_loss_weights_launch_train.sh.
#
# Usage:
#   slurm/vitsize_launch_train.sh [--dry-run]
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

TRAIN_SBATCH="slurm/vitsize_train.sbatch"
JOB_NAME="mermaid-vitsize"
CKPT_ROOT="nbs/model_checkpoints"

# ---------- ViT grid (MUST match vitsize_train.sbatch) ----------
SIZES=(vits vitb vitl vithplus)

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

for i in "${!SIZES[@]}"; do
    run="cbm_vitsize_${SIZES[$i]}"

    # Already has a checkpoint => it has started; leave it alone.
    shopt -s nullglob
    ckpts=("$CKPT_ROOT/$run"/model_step* "$CKPT_ROOT/$run"/model_epoch*)
    shopt -u nullglob
    if [[ ${#ckpts[@]} -gt 0 ]]; then
        echo "skip    : task $i  $run  (has ${#ckpts[@]} checkpoint(s))"
        n_ckpt=$((n_ckpt + 1))
        continue
    fi

    # Queued/running but no checkpoint yet => leave alone.
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
echo "grid points          : ${#SIZES[@]}"
echo "skipped (checkpoint)  : $n_ckpt"
echo "skipped (in queue)    : $n_running"
echo "to launch             : ${#to_launch[@]}"

if [[ ${#to_launch[@]} -eq 0 ]]; then
    echo "nothing to launch — every run is checkpointed or already queued."
    echo "Restart a timed-out run with: sbatch $TRAIN_SBATCH"
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
