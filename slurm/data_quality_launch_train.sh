#!/bin/bash
# Idempotent train launcher for the data-quality sweep.
#
# Submits one training job per data-quality profile that has NOT started yet,
# i.e. with no saved checkpoint AND no queued/running job, and whose data has
# been built (sbatch slurm/data_quality_build.sbatch ...). With no profile names
# it takes every built profile under $DQ_ROOT/profiles. A timed-out run is
# restarted by resubmitting its job, which resumes from the latest model_step*:
#
#   sbatch --job-name=mermaid-dq-<profile> slurm/data_quality_train.sbatch <profile>
#
# This launcher will not do that restart: a run that already has a checkpoint
# is left alone, same as slurm/vitsize_launch_train.sh.
#
# Usage:
#   slurm/data_quality_launch_train.sh [--dry-run] [PROFILE ...]
#   slurm/data_quality_launch_train.sh phylum_0_0_0_0_0_0.1
#
# Run from the login node (from anywhere; paths resolve relative to the repo).

set -euo pipefail

USAGE="usage: slurm/data_quality_launch_train.sh [--dry-run] [PROFILE ...]"
DRY_RUN=0
PROFILES=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -*) echo "unknown option: $arg" >&2; echo "$USAGE" >&2; exit 2 ;;
        *) PROFILES+=("$arg") ;;
    esac
done

# Resolve repo root from this script's location (slurm/<this>).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_SBATCH="slurm/data_quality_train.sbatch"
JOB_PREFIX="mermaid-dq-"
CKPT_ROOT="nbs/model_checkpoints"
DQ_ROOT="${DQ_ROOT:-/data/vision/beery/scratch/sauder/mermaid-segmentation/data_quality}"
PROFILE_ROOT="$DQ_ROOT/profiles"

if [[ ${#PROFILES[@]} -eq 0 ]]; then
    shopt -s nullglob
    for config in "$PROFILE_ROOT"/*/data_config.yaml; do
        PROFILES+=("$(basename "$(dirname "$config")")")
    done
    shopt -u nullglob
    if [[ ${#PROFILES[@]} -eq 0 ]]; then
        echo "no built profiles under $PROFILE_ROOT; build one with sbatch slurm/data_quality_build.sbatch ..."
        exit 0
    fi
fi

# Snapshot the job names already queued/running (one squeue call for all profiles).
QUEUED_NAMES=""
if command -v squeue >/dev/null 2>&1; then
    QUEUED_NAMES="$(squeue --me -h -o '%j' 2>/dev/null || true)"
fi

to_launch=()
n_ckpt=0
n_running=0
n_no_data=0

for profile in "${PROFILES[@]}"; do
    run="cbm_dq_${profile}"

    # Already has a checkpoint => it has started; leave it alone.
    shopt -s nullglob
    ckpts=("$CKPT_ROOT/$run"/model_step* "$CKPT_ROOT/$run"/model_epoch*)
    shopt -u nullglob
    if [[ ${#ckpts[@]} -gt 0 ]]; then
        echo "skip    : $run  (has ${#ckpts[@]} checkpoint(s))"
        n_ckpt=$((n_ckpt + 1))
        continue
    fi

    # Queued/running but no checkpoint yet => leave alone.
    if [[ -n "$QUEUED_NAMES" ]] && grep -qxF "${JOB_PREFIX}${profile}" <<<"$QUEUED_NAMES"; then
        echo "skip    : $run  (already queued/running)"
        n_running=$((n_running + 1))
        continue
    fi

    if [[ ! -f "$PROFILE_ROOT/$profile/data_config.yaml" ]]; then
        build_args="$(sed 's/_/ /g; s/phylum /phylum --quantiles /' <<<"$profile")"
        echo "skip    : $run  (not built: sbatch slurm/data_quality_build.sbatch $build_args)"
        n_no_data=$((n_no_data + 1))
        continue
    fi

    echo "launch  : $run  (no checkpoint, not in queue)"
    to_launch+=("$profile")
done

echo
echo "===================== train launcher summary ===================="
echo "profiles              : ${#PROFILES[@]}"
echo "skipped (checkpoint)  : $n_ckpt"
echo "skipped (in queue)    : $n_running"
echo "skipped (not built)   : $n_no_data"
echo "to launch             : ${#to_launch[@]}"

if [[ ${#to_launch[@]} -eq 0 ]]; then
    echo "nothing to launch — every run is checkpointed, queued, or not built."
    echo "Restart a timed-out run with: sbatch --job-name=${JOB_PREFIX}<profile> $TRAIN_SBATCH <profile>"
    echo "================================================================="
    exit 0
fi

for profile in "${to_launch[@]}"; do
    cmd=(sbatch --job-name="${JOB_PREFIX}${profile}" "$TRAIN_SBATCH" "$profile")
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "would submit          : ${cmd[*]} (dry-run)"
    else
        "${cmd[@]}"
        echo "submitted             : ${cmd[*]}"
    fi
done
echo "================================================================="
