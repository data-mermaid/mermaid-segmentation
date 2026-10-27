#!/bin/bash
# One-shot, idempotent launcher for the extended eval protocol on every sweep.
#
# For every (checkpoint, resolution) cell of the sweeps below, runs CoralNet on the
# val + test sources and the CoralscapesV2 probe on val + test (one SVM fit), with
# bleached / anthropogenic and hierarchical metrics. Pacific and Benthos are not
# re-run. Results go to <cell>/extended/ next to the cell's original eval, through
# that sweep's own eval sbatch (same model config and batch size), and are merged
# into the compiled CSVs by scripts/compile_*.py.
#
# Safe to run repeatedly: it skips evals that are already done or currently queued,
# and re-submits ones that failed (no .done marker, not in the queue).
#
# Usage:
#   slurm/extended_launch_evals.sh [--dry-run] [--force] [SWEEP ...]
#
# SWEEP restricts the launch to the named sweeps (default: all of them), e.g.
#   slurm/extended_launch_evals.sh hires vitsize
# --force resubmits evals that already have an extended/.done marker.
#
# Run from the login node (from anywhere; paths resolve relative to the repo).

set -euo pipefail

DRY_RUN=0
FORCE=0
ONLY=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        -*)
            echo "unknown argument: $1" >&2
            echo "usage: slurm/extended_launch_evals.sh [--dry-run] [--force] [SWEEP ...]" >&2
            exit 2
            ;;
        *) ONLY+=("$1") ;;
    esac
    shift
done

# Resolve repo root from this script's location (slurm/<this>).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EVAL_ARGS="--evals coralnet coralscapes --coralnet-splits val test --coralscapes-splits val test"
# Four split evals instead of the original suite; the hires sbatch's 4 h default is
# too short at 768/1024.
TIME_LIMIT="12:00:00"
# Skip checkpoints touched within this many seconds (still being written).
MIN_AGE_SEC=180

# name|checkpoint glob|eval sbatch|output root|prediction resolutions.
# Must match each sweep's own slurm/*_launch_*evals.sh.
SWEEPS=(
    "lossw|nbs/model_checkpoints/cbm_lossw_*/model_epoch*|slurm/sweep_loss_weights_eval.sbatch|eval_out/sweep_loss_weights|256 512"
    "duration_peak|nbs/model_checkpoints/cbm_duration_pix0.5_img0.5/model_epoch*|slurm/sweep_loss_weights_eval.sbatch|eval_out/duration_peak|256 512"
    "duration_cooldown|nbs/model_checkpoints/cbm_duration_pix0.5_img0.5_cd*/model_epoch0|slurm/sweep_loss_weights_eval.sbatch|eval_out/duration_cooldown|256 512"
    "vitsize|nbs/model_checkpoints/cbm_vitsize_*/model_step*|slurm/vitsize_eval.sbatch|eval_out/vitsize|256 512"
    "hires|nbs/model_checkpoints/cbm_hires_*/model_step*|slurm/hires_eval.sbatch|eval_out/hires|256 384 512 768 1024"
    "data_quality|nbs/model_checkpoints/cbm_dq_*/model_step*|slurm/data_quality_eval.sbatch|eval_out/data_quality|256 512"
)

# Snapshot currently queued/running eval job names once (avoids one squeue per job).
QUEUED_NAMES=""
if command -v squeue >/dev/null 2>&1; then
    QUEUED_NAMES="$(squeue --me -h -o '%j' 2>/dev/null || true)"
fi

now=$(date +%s)
n_ckpt=0
n_done=0
n_running=0
n_fresh=0
n_submitted=0

for spec in "${SWEEPS[@]}"; do
    IFS='|' read -r sweep ckpt_glob eval_sbatch out_base resolutions <<<"$spec"
    if [[ ${#ONLY[@]} -gt 0 && " ${ONLY[*]} " != *" $sweep "* ]]; then
        continue
    fi

    shopt -s nullglob
    checkpoints=($ckpt_glob)
    shopt -u nullglob
    echo "== $sweep: ${#checkpoints[@]} checkpoints ($ckpt_glob)"

    for ckpt in ${checkpoints[@]+"${checkpoints[@]}"}; do
        [[ -f "$ckpt" ]] || continue
        n_ckpt=$((n_ckpt + 1))
        run="$(basename "$(dirname "$ckpt")")"
        tag="$(basename "$ckpt")"
        tag="${tag#model_}"                          # epoch3 / step25000

        # Skip freshly written checkpoints (torch.save may still be in progress).
        mtime=$(stat -c %Y "$ckpt" 2>/dev/null || stat -f %m "$ckpt")
        age=$((now - mtime))
        if [[ $age -lt $MIN_AGE_SEC ]]; then
            echo "fresh   : $run $tag (age ${age}s < ${MIN_AGE_SEC}s) — skipping for now"
            n_fresh=$((n_fresh + 1))
            continue
        fi

        for res in $resolutions; do
            out="$out_base/$run/${tag}_res${res}/extended"
            job_name="evx-${run}-${tag}-r${res}"

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
                    --time="$TIME_LIMIT" \
                    "$eval_sbatch" "$ckpt" "$res" "$out" "$EVAL_ARGS"
                echo "submitted: $job_name"
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo
echo "================ extended eval launcher summary ================="
echo "checkpoints scanned : $n_ckpt"
echo "already done        : $n_done"
echo "already running     : $n_running"
echo "fresh (skipped)     : $n_fresh"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "would submit        : $n_submitted (dry-run, nothing submitted)"
else
    echo "submitted           : $n_submitted"
fi
echo "================================================================="
