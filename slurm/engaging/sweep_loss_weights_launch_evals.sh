#!/bin/bash
# Idempotent eval launcher for the Engaging loss-weight sweep.
#
# Submits one eval per (model_epoch*, resolution) that is not done and not
# queued. model_preempt* files are preemption checkpoints and are not evaluated.
# Resolutions stay 256 and 512. Default GPU is an H100.
#
#   slurm/engaging/sweep_loss_weights_launch_evals.sh [--dry-run] [--force] [--gpu a100|h100|h200]

set -euo pipefail

DRY_RUN=0
FORCE=0
GPU=h100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --gpu)
            GPU="${2:?--gpu needs a100, h100, or h200}"
            shift
            ;;
        --gpu=*) GPU="${1#--gpu=}" ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: slurm/engaging/sweep_loss_weights_launch_evals.sh [--dry-run] [--force] [--gpu a100|h100|h200]" >&2
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

CKPT_GLOB="nbs/model_checkpoints/cbm_lossw_*/model_epoch*"
EVAL_SBATCH="slurm/engaging/sweep_loss_weights_eval.sbatch"
OUT_BASE="eval_out/sweep_loss_weights"
RESOLUTIONS=(256 512)
MIN_AGE_SEC=180

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
n_preempt=0

shopt -s nullglob
checkpoints=($CKPT_GLOB)
shopt -u nullglob

if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "No checkpoints found matching: $CKPT_GLOB"
    echo "(training may not have produced any epoch checkpoints yet)"
    exit 0
fi

for ckpt in "${checkpoints[@]}"; do
    [[ -f "$ckpt" ]] || { n_missing_ckpt=$((n_missing_ckpt + 1)); continue; }
    base="$(basename "$ckpt")"
    if [[ "$base" == model_preempt* || "$base" == *.partial ]]; then
        n_preempt=$((n_preempt + 1))
        continue
    fi

    run="$(basename "$(dirname "$ckpt")")"
    epoch="${base#model_epoch}"

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

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "SUBMIT  : sbatch -G ${GPU}:1 --job-name=$job_name $EVAL_SBATCH $ckpt $res $out  (dry-run)"
        else
            mkdir -p "$HOME/orcd/scratch/mermaidseg/logs" "$out"
            sbatch -G "${GPU}:1" --job-name="$job_name" "$EVAL_SBATCH" "$ckpt" "$res" "$out"
            echo "submitted: $job_name on ${GPU}"
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
echo "preempt (skipped)   : $n_preempt"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "would submit        : $n_submitted (dry-run, nothing submitted)"
else
    echo "submitted           : $n_submitted"
fi
[[ $n_missing_ckpt -gt 0 ]] && echo "missing (skipped)   : $n_missing_ckpt"
echo "================================================================="
