#!/bin/bash
# Idempotent eval launcher for the scratch100k runs (slurm/scratch100k_train.sbatch).
#
# Submits slurm/scratch100k_eval.sbatch for every finished checkpoint that has no .done marker
# and no queued job. Checkpoints are named by optimizer update:
#   <RUN>/model_epoch{0,1,2}  -> step25000 / step50000 / step75000 (constant-LR phase)
#   <RUN>_cd/model_epoch0     -> step100000 (after the 25k cooldown)
# Outputs: eval_out/scratch100k_hier/<RUN>/step<N>_res256/, which
#   python scripts/compile_hierarchical_evals.py --root eval_out/scratch100k_hier
# compiles. Re-run as more checkpoints appear.
# Runs with a pinned <RUN>/class_to_concepts.csv (tag=... runs) are scored against that CSV and
# need their concept map at eval_out/scratch100k_hier/<RUN>/concept_id2name.json.
#
# Usage (from the login node): slurm/scratch100k_launch_evals.sh [--dry-run] [--force]
set -euo pipefail

DRY_RUN=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        *) echo "usage: $0 [--dry-run] [--force]" >&2; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT_BASE=eval_out/scratch100k_hier
EVAL_SBATCH=slurm/scratch100k_eval.sbatch
MIN_AGE_SEC=180  # skip checkpoints written in the last 3 min (torch.save may still be running)

QUEUED_NAMES="$(squeue --me -h -o '%j' 2>/dev/null || true)"
now=$(date +%s)
n_done=0 n_queued=0 n_fresh=0 n_submitted=0

shopt -s nullglob
for ckpt in nbs/model_checkpoints/cbm_scratch100k_*/model_epoch*; do
    dir=$(basename "$(dirname "$ckpt")")
    epoch=${ckpt##*model_epoch}
    if [[ $dir == *_cd ]]; then
        run=${dir%_cd}
        step=$((100000 + epoch * 25000))
    else
        run=$dir
        step=$(((epoch + 1) * 25000))
    fi
    out=$OUT_BASE/$run/step${step}_res256
    job_name="ev-${run#cbm_scratch100k_}-s$((step / 1000))k"

    if [[ -f $out/.done && $FORCE -eq 0 ]]; then
        n_done=$((n_done + 1)); continue
    fi
    if grep -qxF "$job_name" <<<"$QUEUED_NAMES"; then
        echo "queued  : $job_name"; n_queued=$((n_queued + 1)); continue
    fi
    if (( now - $(stat -c %Y "$ckpt") < MIN_AGE_SEC )); then
        echo "fresh   : $ckpt (still being written?)"; n_fresh=$((n_fresh + 1)); continue
    fi
    extra=()
    pinned_csv=nbs/model_checkpoints/$run/class_to_concepts.csv
    if [[ -f $pinned_csv ]]; then
        concepts=$OUT_BASE/$run/concept_id2name.json
        if [[ ! -f $concepts ]]; then
            echo "skip    : $job_name (pinned CSV but no $concepts)"; continue
        fi
        extra=("$concepts" "$pinned_csv")
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "SUBMIT  : $job_name  $ckpt -> $out ${extra[*]} (dry-run)"
    else
        mkdir -p "$out"
        echo "submit  : $job_name  $(sbatch --parsable --job-name="$job_name" "$EVAL_SBATCH" "$ckpt" "$out" "${extra[@]}")"
    fi
    n_submitted=$((n_submitted + 1))
done

echo "done: $n_done  queued: $n_queued  fresh: $n_fresh  $([[ $DRY_RUN -eq 1 ]] && echo 'would submit' || echo submitted): $n_submitted"
