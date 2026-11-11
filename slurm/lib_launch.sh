#!/bin/bash
# Helpers shared by the slurm/launch_*.sh launchers. Source it; do not run it.
#
# Every launcher is idempotent: a run that is finished (nbs/model_checkpoints/RUN/.done)
# or has a queued/running job named RUN is skipped, so rerunning a launcher only
# submits what is missing. Common options:
#
#   --dry-run    print the sbatch commands instead of submitting them
#   RUN...       only consider these runs (default: every run of the launcher)

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT_ROOT="nbs/model_checkpoints"
EVAL_ROOT="${EVAL_ROOT:-eval_out}"
DS_ROOT="${DS_ROOT:-$REPO_ROOT/data_size}"
SPARSE_ROOT="${SPARSE_ROOT:-$REPO_ROOT/data_sparse}"

DRY_RUN=0
SELECTED=()
LAST_JOB_ID=""
N_SUBMITTED=0
_FAKE_JOB_ID=100000

parse_launcher_args() {
    while (($#)); do
        case "$1" in
            --dry-run) DRY_RUN=1 ;;
            -h | --help)
                echo "${USAGE:-}"
                exit 0
                ;;
            -*)
                echo "unknown option: $1" >&2
                echo "${USAGE:-}" >&2
                exit 2
                ;;
            *) SELECTED+=("$1") ;;
        esac
        shift
    done
    QUEUED_NAMES=""
    if command -v squeue >/dev/null 2>&1; then
        QUEUED_NAMES="$(squeue --me -h -o '%j' 2>/dev/null || true)"
    fi
}

# True if NAME was selected on the command line, or nothing was selected.
wanted() {
    ((${#SELECTED[@]} == 0)) && return 0
    local name
    for name in "${SELECTED[@]}"; do
        [[ "$name" == "$1" ]] && return 0
    done
    return 1
}

# True if a job named NAME is queued or running.
is_queued() {
    [[ -n "$QUEUED_NAMES" ]] && grep -qxF "$1" <<<"$QUEUED_NAMES"
}

run_finished() {
    [[ -e "$CKPT_ROOT/$1/.done" ]]
}

# True if checkpoint PATH exists and was last written over 10 minutes ago, so a
# run that is still writing it is not branched from.
checkpoint_ready() {
    [[ -e "$1" ]] && [[ -n "$(find "$1" -maxdepth 0 -mmin +10 2>/dev/null)" ]]
}

# sbatch_submit SBATCH_ARGS... : submit one job and set LAST_JOB_ID.
sbatch_submit() {
    if ((DRY_RUN)); then
        _FAKE_JOB_ID=$((_FAKE_JOB_ID + 1))
        LAST_JOB_ID="$_FAKE_JOB_ID"
        echo "  would submit: sbatch $*"
    else
        LAST_JOB_ID="$(sbatch --parsable "$@" | cut -d';' -f1)"
        echo "  submitted $LAST_JOB_ID: sbatch $*"
    fi
    N_SUBMITTED=$((N_SUBMITTED + 1))
}

# launch_run RUN DEPENDENCY [SBATCH_OPTS...] -- [TRAIN_ARGS...]
# Submit the training run RUN (slurm/train.sbatch RUN TRAIN_ARGS...) as one job named
# RUN, unless it is unselected, finished or queued. DEPENDENCY (may be empty) gates
# the job. Sets LAST_JOB_ID to the job, or to "" when nothing was submitted.
launch_run() {
    local run="$1" dep="$2"
    shift 2
    LAST_JOB_ID=""
    local opts=()
    while (($#)) && [[ "$1" != "--" ]]; do
        opts+=("$1")
        shift
    done
    (($#)) && shift
    wanted "$run" || return 0
    if run_finished "$run"; then
        echo "done    : $run"
        return 0
    fi
    if is_queued "$run"; then
        echo "queued  : $run"
        return 0
    fi
    local dep_opt=()
    [[ -n "$dep" ]] && dep_opt=(--dependency="$dep")
    echo "launch  : $run"
    sbatch_submit --job-name="$run" ${opts[@]+"${opts[@]}"} ${dep_opt[@]+"${dep_opt[@]}"} \
        slurm/train.sbatch "$run" "$@"
}

# launch_branch RUN SRC INIT_FLAG DEPENDENCY [SBATCH_OPTS...] -- [TRAIN_ARGS...]
# launch_run for a run that starts from checkpoint SRC (repo-relative) with
# INIT_FLAG (--init-weights or --init-checkpoint). Without DEPENDENCY, the run
# waits until SRC is written; with one, the dependency guarantees SRC instead.
launch_branch() {
    local run="$1" src="$2" init_flag="$3" dep="$4"
    shift 4
    local opts=()
    while (($#)) && [[ "$1" != "--" ]]; do
        opts+=("$1")
        shift
    done
    (($#)) && shift
    LAST_JOB_ID=""
    wanted "$run" || return 0
    if [[ -z "$dep" ]] && ! run_finished "$run" && ! is_queued "$run" && ! checkpoint_ready "$src"; then
        echo "waiting : $run (needs $src)"
        return 0
    fi
    launch_run "$run" "$dep" ${opts[@]+"${opts[@]}"} -- "$init_flag" "$src" "$@"
}

launcher_summary() {
    echo
    if ((DRY_RUN)); then
        echo "dry run: $N_SUBMITTED job(s) would be submitted"
    else
        echo "submitted $N_SUBMITTED job(s)"
    fi
}
