# Sourced by Engaging training sbatch scripts after RUN, TRAIN_CMD, EPOCH_ITERS,
# TOTAL_STEPS, SBATCH_PATH, and WANDB_RUN_GROUP are set.
# A mid-epoch resume keeps the training script's default seed and then offsets
# it by global_step, so a requeue does not replay the same shuffle.

: "${RUN:?}"
: "${EPOCH_ITERS:?}"
: "${TOTAL_STEPS:?}"
: "${SBATCH_PATH:?}"
: "${WANDB_RUN_GROUP:?}"
if ((${#TRAIN_CMD[@]} == 0)); then
    echo "[batch] TRAIN_CMD is empty" >&2
    exit 2
fi

: "${WANDB_API_KEY:?set WANDB_API_KEY in $REPO/.env}"
# wandb run ids are alphanumeric plus '_' and '-'. Dots in loss-weight names are replaced.
export WANDB_RUN_ID="engaging-${RUN//[^A-Za-z0-9_-]/-}"
export WANDB_RESUME=allow
export WANDB_RUN_GROUP

CKPT_DIR="$REPO/nbs/model_checkpoints/$RUN"
mkdir -p "$CKPT_DIR"

# A wall-time handoff records the continuation's job id here. If that job is
# still queued, this process (for example a requeue that lost the race) exits
# so two trainers cannot share one run directory.
if [[ -f "$CKPT_DIR/continue.job" ]]; then
    _next="$(tr -d '[:space:]' < "$CKPT_DIR/continue.job")"
    if [[ -n "$_next" && "$_next" != "${SLURM_JOB_ID:-}" ]]; then
        if squeue -h -j "$_next" 2>/dev/null | grep -q .; then
            echo "[batch] continuation job ${_next} is already queued; exiting"
            exit 0
        fi
    fi
    rm -f "$CKPT_DIR/continue.job"
fi

echo "[batch] node=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} task=${SLURM_ARRAY_TASK_ID:-} run=${RUN} wandb=${WANDB_RUN_ID} workers=${MERMAIDSEG_NUM_WORKERS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
engaging_require_gpu

cd "$REPO"
uv sync --locked

if training_complete "$CKPT_DIR" "$TOTAL_STEPS" "$EPOCH_ITERS"; then
    echo "[batch] ${RUN} already finished; exiting"
    exit 0
fi

LATEST="$(latest_checkpoint "$CKPT_DIR" "$EPOCH_ITERS")"
CKPT_ARGS=()
if [[ -n "$LATEST" ]]; then
    echo "[batch] resuming ${RUN} from ${LATEST}"
    verify_ckpt "$LATEST"
    CKPT_ARGS=(--checkpoint "$LATEST")
else
    echo "[batch] no checkpoint found for ${RUN}; starting from scratch"
fi

HANDOFF=0
TRAIN_PID=""
on_usr1() {
    HANDOFF=1
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
    fi
}
trap on_usr1 USR1

cd "$REPO/nbs"
# CKPT_ARGS is empty on a fresh run. Expand it only when it has elements so
# bash 3.2 with set -u does not treat an empty array as unbound.
if ((${#CKPT_ARGS[@]})); then
    srun "${TRAIN_CMD[@]}" "${CKPT_ARGS[@]}" &
else
    srun "${TRAIN_CMD[@]}" &
fi
TRAIN_PID=$!
set +e
wait "$TRAIN_PID"
status=$?
set -e

if [[ "$HANDOFF" -eq 1 ]]; then
    wait "$TRAIN_PID" 2>/dev/null || true
    if training_complete "$CKPT_DIR" "$TOTAL_STEPS" "$EPOCH_ITERS"; then
        echo "[batch] training complete at walltime; not resubmitting"
        exit 0
    fi
    echo "[batch] wall clock nearly elapsed; submitting a continuation"
    engaging_resubmit
    exit 0
fi
exit "$status"
