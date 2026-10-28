# Checkpoint selection and wall-time handoff for Engaging training jobs.
# Sourced, not executed. No side effects beyond function definitions.

# Print the path of the checkpoint with the highest optimizer step, or nothing.
# model_preemptN and model_stepN contribute N. model_epochE contributes
# (E + 1) * epoch_iters. *.partial files are ignored. Equal steps keep the
# first match, and preempt files are listed first.
latest_checkpoint() {
    local dir="$1"
    local epoch_iters="$2"
    local best_step=-1
    local best_path=""
    local path base step
    [[ -d "$dir" ]] || return 0
    shopt -s nullglob
    for path in "$dir"/model_preempt* "$dir"/model_step* "$dir"/model_epoch*; do
        [[ -f "$path" ]] || continue
        base="$(basename "$path")"
        [[ "$base" == *.partial ]] && continue
        step=""
        if [[ "$base" =~ ^model_preempt([0-9]+)$ ]]; then
            step="${BASH_REMATCH[1]}"
        elif [[ "$base" =~ ^model_step([0-9]+)$ ]]; then
            step="${BASH_REMATCH[1]}"
        elif [[ "$base" =~ ^model_epoch([0-9]+)$ ]]; then
            step=$(( (${BASH_REMATCH[1]} + 1) * epoch_iters ))
        else
            continue
        fi
        if (( step > best_step )); then
            best_step=$step
            best_path=$path
        fi
    done
    shopt -u nullglob
    if [[ -n "$best_path" ]]; then
        printf '%s\n' "$best_path"
    fi
}

# Exit 0 when a finished interval checkpoint (or a preempt file at the final
# step) is already on disk. Loads at most the final-step files plus the newest
# model_epoch*, and skips that load when the epoch index cannot have reached
# total_steps yet.
training_complete() {
    local dir="$1"
    local total="$2"
    local epoch_iters="$3"
    uv run --project "$REPO" python - "$dir" "$total" "$epoch_iters" <<'PY'
import sys
from pathlib import Path

import torch

directory = Path(sys.argv[1])
total = int(sys.argv[2])
epoch_iters = int(sys.argv[3])


def finished(path: Path) -> bool:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — a truncated file is not "done"
        print(f"[batch] could not read {path}: {exc}", file=sys.stderr)
        return False
    step = int(ckpt.get("global_step") or 0)
    return step >= total and bool(ckpt.get("epoch_complete", True))


if not directory.is_dir():
    raise SystemExit(1)

for name in (f"model_step{total}", f"model_preempt{total}"):
    path = directory / name
    if path.is_file() and finished(path):
        raise SystemExit(0)

epoch_files: list[tuple[int, Path]] = []
for path in directory.glob("model_epoch*"):
    if not path.is_file() or path.name.endswith(".partial"):
        continue
    suffix = path.name.removeprefix("model_epoch")
    if suffix.isdigit():
        epoch_files.append((int(suffix), path))
if epoch_files:
    index, path = max(epoch_files)
    if (index + 1) * epoch_iters >= total and finished(path):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

# Refuse a checkpoint that cannot restore model, optimizer, and scheduler.
verify_ckpt() {
    local path="$1"
    uv run --project "$REPO" python - "$path" <<'PY'
import sys

import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
required = ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")
missing = [key for key in required if key not in ckpt]
if missing:
    raise SystemExit(f"{path} missing {missing}")
state = ckpt["model_state_dict"]
if not isinstance(state, dict) or not state:
    raise SystemExit(f"{path} has no model weights")
print(
    f"[batch] checkpoint ok path={path} global_step={ckpt.get('global_step')} "
    f"epoch={ckpt.get('epoch')} epoch_complete={ckpt.get('epoch_complete')}"
)
PY
}

# 80GB-class only. H100 80GB reports ~81559 MiB; L40S is ~44GB and must not run.
engaging_require_gpu() {
    local mib
    mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
    echo "[batch] gpu memory ${mib} MiB"
    if [[ "$mib" -lt 75000 ]]; then
        echo "[batch] refusing GPU with ${mib} MiB; need at least 75000 MiB" >&2
        exit 2
    fi
}

# Submit the same sbatch to continue past the 48h wall. --requeue does not
# restart a job that hits its time limit. The new job waits until this one
# has left the queue, and keeps the GPU type this allocation actually got.
engaging_resubmit() {
    local gpu="" line submission new_id
    line="$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null || true)"
    if [[ "$line" =~ gpu:([A-Za-z0-9]+):[0-9]+ ]]; then
        gpu="${BASH_REMATCH[1]}"
    fi
    local -a cmd=(sbatch --dependency="afterany:${SLURM_JOB_ID}")
    if [[ -n "$gpu" ]]; then
        cmd+=(-G "${gpu}:1")
    fi
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        cmd+=(--array="${SLURM_ARRAY_TASK_ID}")
    fi
    cmd+=("$SBATCH_PATH")
    echo "[batch] ${cmd[*]}"
    submission="$("${cmd[@]}")"
    echo "$submission"
    new_id="$(awk '{print $NF}' <<<"$submission")"
    if [[ -n "$new_id" ]]; then
        printf '%s\n' "$new_id" > "$CKPT_DIR/continue.job"
    fi
}
