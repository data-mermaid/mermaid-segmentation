#!/bin/bash
# Delete every old training and eval artifact of the project on the CSAIL cluster, so
# the experiments can be retrained from scratch. Dry run by default: it only lists
# what would go and how large it is. Run scripts/cluster/inventory.sh first.
#
#   scripts/cluster/cleanup.sh             # dry run
#   scripts/cluster/cleanup.sh --execute   # delete
#   scripts/cluster/cleanup.sh --execute --force   # even with jobs queued
#
# Deletes, under $SCRATCH/mermaid-segmentation (the shared checkout):
#   nbs/model_checkpoints/*, model_checkpoints/, nbs/concept_id2name.json, eval_out/,
#   logs/*, wandb/, data_size/, data_quality/, data_sparse/,
#   stray *.partial files and __pycache__ directories (outside .venv)
# and also:
#   $SCRATCH/mermaid-segmentation-final (the old clone)
#   $SCRATCH/tmp/*
#
# Never touches .venv, $SCRATCH/hf-cache, uv-cache, uv-python, pip-cache, home,
# mermaid.env, the tracked files of the checkout, or the /archive data cache.

set -euo pipefail

usage="usage: scripts/cluster/cleanup.sh [--execute] [--force]"
EXECUTE=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=1 ;;
        --force) FORCE=1 ;;
        -h | --help)
            echo "$usage"
            exit 0
            ;;
        *)
            echo "$usage" >&2
            exit 2
            ;;
    esac
done

SCRATCH="${SCRATCH:-/data/vision/beery/scratch/sauder}"
REPO="$SCRATCH/mermaid-segmentation"
OLD_CLONE="$SCRATCH/mermaid-segmentation-final"

if [[ ! -d "$REPO/.git" ]]; then
    echo "$REPO is not a git checkout; refusing to run" >&2
    exit 1
fi

if command -v squeue >/dev/null 2>&1; then
    queued="$(squeue --me -h -o '%i %j' 2>/dev/null || true)"
    if [[ -n "$queued" ]]; then
        echo "you have queued or running jobs:"
        echo "$queued"
        if ((!FORCE)); then
            echo "cancel them first (scancel --me), or pass --force" >&2
            exit 1
        fi
    fi
fi

targets=()
add() {
    local path
    for path in "$@"; do
        [[ -e "$path" || -L "$path" ]] && targets+=("$path")
    done
    return 0
}

shopt -s nullglob dotglob
add "$REPO"/nbs/model_checkpoints/*
add "$REPO/model_checkpoints" "$REPO/nbs/concept_id2name.json" "$REPO/eval_out"
add "$REPO"/logs/*
add "$REPO/wandb" "$REPO/data_size" "$REPO/data_quality" "$REPO/data_sparse"
add "$OLD_CLONE"
add "$SCRATCH"/tmp/*
shopt -u nullglob dotglob

while IFS= read -r -d '' path; do
    targets+=("$path")
done < <(find "$REPO" -path "$REPO/.venv" -prune -o -path "$REPO/.git" -prune -o \
    \( -name '*.partial' -o -name '__pycache__' -type d \) -print0)

if ((${#targets[@]} == 0)); then
    echo "nothing to delete"
    exit 0
fi

for path in "${targets[@]}"; do
    case "$path" in
        "$REPO"/* | "$OLD_CLONE" | "$SCRATCH"/tmp/*) ;;
        *)
            echo "refusing to delete $path (outside the cleanup scope)" >&2
            exit 1
            ;;
    esac
    case "$path" in
        "$REPO/.venv" | "$REPO/.venv/"* | "$REPO/.git" | "$REPO/.git/"*)
            echo "refusing to delete $path" >&2
            exit 1
            ;;
    esac
done

echo "== $( ((EXECUTE)) && echo deleting || echo 'would delete (dry run; pass --execute)' )"
for path in "${targets[@]}"; do
    printf '%8s  %s\n' "$(du -sh "$path" 2>/dev/null | cut -f1)" "$path"
done
((EXECUTE)) || exit 0

for path in "${targets[@]}"; do
    rm -rf -- "$path"
done
mkdir -p "$REPO/logs" "$REPO/nbs/model_checkpoints"
echo "deleted ${#targets[@]} path(s)"
