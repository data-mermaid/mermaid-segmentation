#!/bin/bash
# Read-only inventory of the project's scratch space on the CSAIL cluster. Run it on
# a login node before scripts/cluster/cleanup.sh and look over what is there.
#
#   scripts/cluster/inventory.sh
#
# Prints your queued jobs, the size of every untracked or ignored path in the
# shared checkout (except .venv), and the size of the old -final clone, the
# scratch tmp directory and the shared caches.

set -euo pipefail

SCRATCH="${SCRATCH:-/data/vision/beery/scratch/sauder}"
REPO="$SCRATCH/mermaid-segmentation"
OLD_CLONE="$SCRATCH/mermaid-segmentation-final"

size() {
    if [[ -e "$1" ]]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "-"
    fi
}

echo "== queued / running jobs"
if command -v squeue >/dev/null 2>&1; then
    squeue --me -o '%.12i %.45j %.10T %.12M %.22P' || true
else
    echo "(squeue not available)"
fi

echo
echo "== untracked and ignored paths in $REPO"
if [[ -d "$REPO/.git" ]]; then
    git -C "$REPO" status --porcelain --ignored --untracked-files=normal |
        sed -E 's/^(\?\?|!!) //' |
        grep -v -E '^\.venv/?$' |
        while IFS= read -r path; do
            printf '%8s  %s\n' "$(size "$REPO/$path")" "$path"
        done || echo "(git status failed)"
else
    echo "$REPO is not a git checkout"
fi

echo
echo "== other paths"
for path in "$OLD_CLONE" "$SCRATCH/tmp" "$REPO/.venv" "$SCRATCH/hf-cache" "$SCRATCH/uv-cache" \
    "$SCRATCH/uv-python" "$SCRATCH/pip-cache" "$SCRATCH/home"; do
    printf '%8s  %s\n' "$(size "$path")" "$path"
done

echo
echo "== scratch total"
printf '%8s  %s\n' "$(size "$SCRATCH")" "$SCRATCH"
