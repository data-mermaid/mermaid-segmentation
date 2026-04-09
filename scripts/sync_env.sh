#!/bin/bash
# Manual re-sync script for SageMaker JupyterLab terminal sessions.
#
# Run this after a `git pull`, when pyproject.toml changes, or whenever
# you need to update the venv without restarting the space.
#
# Usage (from project root or any path):
#   bash scripts/sync_env.sh          # pulls latest and syncs
#   bash scripts/sync_env.sh --no-pull # skips git pull (useful on detached branches)
#
# After sync completes, restart the kernel in JupyterLab to pick up new packages:
#   Kernel menu → Restart Kernel…
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/mermaid-segmentation}"
export PATH="$HOME/.local/bin:$PATH"

NO_PULL=false
for arg in "$@"; do
    [[ "$arg" == "--no-pull" ]] && NO_PULL=true
done

cd "$PROJECT_DIR"

if [ "$NO_PULL" = false ]; then
    echo "Pulling latest changes..."
    git pull --ff-only
fi

echo "Syncing uv environment (--group notebooks --locked)..."
uv sync --group notebooks --locked

echo "Registering Jupyter kernel..."
uv run python -m ipykernel install \
    --user \
    --name=mermaid-seg \
    --display-name "Python (mermaid-seg)"

echo "Sanity check..."
uv run python -c "
import mermaidseg, mlflow
v = getattr(mermaidseg, '__version__', 'dev')
print(f'OK: mermaidseg={v}, mlflow={mlflow.__version__}')
"

echo ""
echo "Sync complete. To use new packages: Kernel → Restart Kernel in JupyterLab."
