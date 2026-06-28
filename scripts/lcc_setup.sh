#!/bin/bash
# Lifecycle Configuration script for SageMaker JupyterLab spaces.
#
# AWS LCC constraints:
#   - 5-minute timeout: the script must EXIT within 5 min or the space fails to start.
#   - 16 KB size limit per script.
#   - Background slow operations with nohup ... & and exit immediately.
#
# Attach as the default LCC in:
#   SageMaker console > Domains > dev-SG-Project > Environment > Set as default
#
# For shared spaces via CLI:
#   aws sagemaker update-space --domain-id ... --space-name ... \
#       --space-settings '{"SpaceSettings":{"AppType":"JupyterLab","JupyterLabAppSettings":{"LifecycleConfigArns":["arn:..."]}}}'
#
# Monitor progress (all output written to EFS, persists across restarts):
#   tail -f ~/lcc-setup.log
#   make lcc-log   (from project directory)
set -eux
trap 'echo "[LCC ERROR] Failed at line $LINENO — check ~/lcc-setup.log or CloudWatch /aws/sagemaker/studio" >&2' ERR

if [ -z "${PROJECT_DIR:-}" ]; then
    for candidate in \
        "$HOME/mermaid-segmentation" \
        "$HOME/SageMaker/mermaid-segmentation" \
        "$PWD/mermaid-segmentation"; do
        if [ -f "$candidate/pyproject.toml" ]; then
            PROJECT_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "${PROJECT_DIR:-}" ] || [ ! -f "$PROJECT_DIR/.jupyter/jupyter_lab_config.py" ]; then
    echo "[LCC] Could not determine PROJECT_DIR. Set PROJECT_DIR to the repo path before running."
    exit 1
fi

LOG_FILE="$HOME/lcc-setup.log"

# Redirect all subsequent output to the persistent EFS log.
# This log survives space restarts and is readable from any terminal tab.
exec >> "$LOG_FILE" 2>&1
echo "[LCC] ===== Started $(date) ====="

# Install uv to ~/.local/bin (EFS, persists across space restarts).
# On subsequent restarts the binary is already present — this is a no-op.
if ! command -v uv &>/dev/null; then
    echo "[LCC] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Deploy the Jupyter config to ~/.jupyter/ (EFS) so JupyterLab picks it up.
# Source is version-controlled in the repo; destination is where JupyterLab reads it.
# Done synchronously — fast (file copy).
mkdir -p "$HOME/.jupyter"
cp "$PROJECT_DIR/.jupyter/jupyter_lab_config.py" "$HOME/.jupyter/jupyter_lab_config.py"
echo "[LCC] Jupyter config deployed to ~/.jupyter/"

# Background the slow work so the LCC exits within the 5-minute limit.
#
# First run:  uv sync ~60s + ipykernel install ~10s = ~70s total (fine in background).
# Subsequent: uv sync ~5s (diff only) — fast in either mode.
#
# --locked pins exact versions from uv.lock (bit-for-bit reproducible).
# Progress is written to the same EFS log — follow with: tail -f ~/lcc-setup.log
echo "[LCC] Backgrounding uv sync + kernel registration (follow: tail -f $LOG_FILE)"
nohup bash -c "
  set -euo pipefail
  export PATH=\"$HOME/.local/bin:\$PATH\"
  cd '$PROJECT_DIR'
  echo \"[LCC bg] Starting uv sync \$(date)\"
  uv sync --all-extras --locked
  echo \"[LCC bg] uv sync complete \$(date)\"
  uv run --all-extras python -m ipykernel install \
      --user \
      --name=mermaid-seg \
      --display-name 'Python (mermaid-seg)'
  echo '[LCC bg] Kernel registered: mermaid-seg'
  uv run --all-extras python -c \"
import mermaidseg, mlflow
v = getattr(mermaidseg, '__version__', 'dev')
print(f'[LCC bg] OK: mermaidseg={v}, mlflow={mlflow.__version__}')
\"
  echo \"[LCC bg] ===== Sync complete \$(date) =====\"
" >> "$LOG_FILE" 2>&1 &

echo "[LCC] Background job PID: $!"
echo "[LCC] Synchronous setup complete $(date) — kernel and venv loading in background"
echo "[LCC] Monitor: tail -f $LOG_FILE"
