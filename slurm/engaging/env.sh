# Shared environment for Engaging batch scripts.
# Sourced, not executed. $HOME stays the real home directory (the CSAIL scripts
# relocate it). Code lives in $HOME; the dataset, HF cache, checkpoints, and
# wandb files live on scratch.

if [[ -z "${REPO:-}" ]]; then
    REPO="${MERMAIDSEG_REPO:-$HOME/mermaid-segmentation}"
fi
export REPO

export MERMAIDSEG_SCRATCH="${MERMAIDSEG_SCRATCH:-$HOME/orcd/scratch/mermaidseg}"
export PATH="$HOME/.local/bin:$PATH"
export MERMAIDSEG_NUM_WORKERS="${MERMAIDSEG_NUM_WORKERS:-8}"

if [[ -f "$REPO/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO/.env"
    set +a
fi

# Scratch paths win over anything a copied CSAIL .env might have set.
# MERMAIDSEG_LOCAL_CACHE_DIR is kept when the submit shell pointed it at pool.
_CSAIL_CACHE=/archive/vision/beery/red_sea_coral/cache_MERMAID
if [[ -z "${MERMAIDSEG_LOCAL_CACHE_DIR:-}" || "$MERMAIDSEG_LOCAL_CACHE_DIR" == "$_CSAIL_CACHE" ]]; then
    export MERMAIDSEG_LOCAL_CACHE_DIR="$MERMAIDSEG_SCRATCH/cache"
fi
export HF_HOME="$MERMAIDSEG_SCRATCH/hf"
export UV_CACHE_DIR="$MERMAIDSEG_SCRATCH/uv-cache"
export UV_LINK_MODE=copy
export PIP_CACHE_DIR="$MERMAIDSEG_SCRATCH/pip-cache"
export TMPDIR="$MERMAIDSEG_SCRATCH/tmp"
export WANDB_DIR="$MERMAIDSEG_SCRATCH/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-mermaidseg}"

mkdir -p "$TMPDIR" "$MERMAIDSEG_SCRATCH/logs" "$WANDB_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

# No S3. A copied CSAIL .env must not be able to supply keys.
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION || true
export AWS_EC2_METADATA_DISABLED=true
export MERMAIDSEG_S3_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

_ENGAGING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=resume.sh
source "$_ENGAGING_DIR/resume.sh"
