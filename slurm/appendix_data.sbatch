#!/bin/bash
#SBATCH --job-name=mermaid-appendix-data
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-appendix-data-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-appendix-data-%j.err

set -euo pipefail

# CPU-only export of per-label counts and raw sample images for the dataset appendix.
# Extra arguments are forwarded to scripts/export_appendix_data.py, e.g. a smoke run:
#
#   sbatch slurm/appendix_data.sbatch
#   sbatch slurm/appendix_data.sbatch --datasets moorea_labeled_corals coralscapes_v2 \
#       --n-images 2 --output-dir appendix_data_smoke
#
# Then on the laptop:
#   rsync -av <cluster>:/data/vision/beery/scratch/sauder/mermaid-segmentation/appendix_data ./

# ---------- relocate $HOME so nothing touches the real home dir ----------
export SCRATCH=/data/vision/beery/scratch/sauder
export HOME=$SCRATCH/home
export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
export XDG_DATA_HOME=$HOME/.local/share
export UV_CACHE_DIR=$SCRATCH/uv-cache
export UV_PYTHON_INSTALL_DIR=$SCRATCH/uv-python
export UV_LINK_MODE=copy
export HF_HOME=$SCRATCH/hf-cache
export PIP_CACHE_DIR=$SCRATCH/pip-cache
export TMPDIR=$SCRATCH/tmp
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$TMPDIR" "$HOME/.aws" "$SCRATCH/mermaid-segmentation/logs"

# Read the cache the train jobs already warmed. A miss must fail, not download.
export MERMAIDSEG_S3_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ---------- credentials ----------
set -a; source $SCRATCH/mermaid-segmentation/.env; set +a
source "$HOME/.aws/mermaid-training.env"

# ---------- diagnostics ----------
echo "[batch] node=$(hostname)  job=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK  args=$*"
aws sts get-caller-identity \
    || { echo "AWS credentials invalid — check $HOME/.aws/mermaid-training.env (perms 600, keys not rotated/disabled)"; exit 2; }

cd $SCRATCH/mermaid-segmentation
uv sync --locked

OUTPUT_DIR=appendix_data
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--output-dir" ]]; then
        OUTPUT_DIR="${args[$((i + 1))]}"
    fi
done

srun uv run python scripts/export_appendix_data.py --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"

# The pie charts use the source-level rows of the paper table. Always recompute:
# an existing dataset_stats/dataset_table.csv may be from an older CoralNet split.
echo "[batch] computing dataset_stats/dataset_table.csv"
srun uv run python scripts/dataset_table_stats.py --workers "${SLURM_CPUS_PER_TASK:-16}"
cp dataset_stats/dataset_table.csv "$OUTPUT_DIR/dataset_table.csv"
echo "[batch] appendix bundle written to $SCRATCH/mermaid-segmentation/$OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
