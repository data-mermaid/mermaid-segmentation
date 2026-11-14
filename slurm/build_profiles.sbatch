#!/bin/bash
#SBATCH --job-name=mermaid-build-profiles
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.err

# Build the training profiles of the data experiments (CPU only).
#
#   sbatch slurm/build_profiles.sbatch datasize EXP... [ARGS...]
#       scripts/build_data_size_profiles.py into $REPO/data_size/profiles/<exp>_drop<f>/
#       (exp1: drop 0/10/20/50; exp2-exp4: drop 10/20/50)
#   sbatch slurm/build_profiles.sbatch sparse [ARGS...]
#       scripts/build_sparse_profile.py into $REPO/data_sparse/profile/
#
# ARGS go to the builder (e.g. --overwrite). slurm/launch_build_profiles.sh
# submits one job per kind.

set -euo pipefail

usage="usage: sbatch slurm/build_profiles.sbatch {datasize EXP...|sparse} [ARGS...]"
KIND="${1:?$usage}"
shift

source /data/vision/beery/scratch/sauder/mermaid-segmentation/slurm/common.sh

echo "[batch] node=$(hostname) job=${SLURM_JOB_ID:-} kind=$KIND args=$* cpus=${SLURM_CPUS_PER_TASK:-}"

case "$KIND" in
    datasize)
        srun "$PY" scripts/build_data_size_profiles.py --ds-root "$REPO/data_size" "$@"
        ;;
    sparse)
        srun "$PY" scripts/build_sparse_profile.py \
            --sparse-root "$REPO/data_sparse" \
            --num-proc "${SLURM_CPUS_PER_TASK:-32}" \
            "$@"
        ;;
    *)
        echo "$usage" >&2
        exit 2
        ;;
esac
