#!/bin/bash
#SBATCH --job-name=taxon-div
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.err

# Per-point and per-image live hard coral genera / families (scripts/taxon_diversity_predict.py) of one taxv2
# scratch100k run at step 100k (<RUN>_cd/model_epoch0), scored with the run's pinned concept CSV and concept map.
# Writes eval_out/taxon_diversity/test_<NAME>_step100000_<TAG>{,_dense,_images}.parquet.
# Usage: sbatch scripts/analysis_slurm/taxon_diversity_predict.sbatch RUN NAME TAG "SOURCE SOURCE ..."
set -euo pipefail
RUN="${1:?usage: sbatch scripts/analysis_slurm/taxon_diversity_predict.sbatch RUN NAME TAG SOURCES}"
NAME="${2:?}"
TAG="${3:?}"
read -r -a SOURCES <<< "${4:?}"

REPO=/data/vision/beery/scratch/viktordo/mermaid-segmentation
SCRATCH=/data/vision/beery/scratch/viktordo
export HOME=$SCRATCH/home
export HF_HOME=$SCRATCH/hf-cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export TMPDIR=$SCRATCH/tmp
mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$REPO/logs"

cd "$REPO"
set -a; . ./.env; set +a

OUT=eval_out/taxon_diversity/test_${NAME}_step100000_${TAG}.parquet
echo "[batch] node=$(hostname) job=${SLURM_JOB_ID} run=${RUN} out=${OUT}"
.venv/bin/python scripts/taxon_diversity_predict.py --split test --sources "${SOURCES[@]}" \
    --checkpoint "nbs/model_checkpoints/${RUN}_cd/model_epoch0" \
    --concept-id2name "eval_out/scratch100k_hier/$RUN/concept_id2name.json" \
    --class-to-concepts "nbs/model_checkpoints/$RUN/class_to_concepts.csv" \
    --out "$OUT"
