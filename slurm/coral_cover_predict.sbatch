#!/bin/bash
#SBATCH --job-name=coral-cover
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.err

# Per-image and per-point hard coral cover (scripts/coral_cover_predict.py) of one scratch100k run at step 100k
# (<RUN>_cd/model_epoch0) on the 10 CoralNet test sources of nbs/Coral_Cover_GT_vs_Base_10src*.ipynb.
# Runs that pin a concept CSV (<RUN>/class_to_concepts.csv, the *_taxv2 runs) are scored with that CSV and their
# concept map in eval_out/scratch100k_hier/<RUN>/concept_id2name.json.
# Writes eval_out/coral_cover/coralnet_test_pred_<NAME>_step100000_<TAG>{,_points}.parquet.
# Usage: sbatch [--job-name=...] slurm/coral_cover_predict.sbatch RUN NAME [TAG "SOURCE SOURCE ..."]
#   sbatch --job-name=cc-base-taxv2 slurm/coral_cover_predict.sbatch cbm_scratch100k_base_taxv2 base_taxv2
# TAG (default 10src) names the output; the source list (default: the 10 sources below) is one quoted argument.
set -euo pipefail
RUN="${1:?usage: sbatch slurm/coral_cover_predict.sbatch RUN NAME}"
NAME="${2:?usage: sbatch slurm/coral_cover_predict.sbatch RUN NAME}"
TAG="${3:-10src}"
if [[ -n "${4:-}" ]]; then read -r -a SOURCES <<< "$4"; else SOURCES=(6405 6551 6702 4957 2955 1990 3606 3697 4342 7116); fi

REPO=/data/vision/beery/scratch/viktordo/mermaid-segmentation
SCRATCH=/data/vision/beery/scratch/viktordo
export HOME=$SCRATCH/home
export HF_HOME=$SCRATCH/hf-cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export TMPDIR=$SCRATCH/tmp
mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$REPO/logs"

cd "$REPO"
set -a; . ./.env; set +a

extra=()
if [[ -f nbs/model_checkpoints/$RUN/class_to_concepts.csv ]]; then
    extra=(--concept-id2name "eval_out/scratch100k_hier/$RUN/concept_id2name.json"
           --class-to-concepts "nbs/model_checkpoints/$RUN/class_to_concepts.csv")
fi
OUT=eval_out/coral_cover/coralnet_test_pred_${NAME}_step100000_${TAG}
echo "[batch] node=$(hostname) job=${SLURM_JOB_ID} run=${RUN} out=${OUT} ${extra[*]}"
.venv/bin/python scripts/coral_cover_predict.py --split test --sources "${SOURCES[@]}" \
    --checkpoint "nbs/model_checkpoints/${RUN}_cd/model_epoch0" "${extra[@]}" \
    --out "${OUT}.parquet" --points-out "${OUT}_points.parquet"
