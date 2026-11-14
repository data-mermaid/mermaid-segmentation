#!/bin/bash
#SBATCH --job-name=gating-768
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.err

# Geographic gating predictions (points + all pixels, one pass) of cbm_cd275000_res768 at step 50k, 768 px, for
# inat/Genus_Geographic_Gating_CD275k_Res768.ipynb. Same 18 genus-eligible test sources and gate
# (inat/data/gating/gate_allow.json, combined iNat + train/val annotation matrix) as the 256 px KL05 notebook.
# Writes inat/data/gating_cd275k_res768/{points,dense,dense_images}.parquet, genus_probs.npy, genus_names.json.
# Usage: sbatch scripts/analysis_slurm/gating_predict_768.sbatch
set -euo pipefail

SOURCES=(290 1990 2311 2682 3357 3606 3656 3697 4091 4342 4721 4957 5181 5227 6405 6494 6702 7116)

REPO=/data/vision/beery/scratch/viktordo/mermaid-segmentation
SCRATCH=/data/vision/beery/scratch/viktordo
CKPT=/data/vision/beery/scratch/sauder/mermaid-segmentation-final/nbs/model_checkpoints/cbm_cd275000_res768/model_step50000
TAXV2=cbm_scratch100k_kl_w0.5_taxv2  # concept table / map identical to the checkpoint's
export HOME=$SCRATCH/home
export HF_HOME=$SCRATCH/hf-cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export TMPDIR=$SCRATCH/tmp
export PYTHONUNBUFFERED=1
mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$REPO/logs"

cd "$REPO"
set -a; . ./.env; set +a

OUT=inat/data/gating_cd275k_res768
echo "[batch] node=$(hostname) job=${SLURM_JOB_ID} sources=${#SOURCES[@]} out=${OUT} start=$(date +%T)"
.venv/bin/python inat/gating_dense_predict.py --split test --sources "${SOURCES[@]}" \
    --checkpoint "$CKPT" \
    --allow inat/data/gating/gate_allow.json \
    --concept-id2name "eval_out/scratch100k_hier/$TAXV2/concept_id2name.json" \
    --class-to-concepts "nbs/model_checkpoints/$TAXV2/class_to_concepts.csv" \
    --resolution 768 --batch-size 2 --with-points \
    --out-dir "$OUT"
echo "[batch] done=$(date +%T)"
