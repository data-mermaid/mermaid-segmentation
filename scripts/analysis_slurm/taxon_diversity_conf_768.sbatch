#!/bin/bash
#SBATCH --job-name=taxdiv-conf-768
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.err

# Confidence rerun of the 768 px diversity predictions (cbm_cd275000_res768, step 50k, KL w=0.5) for
# nbs/Coral_Taxon_Diversity_Live_CD275k_Res768_Regions_GenusConf.ipynb. Same model, concept table and settings as
# the original run (test_cd275k_res768_step50000_viable*), but only on the 19 sources that are reliably labelled at
# genus or family level (18 each; 290 is genus-only, 1776 family-only). Writes
# eval_out/taxon_diversity/test_cd275k_res768_step50000_viable_conf{,_dense,_hist,_images}.parquet;
# the notebook reads _conf (points with top-2 probabilities) and _conf_hist (pixels by confidence bin).
# Usage: sbatch scripts/analysis_slurm/taxon_diversity_conf_768.sbatch
set -euo pipefail

SOURCES=(290 1776 1990 2311 2682 3357 3606 3656 3697 4091 4342 4721 4957 5181 5227 6405 6494 6702 7116)

REPO=/data/vision/beery/scratch/viktordo/mermaid-segmentation
SCRATCH=/data/vision/beery/scratch/viktordo
CKPT=/data/vision/beery/scratch/sauder/mermaid-segmentation-final/nbs/model_checkpoints/cbm_cd275000_res768/model_step50000
TAXV2=cbm_scratch100k_kl_w0.5_taxv2  # concept table / map identical to the checkpoint's (checked in chat18)
export HOME=$SCRATCH/home
export HF_HOME=$SCRATCH/hf-cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export TMPDIR=$SCRATCH/tmp
export PYTHONUNBUFFERED=1
mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$REPO/logs"

cd "$REPO"
set -a; . ./.env; set +a

OUT=eval_out/taxon_diversity/test_cd275k_res768_step50000_viable_conf.parquet
echo "[batch] node=$(hostname) job=${SLURM_JOB_ID} sources=${#SOURCES[@]} out=${OUT} start=$(date +%T)"
.venv/bin/python scripts/taxon_diversity_predict.py --split test --sources "${SOURCES[@]}" \
    --checkpoint "$CKPT" \
    --concept-id2name "eval_out/scratch100k_hier/$TAXV2/concept_id2name.json" \
    --class-to-concepts "nbs/model_checkpoints/$TAXV2/class_to_concepts.csv" \
    --resolution 768 --batch-size 2 \
    --out "$OUT"
echo "[batch] done=$(date +%T)"
