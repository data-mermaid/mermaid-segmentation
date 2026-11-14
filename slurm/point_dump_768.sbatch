#!/bin/bash
#SBATCH --job-name=point-dump-768
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.err

# Per-point prediction dump on the full CoralNet test split (144 sources) for error analysis, with the 768 px
# diversity model (cbm_cd275000_res768, step 50k, KL w=0.5, taxv2), predicted at 768 px. Same model, concept table
# and settings as the taxon-diversity runs (slurm/taxon_diversity_conf_768.sbatch). Batch size 2: 8 runs out of
# GPU memory at 768 px. Writes results_tmp/cbm_cd275000_res768_step50000/test/ (see
# mermaidseg/evaluation/point_dump.py; analyse with nbs/Test_Point_Analysis.ipynb).
# Usage: sbatch slurm/point_dump_768.sbatch
set -euo pipefail

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

OUT=results_tmp/cbm_cd275000_res768_step50000/test
mkdir -p "$OUT"
echo "[batch] node=$(hostname) job=${SLURM_JOB_ID} out=${OUT} start=$(date +%T)"
.venv/bin/python scripts/dump_point_predictions.py \
    --checkpoint "$CKPT" \
    --model-config configs/model_config_cbm_dpt_lora_vitl_768.yaml \
    --concept-id2name "eval_out/scratch100k_hier/$TAXV2/concept_id2name.json" \
    --taxonomy-csv "nbs/model_checkpoints/$TAXV2/class_to_concepts.csv" \
    --data-config configs/data_config_dedup.yaml --split test \
    --benthic-hierarchy-json eval_out/scratch100k_hier/benthic_hierarchy.json \
    --batch-size 2 --num-workers 16 --amp \
    --output-dir "$OUT"
echo "[batch] done=$(date +%T)"
