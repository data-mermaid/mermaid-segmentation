#!/bin/bash
#SBATCH --job-name=s100k-eval
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.err

# Full eval at 256 px (CoralNet val + Pacific with --hierarchical, Benthos, CoralscapesV2 probe;
# ~27 min) of ONE scratch100k checkpoint. Same as scripts/run_hier_eval.sh except the concept map
# and hierarchy (below) and the data config: these runs were trained on the deduplicated splits, so CoralNet val comes from
# configs/data_config_dedup.yaml (data_config_all.yaml's val sources are partly in dedup train).
# Submit through slurm/scratch100k_launch_evals.sh, which picks CKPT / OUT for every checkpoint.
# Usage: sbatch [--job-name=...] slurm/scratch100k_eval.sbatch CKPT OUT [CONCEPTS TAXONOMY_CSV [RES]]
# CONCEPTS / TAXONOMY_CSV override the frozen 706-channel map and configs/class_to_concepts.csv for
# runs trained on another concept CSV (the *_taxv2 runs pin theirs in <RUN>/class_to_concepts.csv).
# RES (default 256) is the prediction resolution: 256 uses model_config_cbm_dpt_lora_vitl_256.yaml,
# 512 the base model_config_cbm_dpt_lora_vitl.yaml (~4x the tokens; pass a longer --time).
# NOTE: pass CKPT / OUT as positional arguments, not via --export (see hires_eval.sbatch).
set -euo pipefail
CKPT="${1:?usage: sbatch slurm/scratch100k_eval.sbatch CKPT OUT}"
OUT="${2:?usage: sbatch slurm/scratch100k_eval.sbatch CKPT OUT}"

REPO=/data/vision/beery/scratch/viktordo/mermaid-segmentation
SCRATCH=/data/vision/beery/scratch/viktordo
# Frozen copies, shared by every cell: the 706-channel concept map the scratch100k trainer writes
# (nbs/concept_id2name.json, demo/ + 'fish') and the MERMAID hierarchy fetched 2026-10-27
# (812 attributes; the hires_res256_mb32 cache run_hier_eval.sh points at is gone).
CONCEPTS="${3:-eval_out/scratch100k_hier/concept_id2name.json}"
TAXONOMY_CSV="${4:-configs/class_to_concepts.csv}"
RES="${5:-256}"
case "$RES" in
    256) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl_256.yaml ;;
    512) MODEL_CONFIG=configs/model_config_cbm_dpt_lora_vitl.yaml ;;
    *) echo "unsupported RES=$RES (256|512)" >&2; exit 2 ;;
esac
HIERARCHY=eval_out/scratch100k_hier/benthic_hierarchy.json
export HOME=$SCRATCH/home
export HF_HOME=$SCRATCH/hf-cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export TMPDIR=$SCRATCH/tmp
mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$REPO/logs"

cd "$REPO"
set -a; . ./.env; set +a

echo "[batch] node=$(hostname) job=${SLURM_JOB_ID} ckpt=${CKPT} out=${OUT} concepts=${CONCEPTS} csv=${TAXONOMY_CSV} res=${RES}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p "$OUT"
# .done is written only after a clean exit; the launcher skips cells that have it.
rm -f "$OUT/.done"
.venv/bin/python scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --model-config "$MODEL_CONFIG" \
    --data-config configs/data_config_dedup.yaml \
    --id2label demo/id2label.json \
    --concept-id2name "$CONCEPTS" \
    --taxonomy-csv "$TAXONOMY_CSV" \
    --benthic-hierarchy-json "$HIERARCHY" \
    --output-dir "$OUT" \
    --evals all \
    --batch-size 8 --num-workers 16 --amp --hierarchical \
    2>&1 | tee "$OUT/eval.log"
touch "$OUT/.done"
echo "[batch] eval complete -> $OUT/.done"
