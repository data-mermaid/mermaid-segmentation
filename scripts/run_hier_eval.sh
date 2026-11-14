#!/bin/bash
# Full evaluation (CoralNet + Pacific val with --hierarchical, Benthos, CoralscapesV2 probe) of a run's step checkpoints
# at prediction resolution 256, with the same inputs as the baseline evals (concept_id2name,
# id2label, data config, cached benthic hierarchy). Finished cells (.done) are skipped, so it can be
# re-run as more checkpoints appear. Run it interactively on a GPU node or wrap it in sbatch.
#
#   bash scripts/run_hier_eval.sh RUN [CKPT_DIR]
#
# RUN       run name, e.g. cbm_ft50k_distance (outputs: eval_out/ft50k_hier/RUN/)
# CKPT_DIR  directory with model_step* files (default: nbs/model_checkpoints/RUN)
#
#   bash scripts/run_hier_eval.sh cbm_ft50k_distance
#   bash scripts/run_hier_eval.sh cbm_ft50k_kl
#   bash scripts/run_hier_eval.sh cbm_ft50k_baseline \
#       /data/vision/beery/scratch/sauder/mermaid-segmentation/nbs/model_checkpoints/cbm_hires_res256_mb32
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?usage: bash scripts/run_hier_eval.sh RUN [CKPT_DIR]}"
CKPT_DIR="${2:-nbs/model_checkpoints/$RUN}"
RES=256
OUT_BASE=eval_out/ft50k_hier/$RUN
HIERARCHY=/data/vision/beery/scratch/sauder/mermaid-segmentation/eval_out/hires/cbm_hires_res256_mb32/step50000_res256/benthic_hierarchy.json
MIN_AGE_SEC=180  # skip checkpoints written in the last 3 min (torch.save may still be running)

set -a; . ./.env; set +a

shopt -s nullglob
ckpts=("$CKPT_DIR"/model_step*)
shopt -u nullglob
[[ ${#ckpts[@]} -gt 0 ]] || { echo "no model_step* checkpoints in $CKPT_DIR"; exit 0; }

now=$(date +%s)
for ckpt in $(printf '%s\n' "${ckpts[@]}" | sort -V); do
    step="${ckpt##*/model_step}"
    out=$OUT_BASE/step${step}_res${RES}
    if [[ -f $out/.done ]]; then
        echo "skip  step$step (done)"
        continue
    fi
    if (( now - $(stat -c %Y "$ckpt") < MIN_AGE_SEC )); then
        echo "fresh step$step (still being written?) - skipping for now"
        continue
    fi
    mkdir -p "$out"
    echo "start step$step $(date +%T)"
    if .venv/bin/python scripts/evaluate.py \
        --checkpoint "$ckpt" \
        --model-config configs/model_config_cbm_dpt_lora_vitl_${RES}.yaml \
        --data-config configs/data_config_all.yaml \
        --id2label demo/id2label.json \
        --concept-id2name demo/concept_id2name.json \
        --benthic-hierarchy-json "$HIERARCHY" \
        --output-dir "$out" \
        --evals all \
        --batch-size 8 --num-workers 16 --amp --hierarchical \
        > "$out/eval.log" 2>&1; then
        touch "$out/.done"
        echo "done  step$step $(date +%T)"
    else
        echo "FAIL  step$step $(date +%T) (see $out/eval.log)"
    fi
done
echo "finished $RUN $(date +%T)"
