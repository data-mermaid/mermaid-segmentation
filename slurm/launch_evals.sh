#!/bin/bash
# Unified eval launcher: one slurm/eval.sbatch job per (run, checkpoint, resolution),
# written to eval_out/<family>/<run>/<checkpoint>_res<R>/.
#
#   lossw     cbm_lossw_pix<P>_img<I>          model_step100000 at 512
#   taxloss   cbm_tax_{kl,partial,tree}        model_step100000 at 512
#   datasize  cbm_dsize_exp<N>_drop<F>         model_step100000 at 512
#   sparse    cbm_sparse                       model_step100000 at 512
#   vitsize   cbm_vitsize_{vits,vitb,vithplus} model_step100000 at 512
#   duration  cbm_duration_cd<STEP>            model_step25000 (end of each cooldown) at 512,
#             cbm_duration_peak                model_epoch<E> at the same branch steps, at 512
#   hires     cbm_hires_res<R>                 model_step25000 at 256, 384, 512, 768 and 1024
#
# An eval that is done (OUT/.done), queued, or whose checkpoint does not exist yet
# is skipped, so rerun the launcher as runs finish. Then compile everything with
# scripts/compile_evals.py.
#
#   slurm/launch_evals.sh [--dry-run] [FAMILY...]

USAGE="usage: slurm/launch_evals.sh [--dry-run] [lossw|taxloss|datasize|sparse|vitsize|duration|hires ...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

VITL=configs/model_config_cbm_dpt_lora_vitl.yaml
n_done=0
n_queued=0
n_missing=0

# eval_cell FAMILY RUN CHECKPOINT MODEL_CONFIG RES
eval_cell() {
    local family="$1" run="$2" ckpt="$3" model="$4" res="$5"
    local src="$CKPT_ROOT/$run/$ckpt"
    local out="$EVAL_ROOT/$family/$run/${ckpt}_res${res}"
    local job="eval-${run}-${ckpt}-res${res}"
    if [[ -e "$out/.done" ]]; then
        n_done=$((n_done + 1))
    elif is_queued "$job"; then
        n_queued=$((n_queued + 1))
    elif ! checkpoint_ready "$src"; then
        n_missing=$((n_missing + 1))
    else
        echo "launch  : $out"
        sbatch_submit --job-name="$job" slurm/eval.sbatch "$src" "$model" "$res" "$out"
    fi
}

if wanted lossw; then
    PIX=(1.0 0.9 0.75 0.5 0.25 0.1 0.0)
    IMG=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)
    for i in "${!PIX[@]}"; do
        eval_cell lossw "cbm_lossw_pix${PIX[$i]}_img${IMG[$i]}" model_step100000 "$VITL" 512
    done
fi

if wanted taxloss; then
    for loss in kl partial tree; do
        eval_cell taxloss "cbm_tax_${loss}" model_step100000 "$VITL" 512
    done
fi

if wanted datasize; then
    for profile in exp1_drop0 exp1_drop10 exp1_drop20 exp1_drop50 \
        exp2_drop10 exp2_drop20 exp2_drop50 \
        exp3_drop10 exp3_drop20 exp3_drop50 \
        exp4_drop10 exp4_drop20 exp4_drop50; do
        eval_cell datasize "cbm_dsize_${profile}" model_step100000 "$VITL" 512
    done
fi

if wanted sparse; then
    eval_cell sparse cbm_sparse model_step100000 "$VITL" 512
fi

if wanted vitsize; then
    for size in vits vitb vithplus; do
        eval_cell vitsize "cbm_vitsize_${size}" model_step100000 \
            "configs/model_config_cbm_dpt_lora_${size}.yaml" 512
    done
fi

if wanted duration; then
    # Must match slurm/launch_duration.sh.
    ITERS_PER_EPOCH=25000
    for step in 25000 75000 125000 175000 275000 375000 475000 575000 675000 775000 875000 975000; do
        eval_cell duration "cbm_duration_cd${step}" model_step25000 "$VITL" 512
        eval_cell duration cbm_duration_peak "model_epoch$((step / ITERS_PER_EPOCH - 1))" "$VITL" 512
    done
fi

if wanted hires; then
    for train_res in 256 384 512 768; do
        for res in 256 384 512 768 1024; do
            eval_cell hires "cbm_hires_res${train_res}" model_step25000 "$VITL" "$res"
        done
    done
fi

echo
echo "done: $n_done  queued: $n_queued  checkpoint not ready: $n_missing"
launcher_summary
