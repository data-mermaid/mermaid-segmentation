#!/bin/bash
#SBATCH --job-name=scratch100k
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --partition=vision-beery-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=71:59:00
#SBATCH --output=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/viktordo/mermaid-segmentation/logs/slurm-%x-%j.err

# 100k-update run from scratch (DINOv3 ViT-L + fresh LoRA / DPT / concept heads) on the
# deduplicated CoralNet splits (configs/data_config_dedup.yaml), with Jonathan's duration recipe
# at 1/10 the length, in one job:
#   phase 1: 75k updates, 4k warmup then constant peak LR (training_config_cbm_scratch100k.yaml),
#            model_epoch0..2 in model_checkpoints/<RUN>;
#   phase 2: 25k-update linear cooldown to 0 from phase 1's model_epoch2 (model + optimizer,
#            data seed 1004; training_config_cbm_scratch100k_cooldown.yaml), in <RUN>_cd.
# 256 px, microbatch 16 x 2 accumulation (effective batch 32), the same for every run.
#
# LOSS: base (default loss only) | partial | kl | distance | klpartial; WEIGHT: the added term's weight
# (taxonomic_loss.partial_weight / taxonomic_loss.kl_weight / distance_loss.weight; constant from
# update 0, no ramp). The run is cbm_scratch100k_base or cbm_scratch100k_<LOSS>_w<WEIGHT>.
# klpartial takes two weights, KL then partial CE, and is cbm_scratch100k_klpartial_w<KL>_<PARTIAL>.
# An optional trailing tag=NAME appends _NAME to the run, e.g. to rerun after a concept CSV change.
#
# The concept CSV is snapshotted into <RUN>/class_to_concepts.csv on first start and both phases
# read the snapshot, so editing configs/class_to_concepts.csv mid-run cannot change the channels.
#
# Usage (from the repo root, on the login node):
#   sbatch --job-name=s100k-base slurm/scratch100k_train.sbatch base
#   sbatch --job-name=s100k-kl-0.1 slurm/scratch100k_train.sbatch kl 0.1
#   sbatch --job-name=s100k-base-taxv2 slurm/scratch100k_train.sbatch base tag=taxv2
#   sbatch --job-name=s100k-klpartial-taxv2 slurm/scratch100k_train.sbatch klpartial 0.5 0.1 tag=taxv2
# Resubmitting the same command resumes: phase 1 from its latest model_epoch*, phase 2 restarts
# (it is ~7 h); a finished run exits immediately.
# NOTE: pass LOSS / WEIGHT as positional arguments, not via --export (see hires_eval.sbatch).

set -euo pipefail

TAG=""
ARGS=()
for arg in "$@"; do
    case "$arg" in
        tag=*) TAG="${arg#tag=}" ;;
        *) ARGS+=("$arg") ;;
    esac
done
LOSS="${ARGS[0]:?usage: sbatch slurm/scratch100k_train.sbatch LOSS [WEIGHT [WEIGHT2]] [tag=NAME]}"
WEIGHT="${ARGS[1]:-}"
WEIGHT2="${ARGS[2]:-}"
[[ -z "$WEIGHT2" || "$LOSS" == klpartial ]] || { echo "only klpartial takes a second WEIGHT" >&2; exit 2; }

case "$LOSS" in
    base)
        [[ -z "$WEIGHT" ]] || { echo "base takes no WEIGHT" >&2; exit 2; }
        RUN=cbm_scratch100k_base
        LOSS_ARGS=() ;;
    partial)
        LOSS_ARGS=(--set "training.taxonomic_loss={ambiguity_policy: permissive, partial_weight: ${WEIGHT:?}, partial_uniform_anneal_iters: 0, kl_weight: 0.0}") ;;
    kl)
        LOSS_ARGS=(--set "training.taxonomic_loss={ambiguity_policy: permissive, partial_weight: 0.0, kl_weight: ${WEIGHT:?}, kl_start_iters: 0, kl_ramp_iters: 0, kl_stop_grad_coarse: true, kl_pixels: all_valid, kl_boundary_px: 0}") ;;
    distance)
        LOSS_ARGS=(--set "training.distance_loss={weight: ${WEIGHT:?}, ambiguity_policy: permissive}") ;;
    klpartial)
        RUN=cbm_scratch100k_klpartial_w${WEIGHT:?}_${WEIGHT2:?}
        LOSS_ARGS=(--set "training.taxonomic_loss={ambiguity_policy: permissive, partial_weight: ${WEIGHT2}, partial_uniform_anneal_iters: 0, kl_weight: ${WEIGHT}, kl_start_iters: 0, kl_ramp_iters: 0, kl_stop_grad_coarse: true, kl_pixels: all_valid, kl_boundary_px: 0}") ;;
    *)
        echo "unknown LOSS $LOSS (base | partial | kl | distance | klpartial)" >&2; exit 2 ;;
esac
RUN=${RUN:-cbm_scratch100k_${LOSS}_w${WEIGHT}}${TAG:+_${TAG}}
CD_RUN=${RUN}_cd

REPO=/data/vision/beery/scratch/viktordo/mermaid-segmentation
SCRATCH=/data/vision/beery/scratch/viktordo
MODEL_CONFIG="../configs/model_config_cbm_dpt_lora_vitl_256.yaml"
DATA_CONFIG="../configs/data_config_dedup.yaml"
DATA_SEED=4          # the trainer's default, as in the duration run
COOLDOWN_SEED=1004   # as in slurm/duration_cooldown.sbatch
LAST_EPOCH=2         # phase 1: 3 epochs of 25k updates

export MERMAIDSEG_NUM_WORKERS=28
export HOME=$SCRATCH/home
export HF_HOME=$SCRATCH/hf-cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export XDG_CONFIG_HOME=$HOME/.config
export TMPDIR=$SCRATCH/tmp
mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$REPO/logs" "$REPO/wandb"

set -a; source "$REPO/.env"; set +a
: "${WANDB_API_KEY:?add WANDB_API_KEY=... to $REPO/.env}"
: "${HF_TOKEN:?add HF_TOKEN=... to $REPO/.env}"
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION || true
export MERMAIDSEG_S3_OFFLINE=1
export WANDB_DIR=$REPO/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-mermaidseg}
export WANDB_RUN_GROUP=cbm_scratch100k
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=$REPO/.venv/bin/python
NBS_DIR=$REPO/nbs
CKPT_DIR=$NBS_DIR/model_checkpoints/$RUN
CD_DIR=$NBS_DIR/model_checkpoints/$CD_RUN

echo "[batch] node=$(hostname) job=${SLURM_JOB_ID} run=${RUN} loss=${LOSS} weight=${WEIGHT:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd "$NBS_DIR"

if [[ -e "$CD_DIR/model_epoch0" ]]; then
    echo "[batch] ${RUN} already finished: $CD_DIR/model_epoch0"
    exit 0
fi

CONCEPT_CSV=$CKPT_DIR/class_to_concepts.csv
if [[ ! -e "$CONCEPT_CSV" ]]; then
    mkdir -p "$CKPT_DIR"
    cp "$REPO/configs/class_to_concepts.csv" "$CONCEPT_CSV"
fi
echo "[batch] concept CSV: ${CONCEPT_CSV} (sha1 $(sha1sum "$CONCEPT_CSV" | cut -c1-12))"
LOSS_ARGS+=(--set "training.concept_mapping_path=${CONCEPT_CSV}")

# ---------- phase 1: warmup + constant LR, 75k updates ----------
if [[ ! -e "$CKPT_DIR/model_epoch${LAST_EPOCH}" ]]; then
    shopt -s nullglob
    ckpts=("$CKPT_DIR"/model_epoch*)
    shopt -u nullglob
    RESUME_ARGS=()
    if ((${#ckpts[@]})); then
        LATEST=$(printf '%s\n' "${ckpts[@]}" | sort -V | tail -1)
        echo "[batch] phase 1: resuming ${RUN} from ${LATEST}"
        RESUME_ARGS=(--checkpoint "$LATEST")
    else
        echo "[batch] phase 1: starting ${RUN} from scratch"
    fi
    "$PYTHON" Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py \
        --run-name "$RUN" \
        --training-config ../configs/training_config_cbm_scratch100k.yaml \
        --model-config "$MODEL_CONFIG" \
        --data-config "$DATA_CONFIG" \
        --per-pixel-loss-weight 0.5 \
        --per-image-loss-weight 0.5 \
        --data-seed "$DATA_SEED" \
        "${LOSS_ARGS[@]}" \
        "${RESUME_ARGS[@]}"
fi
[[ -e "$CKPT_DIR/model_epoch${LAST_EPOCH}" ]] || { echo "[batch] phase 1 left no model_epoch${LAST_EPOCH}" >&2; exit 1; }

# ---------- phase 2: 25k-update linear cooldown ----------
echo "[batch] phase 2: cooling down ${RUN}/model_epoch${LAST_EPOCH} into ${CD_RUN}"
"$PYTHON" Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py \
    --run-name "$CD_RUN" \
    --training-config ../configs/training_config_cbm_scratch100k_cooldown.yaml \
    --model-config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --per-pixel-loss-weight 0.5 \
    --per-image-loss-weight 0.5 \
    --data-seed "$COOLDOWN_SEED" \
    "${LOSS_ARGS[@]}" \
    --init-checkpoint "$CKPT_DIR/model_epoch${LAST_EPOCH}"

echo "[batch] ${RUN} finished; final model ${CD_DIR}/model_epoch0"
