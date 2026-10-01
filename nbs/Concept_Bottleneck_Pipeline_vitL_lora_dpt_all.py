import copy
import json
import os

import torch
from nb_setup import check_env_wandb
from torch.utils.data import ConcatDataset, DataLoader

from mermaidseg.dataset_reconciliation import (
    ConceptSchema,
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.datasets import (
    build_datasets,
    make_worker_init_fn,
    setup_local_cache,
)
from mermaidseg.io import get_parser, setup_config, update_config_with_args
from mermaidseg.logger import Logger
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.models import align_peft_checkpoint_state_dict
from mermaidseg.model.train import train_model

# ViT-L encoder adapted with LoRA + a DPT segmentation head (concept-bottleneck variant).
VITL_ENCODER_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CHECKPOINT = None#"model_checkpoints/mermaid_base_run_dinov3_lora_dpt_all/model_epoch34" # "model_checkpoints/mermaid_base_run_dinov3_lora_dpt/model_epoch13"


def load_training_checkpoint(
    meta_model: MetaModel,
    checkpoint_path: str,
    device: torch.device | str,
    *,
    load_scheduler: bool = True,
) -> int:
    """Load model (+ optimizer, and scheduler when resuming) and return the next epoch.

    When ``load_scheduler`` is False, optimizer state is restored but the epoch counter,
    ``global_step``, and scheduler are left for the caller to restart (cooldown / a new
    run initialized from this checkpoint). Call ``meta_model.rebuild_scheduler()`` after
    this so the new schedule binds to the restored learning rates.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path!r} has no model_state_dict")
    state_dict = align_peft_checkpoint_state_dict(state_dict, meta_model.model)
    meta_model.model.load_state_dict(state_dict)
    del state_dict

    # Restore optimizer state. torch.load(map_location=device) already placed the
    # state tensors on `device`, and the model was loaded in place (same Parameter
    # objects the optimizer references), so the param<->state mapping stays valid.
    if "optimizer_state_dict" in checkpoint:
        meta_model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if not load_scheduler:
        return 0

    # Restore the LR scheduler so an epoch schedule continues from the correct step
    # instead of restarting its decay. `scheduler` may be None when no scheduler was
    # configured.
    scheduler = getattr(meta_model, "scheduler", None)
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # The iteration-based LinearLR warmup (warmup_iters < iterations_per_train_epoch)
    # always completes within epoch 0, so any resumed checkpoint is already past
    # warmup. `_warmup_iters_completed` is not persisted and resets to 0 in a fresh
    # process; without this, train.py would re-run warmup and overwrite the restored
    # learning rates with the initial (pre-decay) base LRs. Mark warmup complete so
    # the restored optimizer/scheduler LRs are respected.
    if getattr(meta_model, "warmup_iters", 0) > 0:
        meta_model._warmup_iters_completed = meta_model.warmup_iters

    if "global_step" in checkpoint:
        meta_model.global_step = int(checkpoint["global_step"])

    return int(checkpoint.get("epoch", -1)) + 1


# -- 0. Environment --------------------------------------------------------
# wandb-backed, AWS-free run: data is served from the pre-warmed local cache
# (see scripts/prefetch_cache.py) and metrics go to wandb. No MLflow / SageMaker
# / AWS SSO is required. Set MERMAIDSEG_S3_OFFLINE=1 to fail loudly on any cache
# miss instead of silently falling back to S3.
check_env_wandb()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(torch.cuda.device_count()):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

# Match the SLURM CPU allocation when running under Slurm (falls back to 50 for
# local runs). MERMAIDSEG_NUM_WORKERS / MERMAIDSEG_PREFETCH_FACTOR override both
# (the 1M-step job uses 56 workers and prefetch 2 on 64 CPUs).
NUM_WORKERS = int(
    os.environ.get("MERMAIDSEG_NUM_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "50"))
)
PREFETCH_FACTOR = int(os.environ.get("MERMAIDSEG_PREFETCH_FACTOR", "3"))
PERSISTENT_WORKERS = NUM_WORKERS > 0

# -- 1. Config -------------------------------------------------------------
# Optional CLI overrides so this script can be driven by a hyperparameter sweep
# (see slurm/sweep_loss_weights_train.sbatch). With no extra args the behavior is
# identical to the previous hardcoded run.
parser = get_parser()
parser.add_argument(
    "--training-config",
    default="../configs/training_config_cbm.yaml",
    help="path to the training config YAML",
)
parser.add_argument("--per-pixel-loss-weight", type=float, default=None)
parser.add_argument("--per-image-loss-weight", type=float, default=None)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="path to a checkpoint to resume from (overrides the CHECKPOINT constant)",
)
parser.add_argument(
    "--init-checkpoint",
    default=None,
    help=(
        "path to a checkpoint whose model and optimizer initialize a new run "
        "(epoch 0, fresh scheduler). Mutually exclusive with --checkpoint"
    ),
)
parser.add_argument(
    "--data-seed",
    type=int,
    default=None,
    help="torch seed for data shuffling (default: 4). Cooldown jobs pass a different seed",
)
parser.set_defaults(run_name="mermaid_base_run_dinov3_lora_dpt_all")
args = parser.parse_args()
if args.checkpoint and args.init_checkpoint:
    parser.error("pass only one of --checkpoint and --init-checkpoint")

SEED = 4 if args.data_seed is None else args.data_seed
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# CLI --checkpoint (e.g. from the sweep's checkpoint auto-discovery) takes
# precedence over the hardcoded CHECKPOINT constant so runs can resume.
if args.checkpoint:
    CHECKPOINT = args.checkpoint

cfg = setup_config(
    {
        "data": "../configs/data_config_all.yaml",
        "training": args.training_config,
        "model": "../configs/model_config_cbm_dpt_lora_vitl.yaml",
        "logger": "../configs/logger_config.yaml",
    }
)
cfg = update_config_with_args(cfg, args)

# Loss-weight sweep overrides (leave the config value untouched when not given).
if args.per_pixel_loss_weight is not None:
    cfg.training.loss.per_pixel_loss_weight = args.per_pixel_loss_weight
if args.per_image_loss_weight is not None:
    cfg.training.loss.per_image_loss_weight = args.per_image_loss_weight

# The LoRA/DPT model config already targets the ViT-L encoder; set it explicitly
# so the value is unambiguous and gets logged below.
cfg.model.encoder_name = VITL_ENCODER_NAME

# Iterations / batch_size / epochs are set in configs/training_config_cbm.yaml.

# Set experiment on the config the Logger actually reads.
cfg_logger = copy.deepcopy(cfg)
cfg_logger.logger.experiment_name = "mermaid"

# -- 2. Datasets -----------------------------------------------------------
cache_stats = setup_local_cache(cfg.data)

# Shared build loop (also used by scripts/prefetch_cache.py) so the training job
# and the cache-prefetch job always agree on which source objects are needed.
dataset_dict = build_datasets(cfg.data, padding=cfg.training.padding)

loader_kwargs = {
    "batch_size": cfg.training.batch_size,
    "num_workers": NUM_WORKERS,
    "pin_memory": True,
    "persistent_workers": PERSISTENT_WORKERS,
    "drop_last": True,
    "prefetch_factor": PREFETCH_FACTOR,
}
if NUM_WORKERS > 0:
    loader_kwargs["worker_init_fn"] = make_worker_init_fn(cache_stats)

concept_mapping_path = (
    cfg.training.get("concept_mapping_path")
    or "/data/vision/beery/scratch/sauder/mermaid-segmentation/configs/class_to_concepts.csv"
)

# -- 3. Registry / model / evaluator --------------------------------------
_, registry_datasets = prepare_splits_for_registry(dataset_dict)

run_sources = {ds.SOURCE_NAME for ds in registry_datasets}
schema = ConceptSchema.from_csv(concept_mapping_path, sources=run_sources)

registry = SourceLabelRegistry(
    registry_datasets,
    target_label_subset=cfg.training.class_subset,
    compute_concepts=cfg.training.training_mode != "standard",
    concept_mapping_path=concept_mapping_path,
    concept_schema=schema,
    label_roll_up=cfg.training.get("label_roll_up", False),
).to(device)

attach_registry(registry, dataset_dict.values())

# repeat coralscapes_v2 train dataset 5 times
train_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "train"] + [
    ds for (name, split), ds in dataset_dict.items() if name == "coralscapes_v2" and split == "train"
] * 5
val_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "val"]

train_loader = DataLoader(ConcatDataset(train_datasets), shuffle=True, **loader_kwargs)
loader_kwargs["prefetch_factor"] = 2
val_loader = DataLoader(ConcatDataset(val_datasets), shuffle=True, **loader_kwargs)

print(f"train batches: {len(train_loader)}   val batches: {len(val_loader)}")
assert registry.num_concepts == schema.num_channels
concept_id2name = schema.channel_id2name()
with open("concept_id2name.json", "w") as f:
    json.dump({str(k): v for k, v in concept_id2name.items()}, f)

meta_model = MetaModel(
    run_name=cfg.run_name,
    num_classes=registry.num_target_classes,
    num_concepts=registry.num_concepts or None,
    device=device,
    model_kwargs=cfg.model.copy(),
    training_kwargs=cfg.training.copy(),
    source_to_target_lookup=registry.source_to_target,
    source_to_concepts_lookup=registry.source_to_concepts,
    concept_matrix=registry.concept_matrix,
    conceptid2labelid=registry.conceptid2labelid(),
    concept_value2id=registry.concept_value2id,
)

start_epoch = 0
init_checkpoint = args.init_checkpoint
if init_checkpoint:
    load_training_checkpoint(meta_model, init_checkpoint, device, load_scheduler=False)
    # Bind a fresh schedule to the restored peak LRs (do not continue the
    # source run's scheduler). Epoch and global_step stay at 0.
    meta_model.rebuild_scheduler()
    print(f"Loaded init checkpoint {init_checkpoint!r}; starting a new run at epoch 0")
    # Re-seed after model init so the shuffle stream is exactly `SEED`, not that
    # seed advanced by construction. Augmentations stay entropy-seeded.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Re-seeded RNG to {SEED} for the new run's data order")
elif CHECKPOINT:
    start_epoch = load_training_checkpoint(meta_model, CHECKPOINT, device)
    print(f"Loaded checkpoint {CHECKPOINT!r}; resuming at epoch {start_epoch}")
    # Offset the RNG by the number of already-trained epochs so the resumed run
    # sees a different data-shuffling stream instead of replaying the epoch
    # 0..N-1 orderings. The DataLoader's RandomSampler draws from the global
    # torch generator (re-seeded to SEED at startup), and the training loop is
    # iterated below, so re-seeding here changes the permutations for every
    # resumed epoch. (Augmentations already vary per run via Albumentations'
    # own entropy-seeded RNG.)
    resume_seed = SEED + start_epoch
    torch.manual_seed(resume_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resume_seed)
    print(f"Re-seeded RNG to {resume_seed} (SEED={SEED} + {start_epoch} trained epochs)")

evaluator = Evaluator(
    num_classes=registry.num_target_classes,
    device=device,
    calculate_concept_metrics=cfg.training.training_mode != "standard",
    concept_value2id=registry.concept_value2id,
)
# -- 5. Train (run lifecycle managed by `with`) ---------------------------
# Backend selection is config-driven (configs/logger_config.yaml sets
# enable_mlflow: false, enable_wandb: true).
with Logger(
    config=cfg_logger,
    meta_model=meta_model,
    log_epochs=cfg_logger.logger.get("log_epochs", 1),
    log_checkpoint=1,
    checkpoint_dir=".",
    id2label={0: "ignore", **registry.target_id2label},
    save_local_checkpoints=True,
) as run_logger:
    assert run_logger.enable_wandb, (
        "wandb logging was not started — check WANDB_API_KEY and "
        "config.logger.enable_wandb (see warnings above)"
    )
    run_logger.log_dict(
        {str(k): v for k, v in concept_id2name.items()},
        "metadata/concept_id2name.json",
    )
    run_logger.log_params(
        {
            "model/encoder_name": VITL_ENCODER_NAME,
            "model/head": "dpt",
            "model/adapter": "lora",
            "loss/per_pixel_loss_weight": cfg.training.loss.per_pixel_loss_weight,
            "loss/per_image_loss_weight": cfg.training.loss.per_image_loss_weight,
        }
    )
    if init_checkpoint:
        run_logger.log_params({"init/checkpoint": init_checkpoint, "init/start_epoch": 0})
    elif CHECKPOINT:
        run_logger.log_params({"init/checkpoint": CHECKPOINT, "init/start_epoch": start_epoch})
    print(f"\nwandb run_id : {run_logger.wandb_run_id}")
    print(f"wandb run URL: {run_logger.run_url}\n")

    # Reuse the logger helpers instead of hand-rolling concept extraction.
    run_logger.log_dataloader_params(train_loader, prefix="train_loader")
    run_logger.log_dataloader_params(val_loader, prefix="val_loader")
    run_logger.log_reconciliation(registry)  # writes metadata/concept_id2name.json

    train_size = sum(len(d) for d in train_datasets)
    val_size = sum(len(d) for d in val_datasets)
    run_logger.log_params(
        {
            "data/train_size": train_size,
            "data/val_size": val_size,
            "data/total_size": sum(len(d) for d in dataset_dict.values()),
            "data/seed": SEED,
            "data/dataset_name": "ALL",
        }
    )

    metrics_all: dict[int, dict] = {}

    for epoch in range(start_epoch, cfg.training.epochs):
        metrics = train_model(
        meta_model=meta_model,
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=run_logger,
        start_epoch=epoch,
        end_epoch=epoch + 1,
        metric_of_interest="accuracy",
        train_log_interval=cfg.logger.get("train_log_interval", 1),
        )
        metrics_all.update(metrics)
        if cfg.training.iterations_per_train_epoch>1000:
            run_logger.save_model_checkpoint(
            meta_model,
            epoch,
            metrics[epoch].get("validation_metrics", {}),
            is_best=False,
        )
    final_epoch = max(metrics)
    print("Final train metrics     :", metrics[final_epoch].get("train_metrics"))
    print("Final validation metrics:", metrics[final_epoch].get("validation_metrics"))
    print(f"wandb run URL: {run_logger.run_url}")
