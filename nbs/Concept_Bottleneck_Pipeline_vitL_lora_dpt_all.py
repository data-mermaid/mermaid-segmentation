import copy
import json

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
) -> int:
    """Load model (+ optimizer/scheduler when present) and return the next epoch index."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path!r} has no model_state_dict")
    state_dict = align_peft_checkpoint_state_dict(state_dict, meta_model.model)
    meta_model.model.load_state_dict(state_dict)

    #if "optimizer_state_dict" in checkpoint:
    #    meta_model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #if hasattr(meta_model, "scheduler") and "scheduler_state_dict" in checkpoint:
    #    meta_model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    del state_dict
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

SEED = 4
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NUM_WORKERS = 50
PERSISTENT_WORKERS = NUM_WORKERS > 0

# -- 1. Config -------------------------------------------------------------
cfg = setup_config(
    {
        "data": "../configs/data_config_all.yaml",
        "training": "../configs/training_config_cbm.yaml",
        "model": "../configs/model_config_cbm_dpt_lora_vitl.yaml",
        "logger": "../configs/logger_config.yaml",
    }
)
args = get_parser().parse_args(["--run-name=mermaid_base_run_dinov3_lora_dpt_all"])
cfg = update_config_with_args(cfg, args)

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
    "prefetch_factor": 3,
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
if CHECKPOINT:
    start_epoch = load_training_checkpoint(meta_model, CHECKPOINT, device)
    print(f"Loaded checkpoint {CHECKPOINT!r}; resuming at epoch {start_epoch}")

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
        }
    )
    if CHECKPOINT:
        run_logger.log_params(
            {"init/checkpoint": CHECKPOINT, "init/start_epoch": start_epoch}
        )
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
