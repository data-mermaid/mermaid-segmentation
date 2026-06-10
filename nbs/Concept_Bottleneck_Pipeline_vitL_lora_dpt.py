import copy
import os
import re
import urllib.parse

import mlflow
import torch
from torch.utils.data import ConcatDataset, DataLoader

from mermaidseg.dataset_reconciliation import (
    ConceptSchema,
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.datasets import (
    BenthosYuvalCoralsDataset,
    CatlinSeaviewDataset,
    CoralNetDataset,
    CoralscapesDataset,
    MermaidDataset,
    MooreaLabeledCoralsDataset,
    PacificLabeledCoralsDataset,
    make_worker_init_fn,
    setup_local_cache,
)
from mermaidseg.io import get_parser, setup_config, update_config_with_args
from mermaidseg.logger import Logger
from mermaidseg.model.eval import Evaluator

from mermaidseg.model.meta import MetaModel
from mermaidseg.model.train import train_model

from nb_setup import check_aws_session, check_env, check_mlflow_version

# ViT-L encoder adapted with LoRA + a DPT segmentation head (concept-bottleneck variant).
VITL_ENCODER_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"


# -- 0. Environment --------------------------------------------------------
if not os.getenv("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = (
        "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-3546X3USYJNZ"
    )

check_env()
check_aws_session()
check_mlflow_version()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(torch.cuda.device_count()):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

SEED = 3
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NUM_WORKERS = 6
PERSISTENT_WORKERS = NUM_WORKERS > 0

# -- 1. Config -------------------------------------------------------------
cfg = setup_config(
    {
        "data": "../configs/data_config.yaml",
        "training": "../configs/training_config_cbm.yaml",
        "model": "../configs/model_config_cbm_dpt_lora_vitl.yaml",
        "logger": "../configs/logger_config.yaml",
    }
)
args = get_parser().parse_args("--run-name=mermaid_base_run_dinov3_lora_dpt".split())
cfg = update_config_with_args(cfg, args)

# The LoRA/DPT model config already targets the ViT-L encoder; set it explicitly
# so the value is unambiguous and gets logged below.
cfg.model.encoder_name = VITL_ENCODER_NAME

# Hyperparameters for this run
cfg.training.iterations_per_train_epoch = 4000
cfg.training.iterations_per_val_epoch = 400  # None => use full val set (len(val_loader))
cfg.training.batch_size = 12

# Set experiment on the config the Logger actually reads.
cfg_logger = copy.deepcopy(cfg)
cfg_logger.logger.experiment_name = "mermaid"

# -- 2. Datasets -----------------------------------------------------------
cache_stats = setup_local_cache(cfg.data)

DATASET_CLASSES = {
    "pacific_labeled_corals": PacificLabeledCoralsDataset,
    "moorea_labeled_corals": MooreaLabeledCoralsDataset,
    "catlin_seaview": CatlinSeaviewDataset,
    "mermaid": MermaidDataset,
    "coralnet": CoralNetDataset,
    "coralscapes": CoralscapesDataset,
    "benthos_yuval": BenthosYuvalCoralsDataset,
}

# coralscapes uses a different signature (no `padding`)
def _build(name, split_cfg):
    cls = DATASET_CLASSES[name]
    if name == "coralscapes":
        return cls(**split_cfg)
    return cls(**split_cfg, padding=cfg.training.padding)

dataset_dict: dict[tuple[str, str], object] = {}
for name in DATASET_CLASSES:
    for split, split_cfg in cfg.data[name].items():
        # data_config.yaml uses literal `None` which PyYAML reads as the
        # string "None"; treat both as "skip this split".
        if split_cfg is None or split_cfg == "None":
            continue
        dataset_dict[(name, split)] = _build(name, split_cfg)
        print(f"{name:>24s} - {split:<5s}: {len(dataset_dict[(name, split)]):>7d} samples")

import json

loader_kwargs = dict(
    batch_size=cfg.training.batch_size,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=PERSISTENT_WORKERS,
    drop_last=True,
)
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

train_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "train"]
val_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "val"]

train_loader = DataLoader(ConcatDataset(train_datasets), shuffle=True, **loader_kwargs)
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

evaluator = Evaluator(
    num_classes=registry.num_target_classes,
    device=device,
    calculate_concept_metrics=cfg.training.training_mode != "standard",
    concept_value2id=registry.concept_value2id,
)
# -- 4. Build a clickable URL ---------------------------------------------
def mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Best-effort clickable link for a run.

    - http(s) tracking server: real deep link into the MLflow UI.
    - SageMaker MLflow App ARN: link to the SageMaker console page for the
      app; the actual MLflow UI is opened from there inside Studio.
    - local file store: just the file path.
    """
    if tracking_uri.startswith(("http://", "https://")):
        return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"

    m = re.match(r"arn:aws:sagemaker:([^:]+):\d+:mlflow-app/(.+)$", tracking_uri)
    if m:
        region, app_id = m.group(1), m.group(2)
        return (
            f"https://{region}.console.aws.amazon.com/sagemaker/home"
            f"?region={region}#/mlflow/{app_id}"
            f"  (then open run_id={run_id}, experiment_id={experiment_id})"
        )
    return f"file://{urllib.parse.quote(tracking_uri)}/#/experiments/{experiment_id}/runs/{run_id}"

# -- 5. Train (run lifecycle managed by `with`) ---------------------------
with Logger(
    config=cfg_logger,
    meta_model=meta_model,
    log_epochs=cfg_logger.logger.get("log_epochs", 1),
    log_checkpoint=1,
    checkpoint_dir=".",
    enable_mlflow=True,
    id2label={0: "background", **registry.target_id2label},
    save_local_checkpoints=True,
) as run_logger:
    run = mlflow.active_run()
    assert run is not None, "MLflow run was not started — check MLFLOW_TRACKING_URI and warnings above"
    if mlflow.active_run() is not None:
        mlflow.log_dict(
            {str(k): v for k, v in concept_id2name.items()},
            "metadata/concept_id2name.json",
        )
        mlflow.log_param("model/encoder_name", VITL_ENCODER_NAME)
        mlflow.log_param("model/head", "dpt")
        mlflow.log_param("model/adapter", "lora")
    run_id = run.info.run_id
    exp_id = run.info.experiment_id
    tracking_uri = mlflow.get_tracking_uri()
    print(f"\nMLflow run_id        : {run_id}")
    print(f"MLflow experiment_id : {exp_id}")
    print(f"MLflow tracking URI  : {tracking_uri}")
    print(f"MLflow run URL       : {mlflow_run_url(tracking_uri, exp_id, run_id)}\n")

    # Reuse the logger helpers instead of hand-rolling concept extraction.
    run_logger.log_dataloader_params(train_loader, prefix="train_loader")
    run_logger.log_dataloader_params(val_loader, prefix="val_loader")
    run_logger.log_reconciliation(registry)  # writes metadata/concept_id2name.json

    train_size = sum(len(d) for d in train_datasets)
    val_size = sum(len(d) for d in val_datasets)
    mlflow.log_params(
        {
            "data/train_size": train_size,
            "data/val_size": val_size,
            "data/total_size": sum(len(d) for d in dataset_dict.values()),
            "data/seed": SEED,
            "data/dataset_name": "ALL",
        }
    )

    metrics_all: dict[int, dict] = {}

    for epoch in range(cfg.training.epochs):
        metrics = train_model(
        meta_model=meta_model,
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=run_logger,
        start_epoch=epoch,
        end_epoch=epoch + 1,
        metric_of_interest="accuracy",
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
    print(f"MLflow run URL       : {mlflow_run_url(tracking_uri, exp_id, run_id)}")
