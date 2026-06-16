"""Background-safe training entry point.

Mirrors the logic in nbs/Base_Pipeline.ipynb as a CLI script. Run via uv so
it uses the locked project venv. Cell outputs freeze when the browser
disconnects — this script writes structured logs to a file so progress is
always captured.

The run is configured by four split configs, each with a sensible default:
``--config-data`` (configs/data_config.yaml), ``--config-model``
(configs/model_config_cbm.yaml), ``--config-training``
(configs/training_config_cbm.yaml), and ``--config-logger``
(configs/logger_config.yaml). Override any one independently.

Usage::

    # Foreground (debug) — uses the default split configs
    uv run python scripts/train.py

    # Override an individual split (e.g. standard, non-CBM training)
    uv run python scripts/train.py \\
        --config-model configs/model_config.yaml \\
        --config-training configs/training_config.yaml

    # Background — safe to close the browser window
    nohup uv run python scripts/train.py \\
        --auto-shutdown \\
        > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo $! > logs/train.pid

    # Follow progress from any terminal tab
    tail -f logs/train_*.log

    # Dry run (1 batch, smoke test)
    uv run python scripts/train.py --dry-run
"""

import argparse
import copy
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

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
    CoralscapesV2Dataset,
    MermaidDataset,
    MooreaLabeledCoralsDataset,
    PacificLabeledCoralsDataset,
    worker_init_fn,
)
from mermaidseg.io import get_parser, setup_config, update_config_with_args
from mermaidseg.logger import Logger
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.metric_policy import SUPPORTED_METRIC_NAMES
from mermaidseg.model.train import train_model

_METADATA = Path("/opt/ml/metadata/resource-metadata.json")


def _configure_third_party_loggers() -> None:
    """Reduce verbosity from chatty third-party libraries."""
    logging.getLogger("botocore.tokens").setLevel(logging.WARNING)


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file),
        ],
    )
    _configure_third_party_loggers()
    logging.info("Logging to %s", log_file)


def _cleanup_pid(pid_file: Path) -> None:
    pid_file.unlink(missing_ok=True)


def _write_pid(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def _stop_current_space() -> None:
    """Stop the SageMaker JupyterLab app via delete_app.

    SageMaker has no ``stop_space`` API.  The documented way to stop a running
    JupyterLab app is ``delete_app`` — this terminates the instance while keeping EFS
    data intact.  No-op outside SageMaker.
    """
    if not _METADATA.exists():
        logging.info("Not running on SageMaker — skipping auto-shutdown")
        return
    import boto3
    import botocore.exceptions

    info = json.loads(_METADATA.read_text())
    try:
        boto3.client("sagemaker").delete_app(
            DomainId=info["DomainId"],
            SpaceName=info["SpaceName"],
            AppType="JupyterLab",
            AppName=info.get("AppName", "default"),
        )
        logging.info(
            "Space shutdown initiated (DomainId=%s, SpaceName=%s)",
            info["DomainId"],
            info["SpaceName"],
        )
    except botocore.exceptions.ClientError as e:
        logging.warning("Auto-shutdown failed (check sagemaker:DeleteApp permission): %s", e)


def _batch_is_non_empty(batch: object) -> bool:
    if not isinstance(batch, tuple | list) or len(batch) < 2:
        return True
    images, labels = batch[0], batch[1]
    if isinstance(images, torch.Tensor) and images.numel() == 0:
        return False
    return not (isinstance(labels, torch.Tensor) and labels.numel() == 0)


def _take_first_non_empty_batch(loader: object, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    for batch in loader:
        if _batch_is_non_empty(batch):
            return batch
    raise RuntimeError(
        f"Dry run could not find a non-empty batch for split '{split}'. Check data access credentials and dataset availability."
    )


def _save_failure_report_if_available(
    dataset: object,
    log_dir: Path,
    explicit_output_path: str | None = None,
) -> Path | None:
    num_failures = getattr(dataset, "num_load_failures", None)
    save_failures = getattr(dataset, "save_load_failures", None)
    if not callable(num_failures) or not callable(save_failures):
        return None
    n_failures = int(num_failures())
    if n_failures == 0:
        return None

    report_path = (
        Path(explicit_output_path)
        if explicit_output_path
        else (log_dir / f"data_load_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    )
    saved_path = Path(save_failures(report_path))
    logging.warning("Saved %d data-load failure records to %s", n_failures, saved_path)
    return saved_path


def _build_parser() -> argparse.ArgumentParser:
    base = get_parser()
    base.add_argument(
        "--config-data",
        type=str,
        default="configs/data_config.yaml",
        help="path to data config file",
    )
    base.add_argument(
        "--config-model",
        type=str,
        default="configs/model_config_cbm.yaml",
        help="path to model config file",
    )
    base.add_argument(
        "--config-training",
        type=str,
        default="configs/training_config_cbm.yaml",
        help="path to training config file",
    )
    base.add_argument(
        "--config-logger",
        type=str,
        default="configs/logger_config.yaml",
        help="path to logger config file",
    )
    base.add_argument(
        "--dry-run",
        action="store_true",
        help="run one batch only (smoke test)",
    )
    base.add_argument(
        "--auto-shutdown",
        action="store_true",
        help="stop the SageMaker space after a clean training finish",
    )
    base.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="directory for log files and PID file",
    )
    base.add_argument(
        "--failure-report-path",
        type=str,
        default=None,
        help="optional parquet path for data-load failure report (default: logs/data_load_failures_<timestamp>.parquet)",
    )
    base.add_argument(
        "--early-stopping",
        action="store_true",
        help="enable early stopping on validation metric_of_interest",
    )
    base.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="epochs without improvement before early stopping",
    )
    base.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="minimum improvement required to reset early stopping patience",
    )
    base.add_argument(
        "--metric-of-interest",
        type=str,
        default="accuracy",
        choices=sorted(SUPPORTED_METRIC_NAMES),
        help="metric used for checkpointing and early stopping",
    )
    base.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )
    base.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default: 0)",
    )
    return base


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    _setup_logging(log_dir)
    pid_file = log_dir / "train.pid"
    _write_pid(pid_file)

    def _handle_sigterm(_sig: int, _frame: object) -> None:
        logging.warning("SIGTERM received — exiting")
        _cleanup_pid(pid_file)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        _run_training(args)
    except Exception:
        logging.exception("Training failed")
        sys.exit(1)
    finally:
        _cleanup_pid(pid_file)

    if args.auto_shutdown:
        _SHUTDOWN_GRACE_SEC = 5
        logging.info("Waiting %ds for MLflow flush before shutdown...", _SHUTDOWN_GRACE_SEC)
        time.sleep(_SHUTDOWN_GRACE_SEC)
        _stop_current_space()


def _run_training(args: argparse.Namespace) -> None:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not os.getenv("MLFLOW_TRACKING_URI"):
        logging.warning(
            "MLFLOW_TRACKING_URI is not set — MLflow will log to a local filesystem store. "
            "Set MLFLOW_TRACKING_URI to the SageMaker MLflow App ARN for remote tracking:\n"
            '    export MLFLOW_TRACKING_URI="arn:aws:sagemaker:us-east-1:ACCOUNT:mlflow-app/APP-ID"'
        )

    cfg = setup_config(
        {
            "data": args.config_data,
            "training": args.config_training,
            "model": args.config_model,
            "logger": args.config_logger,
        }
    )

    cfg = update_config_with_args(cfg, args)
    cfg_logger = copy.deepcopy(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Seed: %d", seed)

    DATASET_CLASSES = {
        "pacific_labeled_corals": PacificLabeledCoralsDataset,
        "moorea_labeled_corals": MooreaLabeledCoralsDataset,
        "catlin_seaview": CatlinSeaviewDataset,
        "mermaid": MermaidDataset,
        "coralnet": CoralNetDataset,
        "coralscapes": CoralscapesDataset,
        "coralscapes_v2": CoralscapesV2Dataset,
        "benthos_yuval": BenthosYuvalCoralsDataset,
    }

    # coralscapes uses a different signature (no `padding`)
    def _build(name, split_cfg):
        cls = DATASET_CLASSES[name]
        if name in ("coralscapes", "coralscapes_v2", "benthos_yuval"):
            return cls(**split_cfg)
        return cls(**split_cfg, padding=cfg.training.padding)

    dataset_dict: dict[tuple[str, str], object] = {}
    for name in DATASET_CLASSES:
        for split, split_cfg in cfg.data[name].items():
            if split_cfg is None or split_cfg == "None":
                continue
            dataset_dict[(name, split)] = _build(name, split_cfg)
            print(f"{name:>24s} - {split:<5s}: {len(dataset_dict[(name, split)]):>7d} samples")

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = worker_init_fn

    concept_mapping_path = cfg.training.get("concept_mapping_path")

    _, registry_datasets = prepare_splits_for_registry(dataset_dict)

    run_sources = {ds.SOURCE_NAME for ds in registry_datasets}
    if cfg.training.training_mode != "standard":
        if concept_mapping_path:
            schema = ConceptSchema.from_csv(concept_mapping_path, sources=run_sources)
        else:
            schema = ConceptSchema.from_csv(sources=run_sources)
    else:
        schema = None

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
    if cfg.training.training_mode != "standard":
        assert registry.num_concepts == schema.num_channels

    logging.info(
        "Dataset: %s (%d samples)",
        "Combined",
        (len(train_loader) + len(val_loader)) * cfg.training.batch_size,
    )

    # def _write_failure_report_once() -> None:
    #     nonlocal report_written
    #     if report_written:
    #         return
    #     path = _save_failure_report_if_available(
    #         dataset=dataset, # TODO: Has to be updated to work with multiple datasets
    #         log_dir=Path(args.log_dir),
    #         explicit_output_path=args.failure_report_path,
    #     )
    #     report_written = path is not None

    if args.dry_run:
        max_dry_epochs = 3
        actual_epochs = cfg.training.epochs
        if actual_epochs > max_dry_epochs:
            cfg.training.epochs = max_dry_epochs
            logging.info("Dry run: capping epochs %d → %d", actual_epochs, max_dry_epochs)
        logging.info("Dry run: limiting to 1 batch per epoch")
        try:
            train_loader = [_take_first_non_empty_batch(train_loader, "train")]
            val_loader = [_take_first_non_empty_batch(val_loader, "val")]
        except RuntimeError:
            print("TODO:Update")
            # _write_failure_report_once()
            raise

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

    cfg.logger.experiment_name = "mermaid"
    cfg_logger.logger.experiment_name = "mermaid"

    with Logger(
        config=cfg_logger,
        meta_model=meta_model,
        log_epochs=cfg.logger.get("log_epochs", 1),
        log_checkpoint=cfg_logger.logger.get("log_checkpoint", 50),
        checkpoint_dir=".",
        enable_mlflow=True,
        id2label={0: "background", **registry.target_id2label},
    ) as logger:
        if logger.mlflow_run_id is not None:
            logging.info("MLflow run_id: %s", logger.mlflow_run_id)

        logger.log_dataloader_params(train_loader, prefix="train_loader")
        logger.log_dataloader_params(val_loader, prefix="val_loader")
        logger.log_reconciliation(registry)
        # TODO(#139): re-enable per-run dataset-statistics logging once
        # Logger.log_dataset_statistics is adapted to the multi-dataset (per-split lists)
        # structure; it currently expects single train/val/test datasets.

        try:
            # test_loader is None: the multi-dataset config does not define test splits yet,
            # so there is no test data to evaluate (see data_config.yaml).
            train_model(
                meta_model=meta_model,
                evaluator=evaluator,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,
                logger=logger,
                metric_of_interest=args.metric_of_interest,
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
        finally:
            print("TODO:Update")
            # _write_failure_report_once()
        logging.info("Training complete")


if __name__ == "__main__":
    main()
