"""Background-safe training entry point.

Mirrors the logic in nbs/Base_Pipeline.ipynb as a CLI script. Run via uv so
it uses the locked project venv. Cell outputs freeze when the browser
disconnects — this script writes structured logs to a file so progress is
always captured.

Usage::

    # Foreground (debug)
    uv run python scripts/train.py --config configs/linear-dinov3-base.yaml

    # Background — safe to close the browser window
    nohup uv run python scripts/train.py \\
        --config configs/linear-dinov3-base.yaml \\
        --auto-shutdown \\
        > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo $! > logs/train.pid

    # Follow progress from any terminal tab
    tail -f logs/train_*.log

    # Dry run (1 batch, smoke test)
    uv run python scripts/train.py --config configs/linear-dinov3-base.yaml --dry-run
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

import albumentations as A
import torch
from torch.utils.data import DataLoader, random_split

import mermaidseg.datasets.dataset
from mermaidseg.datasets.dataset import worker_init_fn
from mermaidseg.io import get_parser, setup_config, update_config_with_args
from mermaidseg.logger import Logger
from mermaidseg.model.eval import EvaluatorSemanticSegmentation
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

    SageMaker has no ``stop_space`` API.  The documented way to stop a
    running JupyterLab app is ``delete_app`` — this terminates the
    instance while keeping EFS data intact.  No-op outside SageMaker.
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
        "--config-base",
        type=str,
        default="configs/base_mermaid.yaml",
        help="path to base config file",
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
        config_path=args.config,
        config_base_path=args.config_base,
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

    transforms = {
        split: A.Compose([getattr(A, name)(**params) for name, params in augs.items()])
        for split, augs in cfg.augmentation.items()
    }

    dataset_name = cfg.data.pop("name", None)
    batch_size = cfg.data.pop("batch_size", 8)

    dataset = getattr(mermaidseg.datasets.dataset, dataset_name)(
        transform=transforms["train"], **cfg.data
    )
    logging.info("Dataset: %s (%d samples)", dataset_name, len(dataset))
    collate_fn = getattr(dataset, "collate_fn", None)
    report_written = False

    def _write_failure_report_once() -> None:
        nonlocal report_written
        if report_written:
            return
        path = _save_failure_report_if_available(
            dataset=dataset,
            log_dir=Path(args.log_dir),
            explicit_output_path=args.failure_report_path,
        )
        report_written = path is not None

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    num_workers = args.num_workers
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = worker_init_fn

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)

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
            _write_failure_report_once()
            raise
        test_loader = None

    num_classes = dataset.num_classes
    id2label = getattr(dataset, "id2label", None)
    training_mode = cfg.get("training_mode", "standard")

    meta_model = MetaModel(
        run_name=cfg.run_name,
        num_classes=num_classes,
        model_kwargs=cfg.model,
        training_kwargs=cfg.training,
        training_mode=training_mode,
        device=device,
    )
    evaluator = EvaluatorSemanticSegmentation(
        num_classes=num_classes,
        device=device,
    )

    with Logger(
        config=cfg_logger,
        meta_model=meta_model,
        log_epochs=cfg_logger.logger.log_epochs,
        log_checkpoint=cfg_logger.logger.get("log_checkpoint", 50),
        checkpoint_dir=".",
        enable_mlflow=True,
        id2label=id2label,
    ) as logger:
        if logger.mlflow_run_id is not None:
            logging.info("MLflow run_id: %s", logger.mlflow_run_id)

        logger.log_dataloader_params(train_loader, prefix="train_loader")
        logger.log_dataloader_params(val_loader, prefix="val_loader")
        logger.log_dataloader_params(test_loader, prefix="test_loader")

        try:
            train_model(
                meta_model=meta_model,
                evaluator=evaluator,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                logger=logger,
                metric_of_interest=args.metric_of_interest,
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
        finally:
            _write_failure_report_once()
        logging.info("Training complete")


if __name__ == "__main__":
    main()
