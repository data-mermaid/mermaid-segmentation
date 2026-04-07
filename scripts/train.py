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
from datetime import datetime
from pathlib import Path

import albumentations as A
import torch
from torch.utils.data import DataLoader, random_split

import mermaidseg.datasets.dataset
from mermaidseg.io import get_parser, setup_config, update_config_with_args
from mermaidseg.logger import Logger
from mermaidseg.model.eval import EvaluatorSemanticSegmentation
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.metric_policy import METRIC_POLICY
from mermaidseg.model.train import train_model

_METADATA = Path("/opt/ml/metadata/resource-metadata.json")


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
    logging.info("Logging to %s", log_file)


def _cleanup_pid(pid_file: Path) -> None:
    pid_file.unlink(missing_ok=True)


def _write_pid(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def _stop_current_space() -> None:
    """Stop the SageMaker space immediately.

    No-op outside SageMaker.
    """
    if not _METADATA.exists():
        logging.info("Not running on SageMaker — skipping auto-shutdown")
        return
    import boto3
    import botocore.exceptions

    info = json.loads(_METADATA.read_text())
    try:
        boto3.client("sagemaker").stop_space(DomainId=info["DomainId"], SpaceName=info["SpaceName"])
        logging.info(
            "Space shutdown initiated (DomainId=%s, SpaceName=%s)",
            info["DomainId"],
            info["SpaceName"],
        )
    except botocore.exceptions.ClientError as e:
        logging.warning("Auto-shutdown failed (check sagemaker:StopSpace permission): %s", e)


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
        choices=sorted(METRIC_POLICY),
        help="metric used for checkpointing and early stopping",
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
        _stop_current_space()


def _run_training(args: argparse.Namespace) -> None:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    cfg = setup_config(
        config_path=args.config,
        config_base_path=args.config_base,
    )
    cfg = update_config_with_args(cfg, args)
    cfg_logger = copy.deepcopy(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    transforms = {
        split: A.Compose([getattr(A, name)(**params) for name, params in augmentations.items()]) for split, augmentations in cfg.augmentation.items()
    }

    dataset_name = cfg.data.pop("name", None)
    batch_size = cfg.data.pop("batch_size", 8)

    dataset = getattr(mermaidseg.datasets.dataset, dataset_name)(transform=transforms["train"], **cfg.data)
    logging.info("Dataset: %s (%d samples)", dataset_name, len(dataset))

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    if args.dry_run:
        logging.info("Dry run: limiting to 1 batch")
        train_loader = [next(iter(train_loader))]
        val_loader = [next(iter(val_loader))]
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
        logging.info("Training complete")


if __name__ == "__main__":
    main()
