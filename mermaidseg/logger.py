"""Serialization strategy:
- Training checkpoints: torch.save (pickle) — full state for resume.
"""

import copy
import json
import logging
import os
import re
import tempfile
import time
import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mermaidseg.model.meta import MetaModel

logger = logging.getLogger(__name__)


try:
    import wandb

    WANDB_IMPORT_ERROR = None
except ImportError as err:
    wandb = None
    WANDB_IMPORT_ERROR = err


def prune_rolling_checkpoints(
    directory: str, keep_name: str, prefix: str = "model_preempt"
) -> None:
    """Remove other rolling preempt checkpoints in ``directory``.

    Interval checkpoints (``model_step*``, ``model_epoch*``) use other prefixes and
    are left in place. ``.partial`` files from an interrupted atomic save are removed
    too. ``keep_name`` is the checkpoint that was just renamed into place.
    """
    if not prefix or not os.path.isdir(directory):
        return
    for entry in os.listdir(directory):
        if entry == keep_name or not entry.startswith(prefix):
            continue
        path = os.path.join(directory, entry)
        try:
            os.remove(path)
        except OSError:
            logger.warning("Failed to remove old rolling checkpoint %s", path)


def _safe_wandb_artifact_name(name: str) -> str:
    """Sanitize a string into a valid wandb artifact name (alnum, ``_``, ``-``, ``.``)."""
    return re.sub(r"[^a-zA-Z0-9_.\-]", "-", name)


class Logger:
    """Logger for experiment tracking during training and evaluation.

    Logs to wandb via the ``WandbLogger`` delegate (``self._wandb_logger``) when
    ``enable_wandb`` is set.
    """

    @staticmethod
    def _resolve_backend_flag(name, logger_cfg, default):
        """Resolve a backend enable flag from ``config.logger.<name>``, else ``default``."""
        cfg_val = getattr(logger_cfg, name, None) if logger_cfg is not None else None
        if cfg_val is None:
            return default
        return bool(cfg_val)

    def __init__(
        self,
        config,
        meta_model,
        log_epochs=5,
        log_checkpoint=50,
        checkpoint_dir=".",
        id2label=None,
        id2concept=None,
        save_local_checkpoints=None,
    ):
        self.config = config
        self.log_epochs = log_epochs
        self.log_checkpoint = log_checkpoint
        if self.log_checkpoint <= 0:
            raise ValueError("log_checkpoint must be > 0")
        self.checkpoint_dir = checkpoint_dir
        self.run_name = meta_model.run_name
        self._wandb_logger = None
        self.id2label = id2label
        self.id2concept = id2concept
        logger_cfg = getattr(config, "logger", None)
        experiment_name = getattr(logger_cfg, "experiment_name", None)

        # Backend selection is config-driven (config.logger.enable_wandb).
        enable_wandb = self._resolve_backend_flag("enable_wandb", logger_cfg, default=False)

        # Keep one source of truth for local persistence.
        if save_local_checkpoints is not None:
            self.save_local_checkpoints = save_local_checkpoints
        else:
            cfg_local = getattr(logger_cfg, "save_local_checkpoints", None)
            if cfg_local is None:
                legacy_cfg = getattr(logger_cfg, "save_local_models", None)
                if legacy_cfg is not None:
                    warnings.warn(
                        "config.logger.save_local_models is deprecated; use config.logger.save_local_checkpoints.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                cfg_local = legacy_cfg
            self.save_local_checkpoints = cfg_local if cfg_local is not None else True

        self.save_local_models = self.save_local_checkpoints
        self.wandb_run_id = None

        # Precompute run metadata (config artifact, flattened params, tags).
        config_to_log = None
        params = None
        tags = None
        if config is not None:
            try:
                config_to_log = copy.deepcopy(config)
                config_to_log["num_classes"] = int(meta_model.num_classes)

                params = {
                    "num_classes": int(meta_model.num_classes),
                    "run_name": self.run_name,
                    "log_epochs": log_epochs,
                    "log_checkpoint": log_checkpoint,
                }
                if hasattr(config, "model"):
                    params.update(
                        {
                            f"model_{k}": v
                            for k, v in dict(config.model).items()
                            if isinstance(v, str | int | float | bool)
                        }
                    )
                if hasattr(config, "training"):
                    params.update(
                        {
                            f"training_{k}": v
                            for k, v in dict(config.training).items()
                            if isinstance(v, str | int | float | bool)
                        }
                    )

                tags = {
                    "model_type": meta_model.__class__.__name__,
                    "framework": "pytorch",
                }
            except Exception as e:
                logger.warning("Failed to prepare run metadata (config/params/tags): %s", e)
                config_to_log = params = tags = None

        if enable_wandb:
            if WANDB_IMPORT_ERROR is not None:
                logger.warning("wandb is not installed. Wandb logging is disabled.")
            else:
                try:
                    wandb_project = getattr(logger_cfg, "wandb_project", None) or (
                        experiment_name or "mermaidseg"
                    )
                    wandb_entity = getattr(logger_cfg, "wandb_entity", None)
                    self._wandb_logger = WandbLogger(
                        project=wandb_project,
                        run_name=self.run_name,
                        config=config,
                        num_classes=int(meta_model.num_classes),
                        entity=wandb_entity,
                    )
                    self.wandb_run_id = self._wandb_logger.wandb_run_id
                    if config_to_log is not None:
                        logger.info("Logging config to wandb...")
                        self._wandb_logger.log_config(config_to_log)
                        self._wandb_logger.log_params(params)
                        self._wandb_logger.set_tags(tags)
                        self._wandb_logger.log_concept_metadata(
                            id2label=self.id2label,
                            id2concept=self.id2concept,
                            num_concepts=getattr(meta_model, "num_concepts", None),
                            conceptid2labelid=getattr(meta_model, "conceptid2labelid", None),
                        )
                except Exception as e:
                    logger.warning("Failed to initialize wandb logging: %s", e)

    def _unpack_metrics(self, metrics_dict: dict) -> dict[str, float]:
        """Unpack array-valued metrics into per-class/concept named scalars.

        Array values are expanded using id2concept (for concept metrics) or id2label
        (for class metrics). Scalar values are passed through as-is.

        Args:
            metrics_dict: Raw metrics dict, values may be float or np.ndarray.
        """
        result: dict[str, float] = {}
        for k, v in (metrics_dict or {}).items():
            if isinstance(v, np.ndarray):
                if "concept" in k.lower():
                    id_map = self.id2concept or {}
                    fallback = "concept"
                else:
                    id_map = self.id2label or {}
                    fallback = "class"
                for idx, val in enumerate(v):
                    name = id_map.get(idx, f"{fallback}_{idx}")
                    result[f"{k}/{name}"] = float(val)
            else:
                result[k] = float(v)
        return result

    def log_reconciliation(self, registry) -> None:
        """Log the joint :class:`SourceLabelRegistry` artifacts to wandb.

        Saves ``global_id2source``, ``target_id2label``, ``source_to_target`` and (when present)
        ``concept_id2name`` so runs are self-describing.
        """
        if self._wandb_logger is None:
            return
        try:
            self.log_dict(
                {str(k): list(v) for k, v in registry.global_id2source.items()},
                "metadata/global_id2source.json",
            )
            self.log_dict(
                {str(k): v for k, v in registry.target_id2label.items()},
                "metadata/target_id2label.json",
            )
            self.log_dict(
                {str(k): v for k, v in registry.dataset_offsets.items()},
                "metadata/dataset_offsets.json",
            )
            source_to_target = registry.source_to_target.detach().cpu().tolist()
            self.log_dict(
                {str(i): int(v) for i, v in enumerate(source_to_target)},
                "metadata/source_to_target.json",
            )
            if registry.concept_id2name:
                self.log_dict(
                    {str(k): v for k, v in registry.concept_id2name.items()},
                    "metadata/concept_id2name.json",
                )
            self.log_params(
                {
                    "num_target_classes": int(registry.num_target_classes),
                    "num_global_source_classes": int(registry.num_global_source_classes),
                }
            )
        except Exception as e:
            logger.warning("Failed to log SourceLabelRegistry: %s", e)

    def log_dataloader_params(self, loader: DataLoader, prefix: str = "dataloader") -> None:
        """Log DataLoader configuration as params to wandb."""
        if self._wandb_logger is not None:
            self._wandb_logger.log_dataloader_params(loader, prefix=prefix)

    @property
    def enable_wandb(self) -> bool:
        """True when a ``WandbLogger`` delegate is active."""
        return self._wandb_logger is not None

    def log(self, log_dict, step):
        # Array-valued per-class/per-concept metrics are expanded into named scalars.
        metrics_to_log = self._unpack_metrics(log_dict)

        if self._wandb_logger is not None:
            self._wandb_logger.log(metrics_to_log, step=step)

    def log_params(self, params: dict) -> None:
        """Log flat scalar params to wandb."""
        if not params:
            return
        if self._wandb_logger is not None:
            self._wandb_logger.log_params(params)

    def log_dict(self, data: dict, artifact_path: str) -> None:
        """Log a JSON-serializable dict as a wandb artifact.

        The basename of ``artifact_path`` (e.g. ``"metadata/concept_id2name.json"``),
        without the ``.json`` suffix, becomes the artifact name.
        """
        if self._wandb_logger is not None:
            name = os.path.splitext(os.path.basename(artifact_path))[0]
            self._wandb_logger._log_json_artifact(name, data, artifact_type="metadata")

    @property
    def run_url(self) -> str | None:
        """Best-effort clickable URL for the active wandb run."""
        if self._wandb_logger is not None and self._wandb_logger.wandb_run is not None:
            try:
                return self._wandb_logger.wandb_run.url
            except Exception:
                return None
        return None

    def save_model_checkpoint(
        self,
        meta_model_run: MetaModel,
        epoch: int,
        metrics_dict: dict[str, float | np.ndarray],
        checkpoint_name: str | None = None,
        epoch_complete: bool = True,
        atomic: bool = False,
    ):
        """Save a model checkpoint locally.

        Local persistence is controlled by ``save_local_checkpoints``. Checkpoints are not
        uploaded to wandb.

        Checkpoint files are named ``model_epoch{epoch}`` unless ``checkpoint_name`` is set
        (for example ``model_step25000``). ``epoch_complete`` is false for a mid-epoch step
        checkpoint so a resume continues that epoch instead of skipping it.

        ``atomic=True`` writes ``<name>.partial`` and renames it into place, so a kill during
        ``torch.save`` does not leave a truncated file at the final path. The rolling preempt
        checkpoints use this. Interval checkpoints keep the direct write.

        The live module stays on its device. A detached CPU copy of the state dict is saved
        so a DDP reducer keeps the parameter storage it was built with.
        """
        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = checkpoint_name or f"model_epoch{epoch}"

        checkpoint: dict[str, Any] = {
            "config": self.config,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in meta_model_run.model.state_dict().items()
            },
            "optimizer_state_dict": meta_model_run.optimizer.state_dict(),
            "epoch": epoch,
            "epoch_complete": bool(epoch_complete),
            "global_step": int(getattr(meta_model_run, "global_step", 0)),
            "warmup_iters_completed": int(getattr(meta_model_run, "_warmup_iters_completed", 0)),
            "timestamp": timestamp,
            "metrics": metrics_dict,
        }

        if getattr(meta_model_run, "scheduler", None) is not None:
            checkpoint["scheduler_state_dict"] = meta_model_run.scheduler.state_dict()

        model_path = f"{self.checkpoint_dir}/model_checkpoints/{meta_model_run.run_name}/{filename}"

        if self.save_local_checkpoints:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if atomic:
                partial_path = model_path + ".partial"
                torch.save(checkpoint, partial_path)  # type: ignore
                os.replace(partial_path, model_path)
            else:
                if os.path.exists(model_path):
                    logger.warning(
                        "Checkpoint file already exists and will be overwritten: %s. "
                        "Use a different checkpoint_dir or run_name when resuming training "
                        "to avoid silently overwriting prior checkpoints.",
                        model_path,
                    )
                torch.save(checkpoint, model_path)  # type: ignore
            logger.info("Checkpoint saved locally: %s", model_path)
        else:
            logger.info("Local checkpoint saving disabled, skipping disk write")

    def end_run(self):
        """End the wandb delegate's run, if active.

        Should be called at the end of training to ensure proper cleanup.
        """
        if self._wandb_logger is not None:
            self._wandb_logger.end_run()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False


class WandbLogger:
    """Standalone wandb logger, used as the ``Logger`` delegate for wandb-backed runs."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config,
        num_classes: int,
        *,
        entity: str | None = None,
        run_id: str | None = None,
        resume: str = "allow",
    ):
        if WANDB_IMPORT_ERROR is not None:
            raise ImportError(
                "wandb is required for WandbLogger but is not installed. Install with: pip install wandb"
            ) from WANDB_IMPORT_ERROR

        # slurm/train.sbatch sets WANDB_RUN_ID so every resubmitted job of a run
        # appends to the same wandb run. Without it, each process starts a new run.
        if run_id is None:
            env_run_id = os.environ.get("WANDB_RUN_ID", "").strip()
            if env_run_id:
                run_id = env_run_id
                resume = "allow"

        self.wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            id=run_id,
            resume=resume if run_id is not None else None,
        )
        self.wandb_run.config.update({"num_classes": num_classes}, allow_val_change=True)
        self.wandb_run_id = self.wandb_run.id if self.wandb_run is not None else None

    def _log_json_artifact(self, name: str, data: dict, artifact_type: str = "metadata") -> None:
        """Log a dict as a JSON-file wandb artifact."""
        if self.wandb_run is None:
            return
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"{name}.json"
                local_path = os.path.join(tmpdir, filename)
                with open(local_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                artifact = wandb.Artifact(_safe_wandb_artifact_name(name), type=artifact_type)
                artifact.add_file(local_path, name=filename)
                self.wandb_run.log_artifact(artifact)
        except Exception as e:
            logger.warning("Failed to log %s to wandb: %s", name, e)

    def log_config(self, config_to_log: dict) -> None:
        """Log the full training config as a JSON artifact (mirrors config/config.json)."""
        self._log_json_artifact("config", config_to_log, artifact_type="config")

    def log_params(self, params: dict) -> None:
        """Log flattened scalar params (run_name, log_epochs, model_*, training_*, ...)."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.config.update(params, allow_val_change=True)
        except Exception as e:
            logger.warning("Failed to log params to wandb: %s", e)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set run tags (model_type/framework, ...) as both config keys and wandb tags."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.config.update(tags, allow_val_change=True)
            existing = list(self.wandb_run.tags or [])
            new_tags = [f"{k}:{v}" for k, v in tags.items() if f"{k}:{v}" not in existing]
            if new_tags:
                self.wandb_run.tags = existing + new_tags
        except Exception as e:
            logger.warning("Failed to set tags on wandb: %s", e)

    def log_concept_metadata(
        self,
        id2label: dict | None = None,
        id2concept: dict | None = None,
        num_concepts: int | None = None,
        conceptid2labelid: dict | None = None,
    ) -> None:
        """Log concept-related metadata artifacts when available."""
        if self.wandb_run is None:
            return
        if id2label:
            self._log_json_artifact("id2label", id2label, artifact_type="metadata")
        if id2concept:
            self._log_json_artifact("id2concept", id2concept, artifact_type="metadata")
        if num_concepts is not None:
            try:
                self.wandb_run.config.update(
                    {"num_concepts": int(num_concepts)}, allow_val_change=True
                )
            except Exception as e:
                logger.warning("Failed to log num_concepts to wandb: %s", e)
        if conceptid2labelid is not None:
            self._log_json_artifact(
                "conceptid2labelid", conceptid2labelid, artifact_type="metadata"
            )

    def log_dataloader_params(self, loader, prefix: str = "dataloader") -> None:
        """Log DataLoader configuration as wandb config keys."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.config.update(
                {
                    f"{prefix}_num_workers": loader.num_workers,
                    f"{prefix}_pin_memory": loader.pin_memory,
                    f"{prefix}_batch_size": loader.batch_size,
                    f"{prefix}_persistent_workers": getattr(loader, "persistent_workers", False),
                    f"{prefix}_prefetch_factor": getattr(loader, "prefetch_factor", None),
                },
                allow_val_change=True,
            )
        except Exception as e:
            logger.warning("Failed to log dataloader params to wandb: %s", e)

    def log(self, log_dict, step):
        """Log metrics to the active wandb run."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(log_dict, step=step)
        except Exception as e:
            logger.warning("Failed to log metrics to wandb: %s", e)

    def end_run(self):
        """Finish the active wandb run."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
            logger.info("Ended wandb run")
        except Exception as e:
            logger.warning("Failed to end wandb run: %s", e)
