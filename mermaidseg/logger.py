"""Experiment tracking for training and evaluation runs.

The ``logger.tracker`` config key names the tracking backend (``TRACKER_CLASSES``). Each
backend imports its library lazily, so the package imports without the tracker extras.
"""

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

from torch.utils.data import DataLoader

from mermaidseg.config import lookup_class, to_plain_dict
from mermaidseg.model.meta import MetaModel

if TYPE_CHECKING:
    from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry

logger = logging.getLogger(__name__)


def _safe_wandb_artifact_name(name: str) -> str:
    """Sanitize a string into a valid wandb artifact name.

    Allowed characters are alphanumerics, ``_``, ``-`` and ``.``.
    """
    return re.sub(r"[^a-zA-Z0-9_.\-]", "-", name)


class Tracker(Protocol):
    """An experiment-tracking backend: the calls ``Logger`` makes on one run."""

    @property
    def run_id(self) -> str | None:
        """The backend's id of the active run."""
        ...

    @property
    def run_url(self) -> str | None:
        """A browser URL for the active run, when the backend has one."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log flat run parameters."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at ``step``."""
        ...

    def log_dict(self, data: dict[str, Any], artifact_path: str) -> None:
        """Log ``data`` as a JSON artifact at ``artifact_path``."""
        ...

    def end_run(self, failed: bool) -> None:
        """End the active run, marked failed when ``failed``."""
        ...


METRIC_WARNING_INTERVAL = 100


def _warn_metrics_failure(backend: str, failures: int, error: Exception) -> None:
    """Warn on the 1st ``log_metrics`` failure and on every 100th after it.

    ``failures`` is the running count including this one. A tracker outage with a
    per-batch log interval would otherwise write one identical warning per batch.
    """
    if failures == 1 or failures % METRIC_WARNING_INTERVAL == 0:
        logger.warning("Failed to log metrics to %s (failure %d): %s", backend, failures, error)


def _warn_metrics_failure_total(backend: str, failures: int) -> None:
    """Warn with the total of failed ``log_metrics`` calls when it is non-zero."""
    if failures:
        logger.warning("%d log_metrics calls to %s failed during the run", failures, backend)


class WandbTracker:
    """Tracker that logs to one wandb run."""

    def __init__(self, project: str, entity: str | None, run_name: str, config: Any) -> None:
        """Start a wandb run, resuming the run id in ``WANDB_RUN_ID`` when it is set.

        Raises:
            ImportError: If wandb is not installed.
            RuntimeError: If no wandb API key resolves (``WANDB_API_KEY``, ``wandb login`` or
                ``~/.netrc``).
        """
        try:
            import wandb
        except ImportError as err:
            raise ImportError(
                "wandb is required for tracker 'wandb'. Install it with: uv sync --extra wandb"
            ) from err
        if not wandb.api.api_key:
            raise RuntimeError(
                "wandb is not authenticated. Run `wandb login` in a terminal, "
                "set WANDB_API_KEY, or call wandb.login(key=...) in a notebook cell."
            )
        self._wandb = wandb
        # slurm/train.sbatch sets WANDB_RUN_ID so every resubmitted job of a run
        # appends to the same wandb run. Without it, each process starts a new run.
        run_id = os.environ.get("WANDB_RUN_ID", "").strip() or None
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            id=run_id,
            resume="allow" if run_id is not None else None,
        )
        self._metric_failures = 0

    @classmethod
    def from_config(cls, logger_cfg: Any, run_name: str, config: Any) -> Self:
        """Start a run in ``logger_cfg.wandb_project`` and ``logger_cfg.wandb_entity``."""
        return cls(
            project=logger_cfg.wandb_project,
            entity=logger_cfg.wandb_entity,
            run_name=run_name,
            config=config,
        )

    @property
    def run_id(self) -> str | None:
        """The wandb run id."""
        return self._run.id

    @property
    def run_url(self) -> str | None:
        """The wandb run page."""
        return self._run.url

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params as wandb config keys."""
        try:
            self._run.config.update(params, allow_val_change=True)
        except Exception as e:
            logger.warning("Failed to log params to wandb: %s", e)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to the wandb run at ``step``."""
        try:
            self._run.log(metrics, step=step)
        except Exception as e:
            self._metric_failures += 1
            _warn_metrics_failure("wandb", self._metric_failures, e)

    def log_dict(self, data: dict[str, Any], artifact_path: str) -> None:
        """Log ``data`` as a JSON-file artifact named after the basename of ``artifact_path``.

        ``"metadata/concept_id2name.json"`` becomes the artifact ``concept_id2name``.
        """
        name = os.path.splitext(os.path.basename(artifact_path))[0]
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"{name}.json"
                local_path = os.path.join(tmpdir, filename)
                with open(local_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                artifact = self._wandb.Artifact(_safe_wandb_artifact_name(name), type="metadata")
                artifact.add_file(local_path, name=filename)
                self._run.log_artifact(artifact)
        except Exception as e:
            logger.warning("Failed to log %s to wandb: %s", name, e)

    def end_run(self, failed: bool) -> None:
        """Finish the wandb run with exit code 1 when ``failed``, else 0."""
        try:
            self._run.finish(exit_code=1 if failed else 0)
            logger.info("Ended wandb run")
        except Exception as e:
            logger.warning("Failed to end wandb run: %s", e)
        _warn_metrics_failure_total("wandb", self._metric_failures)


def _mlflow_key(key: str) -> str:
    """``key`` with every character MLflow rejects in a metric or param name replaced by ``_``."""
    return re.sub(r"[^\w/.\- :]", "_", key)


def _mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str | None:
    """The MLflow UI page of a run on an http(s) tracking server, else ``None``."""
    if not tracking_uri.startswith(("http://", "https://")):
        return None
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


class MlflowTracker:
    """Tracker that logs to one MLflow run on the server in ``MLFLOW_TRACKING_URI``."""

    def __init__(self, experiment_name: str, run_name: str) -> None:
        """Connect to the tracking server and start a run in ``experiment_name``.

        Raises:
            RuntimeError: If ``MLFLOW_TRACKING_URI`` is unset.
            ImportError: If mlflow, or sagemaker-mlflow for an ``arn:`` URI, is not installed.
        """
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        if not tracking_uri:
            raise RuntimeError("tracker 'mlflow' needs MLFLOW_TRACKING_URI to be set")
        if tracking_uri.startswith("arn:"):
            try:
                import sagemaker_mlflow  # noqa: F401  # registers the arn: URI scheme
            except ImportError as err:
                raise ImportError(
                    "MLFLOW_TRACKING_URI is a SageMaker ARN, which needs sagemaker-mlflow. "
                    "Install it with: uv sync --extra mlflow"
                ) from err
        try:
            import mlflow
            import mlflow.pytorch
        except ImportError as err:
            raise ImportError(
                "mlflow is required for tracker 'mlflow'. Install it with: uv sync --extra mlflow"
            ) from err
        self._mlflow = mlflow
        self._tracking_uri = tracking_uri
        # Autologging re-logs committed metrics, a UniqueViolation on SageMaker's metric table.
        mlflow.pytorch.autolog(disable=True)
        mlflow.set_tracking_uri(tracking_uri)
        # Connectivity probe: fails fast, before a run is created on an unreachable server.
        mlflow.search_experiments(max_results=1)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self._metric_failures = 0

    @classmethod
    def from_config(cls, logger_cfg: Any, run_name: str, config: Any) -> Self:
        """Start a run in ``logger_cfg.experiment_name``; MLflow ignores ``config``."""
        return cls(experiment_name=logger_cfg.experiment_name, run_name=run_name)

    @property
    def run_id(self) -> str | None:
        """The MLflow run id."""
        return self._run.info.run_id

    @property
    def run_url(self) -> str | None:
        """The MLflow UI page of the run, or ``None`` for a non-http tracking URI."""
        return _mlflow_run_url(
            self._tracking_uri, self._run.info.experiment_id, self._run.info.run_id
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params under sanitized keys; MLflow stores each value as its ``str``.

        The write is asynchronous (``synchronous=False``): a failed server write is not
        raised here. It surfaces only as mlflow's own ERROR log lines and at ``end_run``.
        """
        try:
            self._mlflow.log_params(
                {_mlflow_key(k): v for k, v in params.items()}, synchronous=False
            )
        except Exception as e:
            logger.warning("Failed to log params to MLflow: %s", e)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics under sanitized keys at ``step``.

        The write is asynchronous (``synchronous=False``): a failed server write is not
        raised here. It surfaces only as mlflow's own ERROR log lines and at ``end_run``.
        """
        try:
            self._mlflow.log_metrics(
                {_mlflow_key(k): v for k, v in metrics.items()}, step=step, synchronous=False
            )
        except Exception as e:
            self._metric_failures += 1
            _warn_metrics_failure("MLflow", self._metric_failures, e)

    def log_dict(self, data: dict[str, Any], artifact_path: str) -> None:
        """Log ``data`` as the JSON artifact ``artifact_path`` of the run."""
        try:
            self._mlflow.log_dict(data, artifact_path)
        except Exception as e:
            logger.warning("Failed to log %s to MLflow: %s", artifact_path, e)

    def end_run(self, failed: bool) -> None:
        """End the MLflow run as FAILED when ``failed``, else FINISHED.

        Ending the run flushes the asynchronous param and metric queue.
        """
        try:
            self._mlflow.end_run(status="FAILED" if failed else "FINISHED")
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.warning("Failed to end MLflow run: %s", e)
        _warn_metrics_failure_total("MLflow", self._metric_failures)


TRACKER_CLASSES: dict[str, Callable[[Any, str, Any], Tracker]] = {
    "wandb": WandbTracker.from_config,
    "mlflow": MlflowTracker.from_config,
}


def make_tracker(logger_cfg: Any, run_name: str, config: Any) -> Tracker:
    """Start a run on the backend that ``logger_cfg.tracker`` names.

    Args:
        logger_cfg: The validated ``logger`` config section.
        run_name: Name of the run.
        config: The full run config, which wandb records on the run.

    Raises:
        ValueError: If ``logger_cfg.tracker`` is not in ``TRACKER_CLASSES``.
    """
    factory = lookup_class(TRACKER_CLASSES, "tracker", logger_cfg.tracker)
    return factory(logger_cfg, run_name, config)


class Logger:
    """Logger for experiment tracking during training and evaluation.

    Logs to the ``Tracker`` that ``config.logger.tracker`` names (``self.tracker``).
    """

    def __init__(
        self,
        config: Any,
        meta_model: MetaModel,
        id2label: dict | None = None,
    ) -> None:
        """Start the tracker run and log the run metadata.

        Raises:
            Exception: Whatever the tracker raises when its run cannot start.
        """
        self.config = config
        self.run_name = meta_model.run_name
        self.id2label = id2label
        self.tracker = make_tracker(config.logger, self.run_name, config)
        self._log_run_metadata(meta_model)

    def _log_run_metadata(self, meta_model: MetaModel) -> None:
        """Log the config artifact, the flattened run params and ``id2label`` to the tracker."""
        try:
            config_to_log = to_plain_dict(self.config)
            config_to_log["num_classes"] = int(meta_model.num_classes)
            params = _run_params(self.config, meta_model)
        except Exception as e:
            logger.warning("Failed to prepare run metadata (config/params): %s", e)
            return
        self.tracker.log_dict(config_to_log, "config/config.json")
        self.tracker.log_params(params)
        if self.id2label:
            self.tracker.log_dict(self.id2label, "metadata/id2label.json")

    def _unpack_metrics(self, metrics_dict: dict) -> dict[str, float]:
        """Metrics as floats, ``None`` as an empty dict."""
        return {k: float(v) for k, v in (metrics_dict or {}).items()}

    def log_reconciliation(self, registry: "SourceLabelRegistry") -> None:
        """Log the joint :class:`SourceLabelRegistry` artifacts to the tracker.

        Saves ``global_id2source``, ``target_id2label``, ``source_to_target``,
        ``concept_id2name`` and the CoralNet label-mapping rows, so runs are self-describing.
        """
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
        """Log DataLoader configuration as tracker params."""
        self.log_params(
            {
                f"{prefix}_num_workers": loader.num_workers,
                f"{prefix}_pin_memory": loader.pin_memory,
                f"{prefix}_batch_size": loader.batch_size,
                f"{prefix}_persistent_workers": getattr(loader, "persistent_workers", False),
                f"{prefix}_prefetch_factor": getattr(loader, "prefetch_factor", None),
            }
        )

    def log(self, log_dict: dict, step: int) -> None:
        """Log metrics as float scalars."""
        self.tracker.log_metrics(self._unpack_metrics(log_dict), step=step)

    def log_params(self, params: dict) -> None:
        """Log flat scalar params to the tracker."""
        if not params:
            return
        self.tracker.log_params(params)

    def log_dict(self, data: dict, artifact_path: str) -> None:
        """Log a JSON-serializable dict as the tracker artifact ``artifact_path``."""
        self.tracker.log_dict(data, artifact_path)

    @property
    def run_id(self) -> str | None:
        """The tracker's id of the active run."""
        return self.tracker.run_id

    @property
    def run_url(self) -> str | None:
        """A browser URL for the active run, when the tracker has one."""
        return self.tracker.run_url

    def end_run(self, failed: bool) -> None:
        """End the tracker run, marked failed when ``failed``.

        Should be called at the end of training to ensure proper cleanup.
        """
        self.tracker.end_run(failed)

    def __enter__(self) -> Self:
        """Return this logger for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """End the run on context exit, failed when an exception is leaving; never suppress it."""
        self.end_run(exc_type is not None)
        return False


def _run_params(config: Any, meta_model: MetaModel) -> dict[str, Any]:
    """Flat run params: class count, run name, scalar model/training keys and model type."""
    params: dict[str, Any] = {
        "num_classes": int(meta_model.num_classes),
        "run_name": meta_model.run_name,
    }
    for section in ("model", "training"):
        if hasattr(config, section):
            params.update(
                {
                    f"{section}_{k}": v
                    for k, v in dict(config[section]).items()
                    if isinstance(v, str | int | float | bool)
                }
            )
    params["model_type"] = meta_model.__class__.__name__
    params["framework"] = "pytorch"
    params["num_concepts"] = int(meta_model.num_concepts)
    return params
