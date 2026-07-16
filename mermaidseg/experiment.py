"""YAML-first experiment definition — schema-validate a SageMaker run YAML offline.

An "experiment" is the seg ``config:`` block of a run YAML: the four split-config paths
(``config_data`` / ``config_model`` / ``config_training`` / ``config_logger``) plus a set of
CLI ``overrides:``. That block was previously an *unmodeled* raw dict, consumed only by
``scripts/sagemaker_train_entrypoint.py`` which forwards it to ``scripts/train.py`` as CLI flags
— so a typo'd ``metric-of-interest``, a missing config file, or a dataset section
``_run_training`` cannot handle would only surface once a (paid) job was already running.

:class:`ExperimentSpec` models that block; :meth:`Experiment.validate` runs a set of fully
**offline** checks (no torch model build, no ``SourceLabelRegistry``, no AWS/HF/API call) so a run
YAML can be checked on a laptop or in CI before launch. The full loader + ``dry-run`` (which
assemble the real datasets/registry/model) land in a follow-up change.

CLI::

    python -m mermaidseg.experiment validate sagemaker/runs/issue_12_dinov3_baseline.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mermaidseg.datasets import DATASET_REGISTRY
from mermaidseg.io import setup_config
from mermaidseg.model.metric_policy import canonical_metric_name
from mermaidseg.sagemaker.launcher_config import parse_run_config

# A split value of ``None`` or the literal string ``"None"`` disables that split — this mirrors
# the skip condition in ``scripts/train.py::_run_training`` (``if split_cfg is None or == "None"``).
_DISABLED_SPLITS = (None, "None")


def _hyphenate(name: str) -> str:
    """CLI-flag alias for an override field: ``metric_of_interest`` -> ``metric-of-
    interest``."""
    return name.replace("_", "-")


class Overrides(BaseModel):
    """The ``overrides:`` sub-block — a typed view of the ``scripts/train.py`` CLI
    flags.

    Field names are snake_case; each also accepts its hyphenated CLI-flag alias
    (``run_name`` <-> ``run-name``), because that is how the run YAML and
    ``scripts/sagemaker_train_entrypoint.py`` spell them. ``extra="forbid"`` turns a
    mistyped flag (e.g. ``epoch:`` for ``epochs:``) into a validation error instead of a
    silently-ignored key. Defaults mirror ``_build_parser`` in ``scripts/train.py``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, alias_generator=_hyphenate)

    run_name: str | None = None
    experiment_name: str | None = None
    model: str | None = None
    model_checkpoint: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    iterations_per_train_epoch: int | None = None
    iterations_per_val_epoch: int | None = None
    log_epochs: int | None = None
    seed: int = 42
    num_workers: int = 0
    metric_of_interest: str = "miou"
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    per_class_metrics: bool | None = None
    dry_run: bool = False
    auto_shutdown: bool = False
    log_dir: str = "logs"
    failure_report_path: str | None = None

    @field_validator("metric_of_interest")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        # Raises ValueError (-> pydantic ValidationError) for anything not in SUPPORTED_METRIC_NAMES.
        canonical_metric_name(value)
        return value


class ExperimentSpec(BaseModel):
    """The seg ``config:`` block of a run YAML — the validatable experiment identity.

    Config-path defaults match ``scripts/sagemaker_train_entrypoint.py`` so validation
    reflects what the entrypoint would actually run when a path is omitted.
    """

    model_config = ConfigDict(extra="forbid")

    config_data: str = "configs/data_config.yaml"
    config_model: str = "configs/model_config_cbm.yaml"
    config_training: str = "configs/training_config_cbm.yaml"
    config_logger: str = "configs/logger_config.yaml"
    overrides: Overrides = Field(default_factory=Overrides)

    # Data-release + mapping identity. Optional and additive: existing run YAMLs pin the CoralNet
    # corpus via ``job.env.MERMAID_CORALNET_ANNOTATIONS_PATH`` and do not set these, so they keep
    # validating. The env var remains the runtime source of truth until a later change threads
    # ``annotations_path`` through the loader.
    annotations_path: str | None = None
    mapping_source: Literal["remote", "frozen"] = "remote"

    @property
    def config_paths(self) -> dict[str, str]:
        return {
            "data": self.config_data,
            "model": self.config_model,
            "training": self.config_training,
            "logger": self.config_logger,
        }


@dataclass
class ValidationReport:
    """Result of :meth:`Experiment.validate`.

    ``ok`` is true iff there are no hard ``errors``; ``warnings`` are best-effort
    advisories that do not block a launch. ``summary`` is the resolved experiment
    (datasets, cohort, metric, ...).
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"    - {loc}: {err['msg']}")
    return "\n".join(lines)


def _split_active(split_cfg: Any) -> bool:
    """Mirror ``_run_training``: a split is active unless it is ``None`` or the string
    ``"None"``."""
    return split_cfg not in _DISABLED_SPLITS


def offline_target_universe() -> set[str]:
    """Lowercased MERMAID target names known **without any network call** — the
    committed CoralNet snapshot plus the static per-source maps.

    Best-effort: names produced only via benthic-hierarchy roll-up are absent (the hierarchy is
    fetched live), so callers treat misses as warnings rather than errors.
    """
    from mermaidseg.dataset_reconciliation import label_mapping as lm

    universe: set[str] = set()
    static_maps = (
        lm.load_coralnet_snapshot,
        lm.fetch_catlin_seaview_to_mermaid,
        lm.fetch_moorea_labeled_corals_to_mermaid,
        lm.fetch_pacific_labeled_corals_to_mermaid,
        lm.fetch_benthos_yuval_to_mermaid,
        lm.fetch_ucsd_mosaics_to_mermaid,
        lm.coralscapes_to_mermaid,
        lm.coralscapes_v2_to_mermaid,
    )
    for build_map in static_maps:
        try:
            universe |= {value.lower() for value in build_map().values() if value}
        except Exception:  # noqa: BLE001 — a committed/static map must never break validation
            continue
    return universe


def _coralnet_cohort(coralnet: Any) -> dict[str, int] | None:
    """Summarize the CoralNet source cohort (whitelist/blacklist sizes) for the
    report."""
    if not isinstance(coralnet, Mapping):
        return None
    cohort: dict[str, int] = {}
    for split in ("train", "val"):
        split_cfg = coralnet.get(split)
        if not isinstance(split_cfg, Mapping):
            continue
        for key in ("whitelist_sources", "blacklist_sources"):
            values = split_cfg.get(key)
            if isinstance(values, (list, tuple)):
                cohort[f"{split}.{key}"] = len(values)
    return cohort or None


def _build_summary(spec: ExperimentSpec, cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_block = cfg.get("data") or {}
    enabled: dict[str, list[str]] = {}
    for name, splits in data_block.items():
        if not isinstance(splits, Mapping):
            continue
        active = [split for split in ("train", "val") if _split_active(splits.get(split))]
        if active:
            enabled[name] = active

    training = cfg.get("training") or {}
    overrides = spec.overrides
    return {
        "config_paths": spec.config_paths,
        "training_mode": training.get("training_mode"),
        "datasets_enabled": enabled,
        "num_class_subset": len(training.get("class_subset") or []) or None,
        "metric_of_interest": overrides.metric_of_interest,
        "epochs": overrides.epochs if overrides.epochs is not None else training.get("epochs"),
        "early_stopping": overrides.early_stopping,
        "mapping_source": spec.mapping_source,
        "annotations_path": spec.annotations_path,
        "coralnet_cohort": _coralnet_cohort(data_block.get("coralnet")),
    }


class Experiment:
    """A YAML-first experiment.

    For now it carries the validated :class:`ExperimentSpec`; the dataset/registry/model
    loader and ``dry_run`` (which assemble the real objects) land in a follow-up change.
    """

    def __init__(self, spec: ExperimentSpec):
        self.spec = spec

    @classmethod
    def validate(cls, run_yaml: str | os.PathLike[str]) -> ValidationReport:
        """Offline-validate a run YAML's ``config:`` block (and its ``job:`` block if
        present).

        Runs no torch/registry/AWS/API code: parses the YAML, schema-checks the spec,
        confirms the four config files exist and merge, checks dataset sections against
        ``DATASET_REGISTRY``, and flags ``class_subset`` names not in the committed
        mapping universe. Returns a :class:`ValidationReport`; never raises for an
        invalid config.
        """
        report = ValidationReport()
        path = Path(run_yaml)

        try:
            text = path.read_text()
        except OSError as exc:
            report.errors.append(f"cannot read {path}: {exc}")
            return report
        try:
            raw = yaml.safe_load(os.path.expandvars(text))
        except yaml.YAMLError as exc:
            report.errors.append(f"invalid YAML in {path}: {exc}")
            return report
        if not isinstance(raw, Mapping):
            report.errors.append(f"{path}: top level must be a mapping")
            return report

        # Job block (optional): reuse the launcher's schema so `validate` covers the whole run YAML.
        if "job" in raw:
            try:
                parse_run_config(text, kind="training", strict=False)
            except ValidationError as exc:
                report.errors.append("job: block is invalid:\n" + _format_validation_error(exc))

        config_block = raw.get("config")
        if config_block is None:
            report.errors.append("run YAML is missing the top-level `config:` block")
            return report

        try:
            spec = ExperimentSpec.model_validate(config_block)
        except ValidationError as exc:
            report.errors.append("config: block is invalid:\n" + _format_validation_error(exc))
            return report

        # Do the referenced config files exist?
        missing = {sect: p for sect, p in spec.config_paths.items() if not Path(p).is_file()}
        for sect, path_str in missing.items():
            report.errors.append(f"{sect} config file not found: {path_str}")
        if missing:
            return report

        # Load + merge them (offline: YAML + albumentations compile only).
        try:
            cfg = setup_config(spec.config_paths)
        except Exception as exc:  # noqa: BLE001 — surface any load/merge failure as a clean error
            report.errors.append(f"failed to load/merge config files: {exc}")
            return report

        _check_datasets(cfg, report)
        _check_class_subset(cfg, report)

        report.summary = _build_summary(spec, cfg)
        return report


def _check_datasets(cfg: Mapping[str, Any], report: ValidationReport) -> None:
    """Dataset sections must match ``DATASET_REGISTRY`` exactly.

    ``_run_training`` iterates the full registry and does ``cfg.data[name]``, so a
    missing key would ``KeyError`` at runtime and an extra key is a silently-ignored
    typo.
    """
    data_block = cfg.get("data") or {}
    declared = set(data_block)
    registry = set(DATASET_REGISTRY)
    for name in sorted(registry - declared):
        report.errors.append(
            f"data config missing dataset '{name}' (the training run iterates DATASET_REGISTRY in "
            "full and would KeyError). Declare it, disabling splits with None."
        )
    for name in sorted(declared - registry):
        report.errors.append(
            f"data config has unknown dataset '{name}' (not in DATASET_REGISTRY; it would be "
            "silently ignored). Likely a typo."
        )


def _check_class_subset(cfg: Mapping[str, Any], report: ValidationReport) -> None:
    training = cfg.get("training") or {}
    class_subset = training.get("class_subset") or []
    if not class_subset:
        return
    universe = offline_target_universe()
    unknown = [name for name in class_subset if str(name).lower() not in universe]
    if unknown:
        report.warnings.append(
            f"{len(unknown)} class_subset name(s) not found in the committed mapping universe "
            "(they may be produced via label roll-up — confirm with `dry-run`): "
            + ", ".join(map(str, unknown))
        )


def _render_report(path: Path, report: ValidationReport) -> str:
    lines = [f"[{'OK' if report.ok else 'INVALID'}] {path}"]
    for err in report.errors:
        lines.append(f"  ERROR: {err}")
    for warn in report.warnings:
        lines.append(f"  warning: {warn}")
    if report.summary:
        lines.append("  summary:")
        for key, value in report.summary.items():
            lines.append(f"    {key}: {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mermaidseg.experiment",
        description="Validate a SageMaker run YAML's experiment (`config:`) block.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate", help="offline schema + static checks (no AWS/HF/network)"
    )
    validate_parser.add_argument(
        "run_yaml", type=Path, nargs="+", help="run YAML file(s) to validate"
    )
    args = parser.parse_args(argv)

    if args.command == "validate":
        all_ok = True
        for path in args.run_yaml:
            report = Experiment.validate(path)
            print(_render_report(Path(path), report))
            all_ok = all_ok and report.ok
        return 0 if all_ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
