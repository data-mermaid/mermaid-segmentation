"""YAML-first experiment definition — validate, load, and dry-run a SageMaker run YAML.

An "experiment" is the seg ``config:`` block of a run YAML: the four split-config paths
(``config_data`` / ``config_model`` / ``config_training`` / ``config_logger``) plus a set of CLI
``overrides:``. That block was previously an *unmodeled* raw dict, consumed only by
``scripts/sagemaker_train_entrypoint.py`` which forwards it to ``scripts/train.py`` as CLI flags —
so a typo'd ``metric-of-interest``, a missing config file, or a dataset section ``_run_training``
cannot handle would only surface once a (paid) job was already running.

This module gives that block one owner:

- :class:`ExperimentSpec` models it (pydantic).
- :meth:`Experiment.validate` runs fully **offline** checks (no torch/registry/AWS/API) so a run
  YAML can be checked on a laptop or in CI before launch.
- :class:`Experiment` also owns the run assembly (datasets -> registry -> model -> evaluator) that
  ``scripts/train.py`` used to inline, so the CLI, a notebook, and :meth:`Experiment.dry_run` all
  build the same objects the same way instead of re-deriving them.

CLI::

    python -m mermaidseg.experiment validate sagemaker/runs/issue_12_dinov3_baseline.yaml
    python -m mermaidseg.experiment dry-run  sagemaker/runs/issue_12_dinov3_baseline.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from torch.utils.data import ConcatDataset, DataLoader

from mermaidseg.dataset_reconciliation import (
    ConceptSchema,
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.datasets import DATASET_REGISTRY, BaseCoralDataset, worker_init_fn
from mermaidseg.io import ConfigDict as RunConfigDict
from mermaidseg.io import setup_config, update_config_with_args
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel
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

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=_hyphenate,
        protected_namespaces=(),
    )

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


def _spec_from_args(args: argparse.Namespace) -> ExperimentSpec:
    """Build an :class:`ExperimentSpec` from ``scripts/train.py`` argparse args.

    Uses ``getattr`` defaults matching ``_build_parser`` so a partial ``Namespace``
    (e.g. from a test) still resolves; the resulting spec drives the same config merge
    the CLI always used.
    """

    def g(name: str, default: Any = None) -> Any:
        return getattr(args, name, default)

    overrides = Overrides(
        run_name=g("run_name"),
        experiment_name=g("experiment_name"),
        model=g("model"),
        model_checkpoint=g("model_checkpoint"),
        epochs=g("epochs"),
        batch_size=g("batch_size"),
        lr=g("lr"),
        iterations_per_train_epoch=g("iterations_per_train_epoch"),
        iterations_per_val_epoch=g("iterations_per_val_epoch"),
        log_epochs=g("log_epochs"),
        seed=g("seed", 42),
        num_workers=g("num_workers", 0),
        metric_of_interest=g("metric_of_interest", "miou"),
        early_stopping=g("early_stopping", False),
        early_stopping_patience=g("early_stopping_patience", 10),
        early_stopping_min_delta=g("early_stopping_min_delta", 0.0),
        per_class_metrics=g("per_class_metrics"),
        dry_run=g("dry_run", False),
        auto_shutdown=g("auto_shutdown", False),
        log_dir=g("log_dir", "logs"),
        failure_report_path=g("failure_report_path"),
    )
    return ExperimentSpec(
        config_data=g("config_data", "configs/data_config.yaml"),
        config_model=g("config_model", "configs/model_config_cbm.yaml"),
        config_training=g("config_training", "configs/training_config_cbm.yaml"),
        config_logger=g("config_logger", "configs/logger_config.yaml"),
        overrides=overrides,
    )


def _apply_overrides_to_config(cfg: RunConfigDict, overrides: Overrides) -> RunConfigDict:
    """Merge overrides into the config via the SAME path the CLI always used.

    Reuses ``mermaidseg.io.update_config_with_args`` (which reads snake_case attributes)
    by handing it a ``Namespace`` of the override values, so ``Experiment.from_args`` is
    byte-identical to the legacy ``setup_config(...) + update_config_with_args(cfg,
    args)`` and the notebook/YAML path merges the same way.
    """
    return update_config_with_args(cfg, argparse.Namespace(**overrides.model_dump()))


def load_spec(run_yaml: str | os.PathLike[str]) -> ExperimentSpec:
    """Parse a run YAML's ``config:`` block into an :class:`ExperimentSpec` (ignores
    ``job:``)."""
    raw = yaml.safe_load(os.path.expandvars(Path(run_yaml).read_text()))
    if not isinstance(raw, Mapping) or raw.get("config") is None:
        raise ValueError(f"{run_yaml}: missing top-level `config:` block")
    return ExperimentSpec.model_validate(raw["config"])


class Experiment:
    """A YAML-first experiment: a validated spec plus the run assembly built from it.

    The assembly (dataset construction -> :class:`SourceLabelRegistry` -> :class:`MetaModel` ->
    :class:`Evaluator`) is the code ``scripts/train.py::_run_training`` used to inline. Concentrating
    it here lets the CLI, notebooks, and :meth:`dry_run` build identical objects. Heavy pieces are
    built lazily and cached, so constructing an ``Experiment`` (or calling :meth:`validate`) touches
    no S3/GPU until you ask for datasets/registry/model.
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        config: RunConfigDict,
        *,
        device: torch.device | None = None,
    ):
        self.spec = spec
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dataset_dict: dict[tuple[str, str], object] | None = None
        self._registry: SourceLabelRegistry | None = None
        self._schema: ConceptSchema | None = None
        self._compute_concepts = False
        self.train_dataset: ConcatDataset | None = None
        self.val_dataset: ConcatDataset | None = None

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: ExperimentSpec, *, device: torch.device | None = None) -> Experiment:
        cfg = setup_config(spec.config_paths)
        _apply_overrides_to_config(cfg, spec.overrides)
        return cls(spec, cfg, device=device)

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, *, device: torch.device | None = None
    ) -> Experiment:
        """CLI constructor.

        Behaviorally identical to the legacy setup_config + override merge.
        """
        return cls.from_spec(_spec_from_args(args), device=device)

    @classmethod
    def from_run_yaml(
        cls, run_yaml: str | os.PathLike[str], *, device: torch.device | None = None
    ) -> Experiment:
        """Notebook/entrypoint constructor: load the ``config:`` block, ignore
        ``job:``."""
        return cls.from_spec(load_spec(run_yaml), device=device)

    @property
    def overrides(self) -> Overrides:
        return self.spec.overrides

    # -- assembly ------------------------------------------------------------------------------

    def datasets(self) -> dict[tuple[str, str], object]:
        """The ``(name, split) -> dataset`` map, built once from
        ``DATASET_REGISTRY``."""
        if self._dataset_dict is None:
            cfg = self.config
            dataset_dict: dict[tuple[str, str], object] = {}
            for name in DATASET_REGISTRY:
                for split, split_cfg in cfg.data[name].items():
                    if split_cfg is None or split_cfg == "None":
                        continue
                    dataset_dict[(name, split)] = DATASET_REGISTRY[name](
                        **split_cfg, padding=cfg.training.padding
                    )
                    print(
                        f"{name:>24s} - {split:<5s}: {len(dataset_dict[(name, split)]):>7d} samples"
                    )
            self._dataset_dict = dataset_dict
        return self._dataset_dict

    @property
    def registry(self) -> SourceLabelRegistry:
        """The :class:`SourceLabelRegistry`, built once and attached to every dataset
        instance."""
        if self._registry is None:
            cfg = self.config
            dataset_dict = self.datasets()
            concept_mapping_path = cfg.training.get("concept_mapping_path")
            _, registry_datasets = prepare_splits_for_registry(dataset_dict)
            run_sources = {ds.SOURCE_NAME for ds in registry_datasets}

            # Standard (non-CBM) mode trains only the segmentation head, so it needs neither a
            # ConceptSchema nor a concept_mapping_path; building the schema here would crash on
            # ConceptSchema.from_csv(None, ...). Guard both on training_mode.
            self._compute_concepts = cfg.training.training_mode != "standard"
            self._schema = (
                ConceptSchema.from_csv(concept_mapping_path, sources=run_sources)
                if self._compute_concepts
                else None
            )
            registry = SourceLabelRegistry(
                registry_datasets,
                target_label_subset=cfg.training.class_subset,
                compute_concepts=self._compute_concepts,
                concept_mapping_path=concept_mapping_path,
                concept_schema=self._schema,
                label_roll_up=cfg.training.get("label_roll_up", False),
            ).to(self.device)
            attach_registry(registry, dataset_dict.values())
            self._registry = registry
        return self._registry

    def _base_loader_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        loader_kwargs: dict[str, Any] = {
            "batch_size": cfg.training.batch_size,
            "num_workers": self.overrides.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": True,
            # Drop the (None, None) placeholders BaseCoralDataset returns on load failures so one
            # bad image leaves the batch instead of crashing default_collate.
            "collate_fn": BaseCoralDataset.collate_fn,
        }
        if self.overrides.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["worker_init_fn"] = worker_init_fn
        return loader_kwargs

    def dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build the ``(train, val)`` loaders (ConcatDataset over the active splits)."""
        dataset_dict = self.datasets()
        registry = self.registry  # ensure attach_registry ran before any __getitem__
        loader_kwargs = self._base_loader_kwargs()

        train_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "train"]
        val_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "val"]
        if not train_datasets:
            raise ValueError("No training datasets enabled in data config.")
        if not val_datasets:
            raise ValueError("No validation datasets enabled in data config.")
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)
        train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(self.val_dataset, shuffle=True, **loader_kwargs)

        print(f"train batches: {len(train_loader)}   val batches: {len(val_loader)}")
        if self._compute_concepts:
            assert registry.num_concepts == self._schema.num_channels
        return train_loader, val_loader

    def dataset_val_loaders(self) -> dict[str, DataLoader]:
        """Per-dataset validation loaders (e.g. ``{"mermaid": ..., "coralnet": ...}``),
        for MLflow keys like ``validation/mermaid/miou``.

        Checkpoint selection and early stopping still use the combined loader from
        :meth:`dataloaders`.
        """
        dataset_dict = self.datasets()
        _ = self.registry  # ensure attach_registry ran before any __getitem__
        loader_kwargs = {**self._base_loader_kwargs(), "drop_last": False}
        return {
            name: DataLoader(ds, shuffle=False, **loader_kwargs)
            for (name, split), ds in dataset_dict.items()
            if split == "val"
        }

    def meta_model(self) -> MetaModel:
        """Construct the :class:`MetaModel` from the registry's derived lookups."""
        cfg = self.config
        registry = self.registry
        return MetaModel(
            run_name=cfg.run_name,
            num_classes=registry.num_target_classes,
            num_concepts=registry.num_concepts or None,
            device=self.device,
            model_kwargs=cfg.model.copy(),
            training_kwargs=cfg.training.copy(),
            source_to_target_lookup=registry.source_to_target,
            source_to_concepts_lookup=registry.source_to_concepts,
            concept_matrix=registry.concept_matrix,
            conceptid2labelid=registry.conceptid2labelid(),
            concept_value2id=registry.concept_value2id,
        )

    def evaluator(self) -> Evaluator:
        """Construct the stock :class:`Evaluator` (per-class default keyed off
        training_mode)."""
        cfg = self.config
        registry = self.registry
        per_class_metrics = self.overrides.per_class_metrics
        if per_class_metrics is None:
            # On for standard segmentation; off for concept/concept-bottleneck, where taxonomic
            # ranks carry many values and would flood MLflow's metric list by default.
            per_class_metrics = cfg.training.training_mode == "standard"
        return Evaluator(
            num_classes=registry.num_target_classes,
            device=self.device,
            calculate_concept_metrics=cfg.training.training_mode != "standard",
            concept_value2id=registry.concept_value2id,
            per_class_metrics=per_class_metrics,
        )

    def dry_run(self) -> dict[str, Any]:
        """Assemble the full run (datasets -> registry -> model -> evaluator) and pull
        one batch to confirm the data path, WITHOUT training.

        Uses S3/HF/API exactly as a real run would, so it catches registry-offset,
        model-channel, and checkpoint-load errors an offline :meth:`validate` cannot.
        Returns a summary dict.
        """
        train_loader, val_loader = self.dataloaders()
        registry = self.registry
        self.meta_model()  # constructs the model — raises on wiring/shape errors
        self.evaluator()

        summary: dict[str, Any] = {
            "datasets": sorted({name for (name, _) in self.datasets()}),
            "num_target_classes": registry.num_target_classes,
            "num_concepts": registry.num_concepts,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "metric_of_interest": self.overrides.metric_of_interest,
        }
        try:
            images, labels = next(iter(train_loader))
            summary["train_batch_image_shape"] = tuple(images.shape)
            summary["train_batch_label_shape"] = tuple(labels.shape)
        except StopIteration:
            summary["train_batch"] = "empty (no non-failed samples in the first batch)"
        return summary

    # -- offline validation --------------------------------------------------------------------

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


def _cmd_validate(paths: list[Path]) -> int:
    all_ok = True
    for path in paths:
        report = Experiment.validate(path)
        print(_render_report(Path(path), report))
        all_ok = all_ok and report.ok
    return 0 if all_ok else 1


def _cmd_dry_run(path: Path) -> int:
    # Offline schema/static checks first — fail fast without spinning up S3/HF.
    report = Experiment.validate(path)
    print(_render_report(path, report))
    if not report.ok:
        return 1
    print(f"[dry-run] assembling {path} (datasets -> registry -> model -> evaluator) ...")
    try:
        summary = Experiment.from_run_yaml(path).dry_run()
    except Exception as exc:  # noqa: BLE001 — report assembly failure as the dry-run result
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        return 1
    print("[OK] assembled:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mermaidseg.experiment",
        description="Validate or dry-run a SageMaker run YAML's experiment (`config:`) block.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate", help="offline schema + static checks (no AWS/HF/network)"
    )
    validate_parser.add_argument(
        "run_yaml", type=Path, nargs="+", help="run YAML file(s) to validate"
    )
    dry_run_parser = subparsers.add_parser(
        "dry-run", help="validate, then assemble the run (needs S3/HF/creds); no training"
    )
    dry_run_parser.add_argument("run_yaml", type=Path, help="run YAML file to assemble")
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _cmd_validate(args.run_yaml)
    if args.command == "dry-run":
        return _cmd_dry_run(args.run_yaml)
    return 2


if __name__ == "__main__":
    sys.exit(main())
