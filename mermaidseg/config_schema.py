"""Offline schema + value validation for the four split config files.

`Experiment.validate` (``mermaidseg/experiment.py``) already validates a run YAML's ``job:`` /
``config:`` / ``overrides:`` blocks, but the four *referenced* split configs
(``configs/{data,model,training,logger}_config*.yaml``) were loaded straight into a ``ConfigDict``
with no schema. A typo'd ``optimizer.type``, an invalid ``training_mode``, a missing required
field, or a bad ``model.name`` then only failed deep in ``MetaModel.__init__`` via
``getattr(...)`` -> ``AttributeError`` — on SageMaker, *after* a Docker push + job queue.

This module models each split block as a pydantic v2 schema (structural / type checks) plus a few
value-existence helpers that mirror ``MetaModel``'s ``getattr`` resolutions. It is wired into
``Experiment.validate`` as non-raising ``_check_*`` steps that accumulate a ``ValidationReport``.

Design notes:

- **Import-light on purpose.** Only ``typing``/``pydantic``/``difflib`` at module top — no
  ``torch``/``peft``/``transformers`` — so these models (and their tests) load fast and the
  torch-heavy resolution stays lazy inside the ``resolve_*`` helpers. This mirrors the existing
  lazy-import convention in ``experiment.py`` (``offline_target_universe``, ``_check_worker_sizing``).
- **Strict at the block level, open at the param level.** The four block models use
  ``extra="forbid"`` so a misspelled top-level key (``optimzer:``, ``freez_encoder:``, ``vak:``) is
  a clear error. The ``ComponentConfig`` used for ``optimizer``/``loss``/``scheduler`` uses
  ``extra="allow"`` because those params are the open-ended torch/loss *constructor kwarg* surface
  (``lr``, ``betas``, ``eps``, ``power``, ``T_max``, ``gamma`` ...): ``MetaModel`` resolves ``type``
  then splats the rest into the constructor, so forbidding here would reject valid configs. Value
  correctness for those blocks is covered by (a) ``type`` resolving to a real class here, and (b)
  ``dry_run`` actually instantiating them.
"""

from __future__ import annotations

import difflib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

TrainingModeName = Literal["standard", "concept", "concept-bottleneck"]

# _resolve_amp_dtype (meta.py) normalises by lowercasing + stripping '-'/'_', then accepts these.
_AMP_DTYPE_ALIASES = {"bf16", "bfloat16", "fp16", "float16", "half"}


class ComponentConfig(BaseModel):
    """An ``optimizer`` / ``loss`` / ``scheduler`` sub-block: a required ``type`` plus
    open params.

    ``extra="allow"``: ``MetaModel`` pops ``type`` then forwards the remaining keys to
    the resolved torch/loss constructor (``meta.py`` optimizer :238, loss :229,
    scheduler :265). Those kwargs are per-class and open-ended (``lr``,
    ``weight_decay``, ``betas``, ``power``, ``gamma`` ...), so enumerating them here
    would false-block valid configs. ``type`` *resolvability* is checked separately by
    the ``resolve_*`` helpers; kwarg correctness surfaces in ``dry_run``.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class TrainingConfig(BaseModel):
    """The ``training:`` block.

    Required fields are the ones ``MetaModel`` / ``Experiment`` read *by attribute with
    no default*, i.e. a missing value hard-crashes assembly: ``training_mode``
    (asserted, meta.py:159), ``optimizer`` (meta.py:238), ``class_subset``
    (experiment.py), ``padding`` (experiment.py), ``batch_size`` (experiment.py,
    ``_base_loader_kwargs``). Everything else is ``.get``/``.pop`` defaulted downstream
    or consumed only by the train loop, so it is optional — but still declared because
    ``extra="forbid"`` rejects any field not named here.
    """

    model_config = ConfigDict(extra="forbid")

    # Required (hard-crash if absent).
    training_mode: TrainingModeName
    optimizer: ComponentConfig
    class_subset: list[str]
    padding: int
    batch_size: int

    # Optional (defaulted downstream or train-loop-only).
    loss: ComponentConfig | None = None
    scheduler: ComponentConfig | None = None
    epochs: int | None = None
    iterations_per_train_epoch: int | None = None
    iterations_per_val_epoch: int | None = None
    label_roll_up: bool = False
    freeze_encoder: bool | None = None
    max_grad_norm: float | None = 1.0
    mixed_precision: bool = False
    mixed_precision_dtype: str | None = None
    detach_concepts: bool | None = None
    concept_mapping_path: str | None = None

    @field_validator("mixed_precision_dtype")
    @classmethod
    def _known_amp_dtype(cls, value: str | None) -> str | None:
        """Mirror ``_resolve_amp_dtype`` (meta.py), which raises on anything else at
        construction."""
        if value is None:
            return value
        normalised = str(value).lower().replace("-", "").replace("_", "")
        if normalised not in _AMP_DTYPE_ALIASES:
            raise ValueError(
                f"unsupported mixed_precision_dtype={value!r}; expected one of "
                "'bfloat16'/'bf16' or 'float16'/'fp16'/'half'"
            )
        return value

    @model_validator(mode="after")
    def _non_standard_requires_concept_mapping(self) -> TrainingConfig:
        """Non-standard modes build a ``ConceptSchema`` from ``concept_mapping_path``; a
        missing path crashes ``ConceptSchema.from_csv(None, ...)`` at registry build
        (experiment.py)."""
        if self.training_mode != "standard" and not self.concept_mapping_path:
            raise ValueError(
                "concept_mapping_path is required when training_mode is not 'standard' "
                f"(got training_mode={self.training_mode!r})"
            )
        return self


class ModelConfig(BaseModel):
    """The ``model:`` block.

    ``name``/``encoder_name``/``input_size`` are required; the DPT- and LoRA-specific
    fields are optional and typed loosely (their deeper shape is enforced by the model
    constructors, e.g. ``out_indices``/``neck_hidden_sizes`` length match).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    encoder_name: str
    input_size: list[int]

    out_indices: list[int] | None = None
    neck_hidden_sizes: list[int] | None = None
    fusion_hidden_size: int | None = None
    reassemble_factors: list[float] | None = None
    concept_feature_dim: int | None = None
    use_lora: bool | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules: list[str] | None = None
    lora_bias: str | None = None

    @field_validator("input_size")
    @classmethod
    def _input_size_is_hw(cls, value: list[int]) -> list[int]:
        if len(value) != 2:
            raise ValueError(f"input_size must be [height, width] (2 ints); got {value!r}")
        return value


class LoggerConfig(BaseModel):
    """The ``logger:`` block — a small, closed schema, so unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    experiment_name: str
    system_metrics: bool
    system_metrics_sampling_interval: int
    keep_last_n_checkpoints: int
    save_full_state_dict: bool
    log_checkpoint: int | None = None


class DatasetSplits(BaseModel):
    """One dataset's ``train:``/``val:`` splits.

    Intentionally shallow: split *values* are typed ``Any`` because a split may be ``None``, the
    string ``"None"`` (both disable it), or a mapping of dataset-specific constructor kwargs whose
    ``transform`` has already been compiled to an ``albumentations.Compose`` by
    ``preprocess_data_config``. ``extra="forbid"`` still catches a mis-keyed split (e.g. ``vak:``);
    per-dataset kwarg correctness is deferred to ``dry_run``. Registry-exact section membership is
    owned separately by ``_check_datasets``.
    """

    model_config = ConfigDict(extra="forbid")

    train: Any = None
    val: Any = None


# --------------------------------------------------------------------------------------------------
# Value-existence helpers — mirror MetaModel's getattr resolutions. Each returns a human-readable
# error hint, or None when the type/name resolves. Torch and the model/loss modules are imported
# lazily so this module stays import-light; inside Experiment.validate they are already loaded.
# --------------------------------------------------------------------------------------------------


def _did_you_mean(name: str, candidates: list[str]) -> str:
    match = difflib.get_close_matches(name, candidates, n=1)
    return f" (did you mean {match[0]!r}?)" if match else ""


def resolve_optimizer_type(type_name: str) -> str | None:
    """``optimizer.type`` must name a concrete ``torch.optim.Optimizer`` subclass
    (meta.py:238).

    Stricter than ``MetaModel``'s bare ``getattr``: the ``issubclass`` guard rejects
    resolvable but non-optimizer attributes (the abstract ``Optimizer``, or a submodule
    like ``lr_scheduler``).
    """
    import torch

    def _is_optimizer(obj: object) -> bool:
        return (
            isinstance(obj, type)
            and issubclass(obj, torch.optim.Optimizer)
            and obj is not torch.optim.Optimizer
        )

    if _is_optimizer(getattr(torch.optim, type_name, None)):
        return None
    valid = sorted(n for n in dir(torch.optim) if _is_optimizer(getattr(torch.optim, n)))
    return f"unknown optimizer.type {type_name!r}{_did_you_mean(type_name, valid)}"


def resolve_scheduler_type(type_name: str) -> str | None:
    """``scheduler.type`` must name a ``torch.optim.lr_scheduler.LRScheduler`` subclass
    (meta.py:265)."""
    import torch

    base = torch.optim.lr_scheduler.LRScheduler

    def _is_scheduler(obj: object) -> bool:
        return isinstance(obj, type) and issubclass(obj, base) and obj is not base

    if _is_scheduler(getattr(torch.optim.lr_scheduler, type_name, None)):
        return None
    valid = sorted(
        n
        for n in dir(torch.optim.lr_scheduler)
        if _is_scheduler(getattr(torch.optim.lr_scheduler, n))
    )
    return f"unknown scheduler.type {type_name!r}{_did_you_mean(type_name, valid)}"


def resolve_loss_type(type_name: str) -> str | None:
    """``loss.type`` must name a class in ``mermaidseg.model.loss`` (meta.py:229)."""
    from mermaidseg.model import loss as loss_mod

    if isinstance(getattr(loss_mod, type_name, None), type):
        return None
    valid = [
        "CrossEntropyLoss",
        "BCEWithLogitsLoss",
        "ConceptBottleneckLoss",
        "ClassWeightedFocalLoss",
    ]
    return f"unknown loss.type {type_name!r} (expected one of {valid})"


def valid_model_names() -> list[str]:
    """Public ``torch.nn.Module`` subclasses exported by ``mermaidseg.model.models`` —
    the strings accepted for ``model.name``.

    Derived by introspection so it never drifts from the module.
    """
    import torch

    from mermaidseg.model import models as models_mod

    return sorted(
        n
        for n in dir(models_mod)
        if not n.startswith("_")
        and isinstance(getattr(models_mod, n), type)
        and issubclass(getattr(models_mod, n), torch.nn.Module)
    )


def resolve_model_name(name: str) -> str | None:
    """``model.name`` must resolve to a public model class (meta.py:205
    ``getattr(models, name)``)."""
    valid = valid_model_names()
    if name in valid:
        return None
    return f"unknown model.name {name!r}{_did_you_mean(name, valid)}"
