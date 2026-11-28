"""Schema of the YAML config sections: every key is required and an unknown key raises."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from mermaidseg.data.config import DataSection

_T = TypeVar("_T")


def lookup_class(classes: Mapping[str, _T], kind: str, name: str) -> _T:
    """The entry of ``classes`` that a config names.

    Args:
        classes: Allowlist from config name to class.
        kind: What the name refers to, for the error message (``"model"``, ``"loss"``).
        name: The name from the config.

    Raises:
        ValueError: If ``name`` is not in ``classes``.
    """
    if name not in classes:
        raise ValueError(f"unknown {kind} {name!r}; expected one of {sorted(classes)}")
    return classes[name]


def to_plain_dict(value: Any) -> Any:
    """``value`` as plain containers and strings, recursively.

    Mappings become ``dict``, sequences ``list``, and dates ISO strings. Plain containers load under ``torch.load(weights_only=True)`` and dump with ``yaml.safe_dump``.
    """
    if isinstance(value, Mapping):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, datetime.date):  # also datetime.datetime
        return value.isoformat()
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LossConfig(_Strict):
    """``training.loss``."""

    type: Literal["ConceptBottleneckLoss"]
    lambda_weight: float
    ignore_index: int
    per_pixel_loss_weight: float
    per_image_loss_weight: float


class TaxonomicLossConfig(_Strict):
    """``training.taxonomic_loss``."""

    ambiguity_policy: Literal["permissive"]
    kl_weight: float
    kl_stop_grad_coarse: bool


class OptimizerConfig(_Strict):
    """``training.optimizer``."""

    type: Literal["AdamW"]
    weight_decay: float
    lora_lr: float
    dpt_lr: float


class SchedulerConfig(_Strict):
    """``training.scheduler``."""

    type: Literal["LinearLR"]
    start_factor: float
    end_factor: float
    total_iters: int
    step_every: Literal["iteration"]
    warmup_iters: int
    warmup_start_factor: float


class LoaderConfig(_Strict):
    """``training.loader``."""

    prefetch_factor: int
    val_prefetch_factor: int


class TrainingConfig(_Strict):
    """The ``training`` section."""

    detach_concepts: bool
    concept_mapping_path: str
    concept_removal: str
    concept_source_mask: str
    loss: LossConfig
    taxonomic_loss: TaxonomicLossConfig
    epochs: int
    max_grad_norm: float
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    mixed_precision: bool
    mixed_precision_dtype: Literal["bfloat16"]
    iterations_per_val_epoch: int
    iterations_per_train_epoch: int
    batch_size: int
    seed: int
    loader: LoaderConfig
    stall_dump_sec: int
    world_size: int
    grad_accum_steps: int
    train_metrics_every: int
    checkpoint_every_steps: Annotated[int, Field(gt=0)]
    class_subset: list[str]
    padding: int
    label_roll_up: bool
    label_space: str


class ModelConfig(_Strict):
    """The ``model`` section."""

    name: Literal["ConceptBottleneckDPTLoRADINOv3"]
    encoder_name: str
    input_size: list[int]
    concept_feature_dim: int
    out_indices: list[int] | None
    neck_hidden_sizes: list[int]
    fusion_hidden_size: int
    # Integer factors stay ints: an upsampling factor becomes a ConvTranspose2d kernel size.
    reassemble_factors: list[int | float]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    lora_bias: str


class WandbLoggerConfig(_Strict):
    """The ``logger`` section for ``tracker: wandb``."""

    tracker: Literal["wandb"]
    wandb_project: str
    wandb_entity: str | None
    train_log_interval: int


class MlflowLoggerConfig(_Strict):
    """The ``logger`` section for ``tracker: mlflow``."""

    tracker: Literal["mlflow"]
    experiment_name: str
    train_log_interval: int


class LoggerConfig(
    RootModel[Annotated[WandbLoggerConfig | MlflowLoggerConfig, Field(discriminator="tracker")]]
):
    """The ``logger`` section: the keys of the backend that ``tracker`` names."""


SECTION_MODELS: dict[str, type[BaseModel]] = {
    "data": DataSection,
    "model": ModelConfig,
    "training": TrainingConfig,
    "logger": LoggerConfig,
}


def validate_section(raw: Mapping[str, Any], section: str, source: str | Path) -> dict[str, Any]:
    """Validate a parsed config file that holds one section, and return the section as plain data.

    Keys the file leaves out of an optional block (a dataset, a split) stay absent. Values are
    JSON-compatible (a path becomes a string).

    Args:
        raw: The parsed file: a mapping whose only key is ``section``.
        section: The section name, a key of ``SECTION_MODELS``.
        source: Where ``raw`` came from, for the error message.

    Raises:
        ValueError: If the only top-level key is not ``section``, or the section does not match
            its schema. The message starts with ``source``.
    """
    if list(raw) != [section]:
        raise ValueError(
            f"{source}: expected the single top-level key {section!r}, got {sorted(raw)}"
        )
    try:
        validated = SECTION_MODELS[section].model_validate(raw[section])
    except ValidationError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    return validated.model_dump(mode="json", exclude_unset=True)
