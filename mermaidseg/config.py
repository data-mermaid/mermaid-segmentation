"""Schema of the YAML config sections: every key is required and an unknown key raises."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

RESERVED_DATA_KEYS = frozenset({"local_cache_dir", "local_cache_write_through"})

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
    lr: float
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

    training_mode: Literal["concept-bottleneck"]
    freeze_encoder: bool
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
    metric_of_interest: str
    loader: LoaderConfig
    max_load_failure_rate: float
    stall_dump_sec: int
    world_size: int
    grad_accum_steps: int
    train_metrics_every: int
    checkpoint_every_steps: int
    class_subset: list[str]
    padding: int
    label_roll_up: bool


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


class LoggerConfig(_Strict):
    """The ``logger`` section."""

    enable_wandb: bool
    wandb_project: str
    wandb_entity: str | None
    train_log_interval: int


class SplitsConfig(_Strict):
    """A dataset block: split name -> dataset constructor kwargs, or ``null`` to skip it.

    The dataset constructors check the kwargs. An absent split is not built.
    """

    train: dict[str, Any] | None = None
    val: dict[str, Any] | None = None
    test: dict[str, Any] | None = None


class DataConfig(_Strict):
    """The ``data`` section: reserved cache keys, ``default`` and one block per dataset.

    The dataset fields match ``mermaidseg.datasets.factory.DATASET_CLASSES``. An absent dataset is
    not built.
    """

    local_cache_write_through: bool
    local_cache_dir: str | None = None
    default: SplitsConfig | None = None
    pacific_labeled_corals: SplitsConfig | None = None
    moorea_labeled_corals: SplitsConfig | None = None
    catlin_seaview: SplitsConfig | None = None
    mermaid: SplitsConfig | None = None
    coralnet: SplitsConfig | None = None
    coralscapes_v2: SplitsConfig | None = None
    benthos_yuval: SplitsConfig | None = None
    ucsd_mosaics: SplitsConfig | None = None


SECTION_MODELS: dict[str, type[BaseModel]] = {
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "logger": LoggerConfig,
}
