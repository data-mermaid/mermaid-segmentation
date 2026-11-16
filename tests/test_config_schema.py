"""Structural unit tests for ``mermaidseg.config_schema``.

Torch-free (pure pydantic), so they run under just ``--group tests``. Generalised from
the branch-172 ``StandardTrainingConfig`` suite to all three training modes and all four
block models. Value-existence resolution (optimizer/loss/scheduler/model
``type``/``name`` against the real classes) is exercised at the integration level in
``tests/test_experiment.py`` instead, since it imports torch.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mermaidseg.config_schema import (
    DatasetSplits,
    LoggerConfig,
    ModelConfig,
    TrainingConfig,
)


def _training(**over) -> dict:
    """A minimally-valid standard-mode training block; override/extend via kwargs."""
    base = {
        "training_mode": "standard",
        "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 0.01},
        "class_subset": ["Acropora", "Sand"],
        "padding": 3,
        "batch_size": 4,
    }
    base.update(over)
    return base


# --- TrainingConfig: required fields --------------------------------------------------------------


def test_valid_standard_training_validates():
    TrainingConfig.model_validate(_training())


def test_full_standard_training_validates():
    TrainingConfig.model_validate(
        _training(
            loss={"type": "CrossEntropyLoss", "ignore_index": 0},
            scheduler={
                "type": "PolynomialLR",
                "power": 1,
                "total_iters": 200,
                "warmup_iters": 2000,
            },
            epochs=200,
            iterations_per_train_epoch=1000,
            iterations_per_val_epoch=200,
            label_roll_up=True,
            freeze_encoder=True,
            max_grad_norm=0.5,
            mixed_precision=True,
            mixed_precision_dtype="bfloat16",
        )
    )


def test_valid_concept_bottleneck_validates():
    TrainingConfig.model_validate(
        _training(
            training_mode="concept-bottleneck",
            concept_mapping_path="../configs/class_to_concepts.csv",
            detach_concepts=True,
            loss={"type": "ConceptBottleneckLoss", "lambda_weight": 0.1, "ignore_index": 0},
        )
    )


@pytest.mark.parametrize("mode", ["concept", "concept-bottleneck"])
def test_non_standard_mode_requires_concept_mapping(mode):
    with pytest.raises(ValidationError, match="concept_mapping_path is required"):
        TrainingConfig.model_validate(_training(training_mode=mode))


def test_wrong_training_mode_literal_raises():
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_training(training_mode="cbm"))


@pytest.mark.parametrize(
    "missing", ["training_mode", "optimizer", "class_subset", "padding", "batch_size"]
)
def test_missing_required_training_field_raises(missing):
    kwargs = _training()
    del kwargs[missing]
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(kwargs)


def test_unknown_training_key_is_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_training(freez_encoder=True))  # typo of freeze_encoder


# --- TrainingConfig: mixed_precision_dtype + component openness -----------------------------------


@pytest.mark.parametrize("dtype", ["bfloat16", "bf16", "float16", "fp16", "half", None])
def test_accepted_amp_dtypes(dtype):
    TrainingConfig.model_validate(_training(mixed_precision_dtype=dtype))


def test_bad_amp_dtype_raises():
    with pytest.raises(ValidationError, match="mixed_precision_dtype"):
        TrainingConfig.model_validate(_training(mixed_precision_dtype="float8"))


def test_component_params_are_open():
    """Optimizer/loss/scheduler sub-blocks accept arbitrary torch/loss constructor
    kwargs."""
    tc = TrainingConfig.model_validate(
        _training(
            optimizer={"type": "SGD", "lr": 0.1, "momentum": 0.9, "nesterov": True},
            loss={"type": "ClassWeightedFocalLoss", "gamma": 2.0, "weight_path": "w.json"},
        )
    )
    assert tc.optimizer.type == "SGD"


def test_component_missing_type_raises():
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_training(optimizer={"lr": 5e-5}))


# --- ModelConfig ----------------------------------------------------------------------------------


def _model(**over) -> dict:
    base = {
        "name": "LinearDINOv3",
        "encoder_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "input_size": [512, 512],
    }
    base.update(over)
    return base


def test_valid_model_validates():
    ModelConfig.model_validate(_model())


def test_valid_lora_model_validates():
    ModelConfig.model_validate(
        _model(use_lora=True, lora_r=8, lora_alpha=16, lora_target_modules=["q_proj", "v_proj"])
    )


def test_model_missing_encoder_name_raises():
    kwargs = _model()
    del kwargs["encoder_name"]
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(kwargs)


def test_model_bad_input_size_raises():
    with pytest.raises(ValidationError, match="input_size"):
        ModelConfig.model_validate(_model(input_size=[512, 512, 3]))


def test_unknown_model_key_is_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(_model(encodername="typo"))


# --- LoggerConfig ---------------------------------------------------------------------------------


def _logger(**over) -> dict:
    base = {
        "uri": "segmentation",
        "experiment_name": "mermaid",
        "system_metrics": True,
        "system_metrics_sampling_interval": 10,
        "keep_last_n_checkpoints": 3,
        "save_full_state_dict": False,
    }
    base.update(over)
    return base


def test_valid_logger_validates():
    LoggerConfig.model_validate(_logger())


def test_logger_log_checkpoint_is_optional():
    LoggerConfig.model_validate(_logger(log_checkpoint=1))


def test_unknown_logger_key_is_rejected():
    with pytest.raises(ValidationError):
        LoggerConfig.model_validate(_logger(sytem_metrics=True))  # typo of system_metrics


# --- DatasetSplits --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "splits", [{}, {"train": None, "val": None}, {"train": {"whitelist_sources": [1]}}]
)
def test_valid_dataset_splits(splits):
    DatasetSplits.model_validate(splits)


def test_disabled_split_string_none_is_accepted():
    DatasetSplits.model_validate({"train": "None", "val": "None"})


def test_unknown_split_key_is_rejected():
    with pytest.raises(ValidationError):
        DatasetSplits.model_validate({"vak": None})  # typo of val
