"""A tiny offline DINOv3 concept-bottleneck run: model, sidecar and checkpoint on disk.

The encoder is built from a random ``transformers`` config, so no test reaches the hub.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel

from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.export import write_sidecar
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from mermaidseg.model.models import MODEL_CLASSES

INPUT_SIZE = (64, 64)
RUN_NAME = "tiny-run"
MODEL_CLASS = "ConceptBottleneckDPTLoRADINOv3"
RANK_WIDTH = 2
CLASS_NAMES = ["ignore", "acropora", "porites", "sand", "algae"]
CONCEPT_NAMES = [f"{rank}__v{i}" for rank in TAXONOMIC_CONCEPTS for i in range(RANK_WIDTH)] + [
    "live",
    "sand",
]
ENV_ENCODER_KEYS = ("transformers_version", "_name_or_path")


def id2label() -> dict[int, str]:
    return dict(enumerate(CLASS_NAMES))


def concept_id2name() -> dict[int, str]:
    return {i + 1: name for i, name in enumerate(CONCEPT_NAMES)}


def concept_value2id() -> dict[str, dict[str, list[int]]]:
    return {
        rank: {"not_given": [0] * RANK_WIDTH, "v0": [2, 1], "v1": [1, 2]}
        for rank in TAXONOMIC_CONCEPTS
    }


def tiny_encoder() -> Any:
    config = AutoConfig.for_model(
        "dinov3_vit",
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=16,
        image_size=INPUT_SIZE[0],
        num_register_tokens=4,
    )
    return AutoModel.from_config(config)


def model_kwargs() -> dict[str, Any]:
    """The architecture kwargs the sidecar records, with ``encoder_name`` for provenance."""
    return {
        "encoder_name": "offline/tiny-dinov3",
        "num_classes": len(CLASS_NAMES),
        "num_concepts": len(CONCEPT_NAMES),
        "input_size": INPUT_SIZE,
        "concept_value2id": concept_value2id(),
        "concept_feature_dim": 8,
        "detach_concepts": False,
        "out_indices": None,
        "neck_hidden_sizes": [8, 16, 32, 32],
        "fusion_hidden_size": 8,
        "reassemble_factors": [4, 2, 1, 0.5],
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_bias": "none",
    }


def build_trained_model() -> torch.nn.Module:
    """A tiny CBM whose LoRA B matrices are non-zero, so the adapters change the output."""
    torch.manual_seed(0)
    kwargs = {k: v for k, v in model_kwargs().items() if k != "encoder_name"}
    model = MODEL_CLASSES[MODEL_CLASS](base_encoder=tiny_encoder(), **kwargs)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5)
    return model.eval()


def resolved_config() -> dict[str, Any]:
    return {
        "run_name": RUN_NAME,
        "seed": 0,
        "model": {"input_size": list(INPUT_SIZE)},
        "training": {"mixed_precision_dtype": "bfloat16", "concept_removal": None},
    }


def write_run(
    run_dir: Path,
    *,
    git_dirty: bool | None = False,
    config: dict[str, Any] | None = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> tuple[Path, torch.nn.Module]:
    """Write ``model.json`` and ``model_step1`` into ``run_dir``; return the checkpoint and model."""
    model = build_trained_model()
    encoder: Any = model.encoder
    encoder_config = {
        k: v
        for k, v in encoder.get_base_model().config.to_dict().items()
        if k not in ENV_ENCODER_KEYS
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_sidecar(
        run_dir,
        run_name=RUN_NAME,
        model_class=MODEL_CLASS,
        model_kwargs=model_kwargs(),
        encoder_config=encoder_config,
        id2label=id2label(),
        concept_id2name=concept_id2name(),
        input_size=INPUT_SIZE,
        mean=mean,
        std=std,
        lineage={
            "git_sha": "abc123",
            "git_dirty": git_dirty,
            "resolved_config": resolved_config(),
        },
    )
    checkpoint = run_dir / "model_step1"
    torch.save(
        {
            "config": copy.deepcopy(config if config is not None else resolved_config()),
            "model_state_dict": model.state_dict(),
            "epoch": 3,
            "global_step": 42,
        },
        checkpoint,
    )
    return checkpoint, model
