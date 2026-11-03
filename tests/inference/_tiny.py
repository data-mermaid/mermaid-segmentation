"""Tiny offline DINOv3 CBM builders shared by the inference tests.

Every model is built from a random ``transformers`` config saved to a temp directory, so no test
downloads weights or reaches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.model.models import ConceptBottleneckDPTDINOv3, ConceptBottleneckDPTLoRADINOv3

INPUT_SIZE = (64, 64)
RUN_NAME = "tiny-run"
RANK_WIDTH = 2
TAIL_CONCEPTS = ["live", "sand"]
CLASS_NAMES = ["ignore", "acropora", "porites", "sand", "algae"]


def concept_id2name() -> dict[int, str]:
    """1-indexed concept map: two values per taxonomic rank, then the binary tail."""
    names = [f"{rank}__v{i}" for rank in TAXONOMIC_CONCEPTS for i in range(RANK_WIDTH)]
    names += TAIL_CONCEPTS
    return {i + 1: name for i, name in enumerate(names)}


def concept_value2id() -> dict[str, dict[str, list[int]]]:
    mapping: dict[str, dict[str, list[int]]] = {}
    for rank in TAXONOMIC_CONCEPTS:
        values: dict[str, list[int]] = {"not_given": [0] * RANK_WIDTH}
        for i in range(RANK_WIDTH):
            encoded = [1] * RANK_WIDTH
            encoded[i] = 2
            values[f"v{i}"] = encoded
        mapping[rank] = values
    return mapping


def id2label() -> dict[int, str]:
    return dict(enumerate(CLASS_NAMES))


def save_tiny_encoder(directory: Path) -> Path:
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
    config.save_pretrained(directory)
    return directory


def tiny_model_cfg(encoder_dir: Path) -> dict[str, Any]:
    return {
        "name": "ConceptBottleneckDPTLoRADINOv3",
        "encoder_name": str(encoder_dir),
        "input_size": list(INPUT_SIZE),
        "concept_feature_dim": 8,
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


def write_tiny_sidecar(out_dir: Path, model_cfg: dict[str, Any], run_name: str = RUN_NAME) -> Path:
    from mermaidseg.inference import IMAGENET_MEAN, IMAGENET_STD, write_sidecar

    return write_sidecar(
        out_dir,
        run_name=run_name,
        model_cfg=model_cfg,
        id2label=id2label(),
        concept_id2name=concept_id2name(),
        concept_value2id=concept_value2id(),
        input_size=INPUT_SIZE,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def build_tiny_cbm(cls: type[torch.nn.Module], model_cfg: dict[str, Any]) -> torch.nn.Module:
    torch.manual_seed(0)
    kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", "encoder_name")}
    kwargs["input_size"] = tuple(kwargs["input_size"])
    return cls(
        encoder_config=AutoConfig.from_pretrained(model_cfg["encoder_name"]),
        num_classes=len(CLASS_NAMES),
        num_concepts=len(concept_id2name()),
        concept_value2id=concept_value2id(),
        **kwargs,
    )


def build_trained_lora_model(model_cfg: dict[str, Any]) -> ConceptBottleneckDPTLoRADINOv3:
    """A LoRA CBM with non-zero adapters, so merging the adapters changes the weights."""
    model = build_tiny_cbm(ConceptBottleneckDPTLoRADINOv3, model_cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5)
    return model.eval()


class NestedLayersCBM(ConceptBottleneckDPTDINOv3):
    """The inference CBM with its DINOv3 blocks at ``encoder.model.layer``, as transformers 5.x.

    transformers 4.x registers them at ``encoder.layer``; the blocks also stay reachable as
    ``encoder.layer`` (outside the module registry) so the encoder forward still runs.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if "layer" not in self.encoder._modules:
            return  # transformers 5.x: the blocks are already at ``encoder.model.layer``
        layers = self.encoder._modules.pop("layer")
        holder = torch.nn.Module()
        holder.layer = layers
        self.encoder.model = holder
        self.encoder.__dict__["layer"] = layers
