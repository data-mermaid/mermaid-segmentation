"""Loader for research artifacts that are not bundles.

Rebuilds a CBM from a training checkpoint, its model config and the ``id2label`` /
``concept_id2name`` maps. The encoder comes from ``from_pretrained`` (hub or local path), so this
path serves research tooling; deployment loads a bundle (``load_bundle``) instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

import mermaidseg.model.models as mm_models
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS, parse_concept_rank
from mermaidseg.inference.bundle import _load_checkpoint, _reject_batchnorm_head
from mermaidseg.model.models import align_peft_checkpoint_state_dict

SUPPORTED_CBM_MODELS: frozenset[str] = frozenset(
    {
        "ConceptBottleneckDINOv3",
        "ConceptBottleneckDPTLoRADINOv3",
        "ConceptBottleneckDPTDINOv3",
    }
)
DEFAULT_MODEL_NAME = "ConceptBottleneckDINOv3"


def derive_concept_value2id(
    concept_id2name: Mapping[int, str],
) -> dict[str, dict[str, list[int]]]:
    """Per-rank value encodings (``not_given`` plus one-hot-like codes) from the concept names."""
    concept_value2id: dict[str, dict[str, list[int]]] = {}
    for rank in TAXONOMIC_CONCEPTS:
        entries: list[tuple[int, str]] = []
        for idx, name in sorted(concept_id2name.items(), key=lambda kv: int(kv[0])):
            parsed_rank, value = parse_concept_rank(name)
            if parsed_rank == rank:
                entries.append((idx, value))
        if not entries:
            continue
        width = len(entries)
        mapping: dict[str, list[int]] = {"not_given": [0] * width}
        for i, (_, value) in enumerate(entries):
            encoded = [1] * width
            encoded[i] = 2
            mapping[value] = encoded
        concept_value2id[rank] = mapping
    return concept_value2id


def load_checkpoint_model(
    checkpoint_path: str | Path,
    model_cfg: Mapping[str, Any],
    id2label: Mapping[int, str],
    concept_id2name: Mapping[int, str],
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Build the model ``model_cfg`` names and strict-load ``checkpoint_path`` into it.

    ``model_cfg`` is the YAML ``model`` section (``name`` selects the class, the rest are its
    constructor arguments). The checkpoint is a training checkpoint dict or a bare state dict; it
    is read on CPU, remapped with ``align_peft_checkpoint_state_dict``, loaded strictly and then
    moved to ``device``. A BatchNorm DPT head, or label maps that disagree with the weights'
    class or concept counts, raise ``ValueError``.
    """
    if not concept_id2name:
        raise ValueError("concept_id2name is required.")
    state_dict, _ = _load_checkpoint(Path(checkpoint_path))
    _reject_batchnorm_head(state_dict.keys())

    classifier = state_dict.get("concept_classifier.weight")
    if classifier is None:
        raise ValueError("Checkpoint has no 'concept_classifier.weight'; not a CBM checkpoint.")
    weight_classes, weight_concepts = int(classifier.shape[0]), int(classifier.shape[1])
    num_classes = max(int(k) for k in id2label) + 1
    if num_classes != weight_classes:
        raise ValueError(
            f"id2label implies {num_classes} classes but the checkpoint has {weight_classes}."
        )
    if len(concept_id2name) != weight_concepts:
        raise ValueError(
            f"concept_id2name lists {len(concept_id2name)} concepts but the checkpoint has "
            f"{weight_concepts}."
        )

    kwargs = dict(model_cfg)
    model_name = kwargs.pop("name", DEFAULT_MODEL_NAME)
    if model_name not in SUPPORTED_CBM_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected one of {sorted(SUPPORTED_CBM_MODELS)}"
        )
    if isinstance(kwargs.get("input_size"), list):
        kwargs["input_size"] = tuple(kwargs["input_size"])
    kwargs.setdefault("concept_value2id", derive_concept_value2id(concept_id2name))

    model = getattr(mm_models, model_name)(
        num_classes=weight_classes, num_concepts=weight_concepts, **kwargs
    )
    model.load_state_dict(align_peft_checkpoint_state_dict(state_dict, model), strict=True)
    return model.to(device).eval()
