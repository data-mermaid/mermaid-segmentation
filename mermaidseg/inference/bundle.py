"""Loading the exported bundle: a directory holding ``model.json`` and ``weights.safetensors``.

``load_bundle`` rebuilds the model from it without the hub. ``mermaidseg.export`` writes bundles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from mermaidseg.inference.preprocess import CHANNEL_ORDER, INTERPOLATION
from mermaidseg.model.models import MODEL_CLASSES

MODEL_JSON = "model.json"
WEIGHTS = "weights.safetensors"
WEIGHTS_SHA256 = "weights_sha256"
_HASH_CHUNK_BYTES = 1024 * 1024
# Renaming the model class or any ``architecture.kwargs`` key, or changing their meaning, requires
# a SCHEMA_VERSION bump: delivered bundles are rebuilt from these names.
# tests/mermaidseg/test_model_json_contract.py enforces this against
# tests/mermaidseg/fixtures/model_json_architecture/v{SCHEMA_VERSION}.json.
SCHEMA_VERSION = 1
TASK = "segmentation_cbm"


@dataclass(frozen=True)
class ModelBundle:
    """A loaded bundle: its ``model.json`` metadata and the model built from it."""

    path: Path
    metadata: dict
    model: nn.Module
    device: torch.device

    @property
    def id2label(self) -> dict[int, str]:
        """Class names by integer id."""
        return {int(k): v for k, v in self.metadata["id2label"].items()}

    @property
    def concept_id2name(self) -> dict[int, str]:
        """Concept channel names by integer id."""
        return {int(k): v for k, v in self.metadata["concept_id2name"].items()}

    @property
    def input_size(self) -> tuple[int, int]:
        """``(height, width)`` the images are resized to."""
        height, width = self.metadata["preprocessing"]["size_hw"]
        return (int(height), int(width))

    @property
    def mean(self) -> tuple[float, ...]:
        """Per-channel normalisation mean."""
        return tuple(self.metadata["preprocessing"]["mean"])

    @property
    def std(self) -> tuple[float, ...]:
        """Per-channel normalisation standard deviation."""
        return tuple(self.metadata["preprocessing"]["std"])

    @property
    def config(self) -> dict:
        """The run's resolved config, from the lineage."""
        return self.metadata["lineage"]["resolved_config"]


def _check_metadata(metadata: dict, path: Path) -> None:
    for key, expected in (("schema_version", SCHEMA_VERSION), ("task", TASK), ("weights", WEIGHTS)):
        if metadata.get(key) != expected:
            raise ValueError(f"{path}: {key} {metadata.get(key)!r} is not supported ({expected!r})")
    preprocessing = metadata["preprocessing"]
    for key, expected in (("channel_order", CHANNEL_ORDER), ("interpolation", INTERPOLATION)):
        if preprocessing.get(key) != expected:
            raise ValueError(
                f"{path}: preprocessing.{key} {preprocessing.get(key)!r} is not supported ({expected!r})"
            )


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file at ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _check_weights_checksum(metadata: dict, path: Path) -> None:
    expected = metadata.get(WEIGHTS_SHA256)
    if expected is None:
        raise ValueError(
            f"{path}: {MODEL_JSON} records no {WEIGHTS_SHA256}; "
            "re-export the bundle from its checkpoint"
        )
    actual = sha256_file(path / WEIGHTS)
    if actual != expected:
        raise ValueError(
            f"{path}: {WEIGHTS} does not match the recorded {WEIGHTS_SHA256} "
            f"({actual} != {expected})"
        )


def _build_encoder(encoder_config: dict) -> PreTrainedModel:
    """A randomly initialised float32 encoder from its recorded config; never touches the hub."""
    rest = dict(encoder_config)
    model_type = rest.pop("model_type")
    return AutoModel.from_config(AutoConfig.for_model(model_type, **rest), dtype=torch.float32)


def load_bundle(path: str | Path, device: torch.device | str) -> ModelBundle:
    """Load the bundle directory ``path`` onto ``device`` as an eval-mode model.

    The encoder is built from ``architecture.encoder_config`` and the weights come from
    ``weights.safetensors`` with a strict state-dict match, so no hub access is needed.
    ``concept_value2id`` keys are stringified in JSON: bool keys arrive as ``"True"``/``"False"``.
    The model reads only the taxonomic string keys and the widths of their value lists.

    Raises:
        ValueError: ``schema_version``, ``task``, ``weights``, ``preprocessing.channel_order`` or
            ``preprocessing.interpolation`` is not supported, ``weights_sha256`` is missing
            (re-export the bundle), or ``weights.safetensors`` does not match it.
        RuntimeError: The weights do not match the architecture's state dict.
    """
    path = Path(path)
    metadata = json.loads((path / MODEL_JSON).read_text())
    _check_metadata(metadata, path)
    _check_weights_checksum(metadata, path)
    architecture = metadata["architecture"]
    kwargs = dict(architecture["kwargs"])
    kwargs["input_size"] = tuple(kwargs["input_size"])
    model = MODEL_CLASSES[architecture["class"]](
        base_encoder=_build_encoder(architecture["encoder_config"]), **kwargs
    )
    model.load_state_dict(load_file(path / WEIGHTS), strict=True)
    device = torch.device(device)
    return ModelBundle(path=path, metadata=metadata, model=model.to(device).eval(), device=device)
