"""Sidecar, export and loading of portable concept-bottleneck model bundles.

A run's sidecar ``model.json`` (written at training start, no weights) holds what is needed to
rebuild and interpret the model. ``export_bundle`` turns any checkpoint of that run into a bundle
directory: ``model.json`` + ``weights.safetensors`` (LoRA merged) + optionally ``model.pt2``.
``load_bundle`` rebuilds ``ConceptBottleneckDPTDINOv3`` from the stored encoder config, offline.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

import mermaidseg.model.models as mm_models
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS, parse_concept_rank
from mermaidseg.inference.export import export_graph
from mermaidseg.model.models import ConceptBottleneckDPTDINOv3, align_peft_checkpoint_state_dict

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TASK_NAME = "segmentation_cbm"
INFERENCE_CLASS = "ConceptBottleneckDPTDINOv3"
# Trained class name -> whether its encoder carries LoRA adapters to merge.
TRAINED_CLASSES: dict[str, bool] = {
    "ConceptBottleneckDPTLoRADINOv3": True,
    "ConceptBottleneckDPTDINOv3": False,
}
MANIFEST_NAME = "model.json"
WEIGHTS_NAME = "weights.safetensors"
GRAPH_NAME = "model.pt2"
FORMATS = ("auto", "checkpoint", "graph")

LORA_KEYS = ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "lora_bias")
# Constructor arguments that the bundle supplies itself or that do not shape the architecture.
_NON_ARCHITECTURE_KEYS = frozenset(
    {"encoder_name", "num_classes", "num_concepts", "concept_value2id", "detach_concepts"}
)
_BATCHNORM_BUFFERS = (".running_mean", ".running_var", ".num_batches_tracked")
# Bundle weights name DINOv3 blocks as transformers 4.x does (encoder.layer.*); 5.x nests them
# one level deeper (encoder.model.layer.*).
_ENCODER_PREFIX = "encoder."
_NESTED_ENCODER_PREFIX = "encoder.model."


class ManifestError(ValueError):
    """A sidecar, bundle manifest or its weights do not describe a loadable model."""


@dataclass(frozen=True)
class ModelBundle:
    """A loaded bundle: its manifest plus an eager model or a ``torch.export`` program."""

    path: Path
    metadata: dict[str, Any]
    model: torch.nn.Module | torch.export.ExportedProgram
    format: str
    device: torch.device

    @property
    def classes(self) -> list[str]:
        return list(self.metadata["classes"])

    @property
    def concept_names(self) -> list[str]:
        return list(self.metadata["concepts"]["names"])

    @property
    def num_classes(self) -> int:
        return len(self.metadata["classes"])

    @property
    def num_concepts(self) -> int:
        return len(self.metadata["concepts"]["names"])

    @property
    def rank_widths(self) -> dict[str, int]:
        concepts = self.metadata["concepts"]
        return dict(zip(concepts["rank_order"], concepts["rank_widths"], strict=True))

    @property
    def input_size(self) -> tuple[int, int]:
        height, width = self.metadata["input"]["size_hw"]
        return int(height), int(width)

    @property
    def mean(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.metadata["input"]["mean"])

    @property
    def std(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.metadata["input"]["std"])

    @property
    def patch_size(self) -> int:
        return int(self.metadata["encoder_config"]["patch_size"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _constructor_defaults(cls: type) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters.values()
    return {
        p.name: p.default
        for p in params
        if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _index_ordered_classes(id2label: Mapping[Any, str]) -> list[str]:
    by_index = {int(k): v for k, v in id2label.items()}
    if sorted(by_index) != list(range(len(by_index))):
        raise ValueError("id2label keys must be the contiguous class indices 0..N-1.")
    return [by_index[i] for i in range(len(by_index))]


def _rank_widths_from_value2id(
    concept_value2id: Mapping[str, Mapping[str, Sequence[int]]],
) -> list[int]:
    widths = []
    for rank in TAXONOMIC_CONCEPTS:
        if rank not in concept_value2id:
            raise ValueError(
                f"concept_value2id has no {rank!r} rank; the model activates every rank."
            )
        widths.append(len(next(iter(concept_value2id[rank].values()))))
    return widths


def _rank_widths_from_names(names: Sequence[str]) -> list[int]:
    ranks = [parse_concept_rank(name)[0] for name in names]
    return [ranks.count(rank) for rank in TAXONOMIC_CONCEPTS]


def write_sidecar(
    out_dir: str | Path,
    *,
    run_name: str,
    model_cfg: Mapping[str, Any],
    id2label: Mapping[Any, str],
    concept_id2name: Mapping[Any, str],
    concept_value2id: Mapping[str, Mapping[str, Sequence[int]]],
    input_size: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> Path:
    """Write the run's weight-free ``model.json`` into ``out_dir`` and return its path.

    ``model_cfg`` is the run's model config (including ``name``); the label maps come from the
    in-memory registry. Concept names are stored in concept-index (sorted id) order. The
    environment at this call (the run start) is recorded as ``trained_with``.
    """
    cfg = dict(model_cfg)
    trained_class = cfg.get("name")
    if trained_class not in TRAINED_CLASSES:
        raise ValueError(
            f"model_cfg name {trained_class!r} is not exportable; expected one of "
            f"{sorted(TRAINED_CLASSES)}."
        )
    trained_defaults = _constructor_defaults(getattr(mm_models, trained_class))
    kwargs = {
        key: cfg.get(key, default)
        for key, default in _constructor_defaults(ConceptBottleneckDPTDINOv3).items()
        if key not in _NON_ARCHITECTURE_KEYS
    }
    lora = (
        {key: cfg.get(key, trained_defaults[key]) for key in LORA_KEYS}
        if TRAINED_CLASSES[trained_class]
        else None
    )
    names = [concept_id2name[k] for k in sorted(concept_id2name, key=int)]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_NAME,
        "run_name": run_name,
        "architecture": {
            "class": INFERENCE_CLASS,
            "trained_class": trained_class,
            "encoder_name": cfg.get("encoder_name", trained_defaults["encoder_name"]),
            "kwargs": _jsonable(kwargs),
            "lora": _jsonable(lora),
        },
        "input": {
            "size_hw": [int(input_size[0]), int(input_size[1])],
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "channel_order": "RGB",
        },
        "classes": _index_ordered_classes(id2label),
        "concepts": {
            "names": names,
            "rank_order": list(TAXONOMIC_CONCEPTS),
            "rank_widths": _rank_widths_from_value2id(concept_value2id),
        },
        "trained_with": _environment(),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read model manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: schema_version={manifest.get('schema_version')!r} is incompatible; "
            f"this loader reads schema_version={SCHEMA_VERSION}."
        )
    if manifest.get("task") != TASK_NAME:
        raise ManifestError(f"{path}: task={manifest.get('task')!r}, expected {TASK_NAME!r}.")
    arch_class = manifest.get("architecture", {}).get("class")
    if arch_class != INFERENCE_CLASS:
        raise ManifestError(
            f"{path}: architecture class {arch_class!r} is not supported; only "
            f"{INFERENCE_CLASS!r} bundles load."
        )
    return manifest


def _reject_batchnorm_head(keys: Sequence[str] | Any) -> None:
    batchnorm = sorted(
        k for k in keys if k.startswith("dpt_head.") and k.endswith(_BATCHNORM_BUFFERS)
    )
    if batchnorm:
        raise ManifestError(
            "Checkpoint DPT head uses BatchNorm (found e.g. "
            f"{batchnorm[0]!r}); the current head uses GroupNorm, so these weights cannot load."
        )


def _validate_against_weights(
    manifest: Mapping[str, Any], keys: Sequence[str] | Any, classifier_shape: Sequence[int]
) -> None:
    """Check the manifest's label maps against the weight shapes and concept layout."""
    _reject_batchnorm_head(keys)
    num_classes, num_concepts = int(classifier_shape[0]), int(classifier_shape[1])
    classes = manifest["classes"]
    concepts = manifest["concepts"]
    if len(classes) != num_classes:
        raise ManifestError(
            f"Manifest lists {len(classes)} classes but the weights have {num_classes}."
        )
    if len(concepts["names"]) != num_concepts:
        raise ManifestError(
            f"Manifest lists {len(concepts['names'])} concept names but the weights have "
            f"{num_concepts} concepts."
        )
    if list(concepts["rank_order"]) != list(TAXONOMIC_CONCEPTS):
        raise ManifestError(
            f"Manifest rank_order {concepts['rank_order']} differs from TAXONOMIC_CONCEPTS "
            f"{TAXONOMIC_CONCEPTS}."
        )
    derived = _rank_widths_from_names(concepts["names"])
    if derived != list(concepts["rank_widths"]):
        raise ManifestError(
            f"Concept names give per-rank widths {derived} but the manifest rank_widths are "
            f"{concepts['rank_widths']}: the concept label map is stale for these weights."
        )


def _canonical_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename transformers-5.x nested encoder keys to the canonical 4.x layout."""
    return {
        _ENCODER_PREFIX + key[len(_NESTED_ENCODER_PREFIX) :]
        if key.startswith(_NESTED_ENCODER_PREFIX)
        else key: value
        for key, value in state_dict.items()
    }


def _to_module_layout(
    state_dict: Mapping[str, torch.Tensor], module_keys: set[str]
) -> dict[str, torch.Tensor]:
    """Rename canonical encoder keys to the nested layout where the running module nests them."""
    remapped = {}
    for key, value in state_dict.items():
        if key not in module_keys and key.startswith(_ENCODER_PREFIX):
            nested = _NESTED_ENCODER_PREFIX + key[len(_ENCODER_PREFIX) :]
            if nested in module_keys:
                key = nested
        remapped[key] = value
    return remapped


def _build_inference_model(
    manifest: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]
) -> ConceptBottleneckDPTDINOv3:
    classifier = state_dict.get("concept_classifier.weight")
    if classifier is None:
        raise ManifestError("Weights have no 'concept_classifier.weight'; not a CBM checkpoint.")
    _validate_against_weights(manifest, state_dict.keys(), classifier.shape)
    model = ConceptBottleneckDPTDINOv3(
        encoder_config=manifest["encoder_config"], **_model_kwargs(manifest)
    )
    module_keys = set(model.state_dict())
    weights = _to_module_layout(_canonical_state_dict(state_dict), module_keys)
    missing = sorted(module_keys - weights.keys())
    unexpected = sorted(weights.keys() - module_keys)
    if missing or unexpected:
        raise ManifestError(
            f"Bundle weights do not fit the {INFERENCE_CLASS} built by the running transformers "
            f"{_installed_version('transformers')}: missing {missing[:3]} ({len(missing)} in "
            f"all), unexpected {unexpected[:3]} ({len(unexpected)} in all). The bundle was "
            "trained with transformers "
            f"{manifest.get('trained_with', {}).get('transformers', 'unknown')} and exported "
            f"with {manifest.get('exported_with', {}).get('transformers', 'unknown')}."
        )
    model.load_state_dict(weights, strict=True)
    return model.eval()


def _model_kwargs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = dict(manifest["architecture"]["kwargs"])
    kwargs["input_size"] = tuple(kwargs["input_size"])
    concepts = manifest["concepts"]
    # The concept activation reads only each rank's width from concept_value2id.
    kwargs["concept_value2id"] = {
        rank: {"not_given": [0] * width}
        for rank, width in zip(concepts["rank_order"], concepts["rank_widths"], strict=True)
    }
    kwargs["num_classes"] = len(manifest["classes"])
    kwargs["num_concepts"] = len(concepts["names"])
    return kwargs


def _git(*args: str) -> str | None:
    """Stdout of a git command in the mermaidseg checkout, or None when git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s unavailable: %s", " ".join(args), exc)
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _installed_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unknown"


def _environment() -> dict[str, Any]:
    """The mermaidseg commit (and whether tracked files differ from it) plus library versions."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "mermaidseg": _git("rev-parse", "HEAD") or "unknown",
        "mermaidseg_dirty": None if status is None else bool(status),
        "torch": torch.__version__,
        "transformers": _installed_version("transformers"),
        "peft": _installed_version("peft"),
    }


def _load_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, torch.Tensor], Mapping[str, Any] | None]:
    """Return the checkpoint's state dict and its run config (None for a bare state dict)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = None
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config")
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise ManifestError(f"{checkpoint_path} holds neither a checkpoint dict nor a state dict.")
    return dict(checkpoint), config if isinstance(config, Mapping) else None


def _check_checkpoint_matches_sidecar(
    manifest: Mapping[str, Any], config: Mapping[str, Any] | None, checkpoint_path: Path
) -> None:
    """Refuse a checkpoint whose saved run config names another run or architecture.

    Equal weight shapes do not prove the sidecar's label maps belong to these weights; the run
    name and the model config the checkpoint was trained with do.
    """
    if config is None:
        logger.warning(
            "%s carries no run config; its run_name and architecture are not checked against "
            "the sidecar.",
            checkpoint_path,
        )
        return
    mismatches = []
    checkpoint_run = config.get("run_name")
    if checkpoint_run is None:
        logger.warning(
            "%s config has no run_name; the run is not checked against the sidecar.",
            checkpoint_path,
        )
    elif checkpoint_run != manifest.get("run_name"):
        mismatches.append(
            f"run_name: checkpoint {checkpoint_run!r}, sidecar {manifest.get('run_name')!r}"
        )
    model_cfg = config.get("model")
    if isinstance(model_cfg, Mapping):
        arch = manifest["architecture"]
        expected = {
            "name": arch["trained_class"],
            "encoder_name": arch["encoder_name"],
            **arch["kwargs"],
            **(arch["lora"] or {}),
        }
        mismatches += [
            f"{key}: checkpoint {_jsonable(model_cfg[key])!r}, sidecar {value!r}"
            for key, value in expected.items()
            if key in model_cfg and _jsonable(model_cfg[key]) != value
        ]
    else:
        logger.warning(
            "%s config has no model section; the architecture is not checked against the sidecar.",
            checkpoint_path,
        )
    if mismatches:
        raise ManifestError(
            f"{checkpoint_path} does not belong to the sidecar's run: " + "; ".join(mismatches)
        )


def export_bundle(
    checkpoint_path: str | Path,
    sidecar_path: str | Path,
    out_dir: str | Path,
    *,
    graph: bool = False,
) -> Path:
    """Export a run checkpoint plus its sidecar into a bundle directory and return it.

    A checkpoint dict's saved config must name the sidecar's ``run_name`` and architecture, or
    ``ManifestError`` is raised; a bare state dict is exported unchecked, with a warning. The
    trained model is rebuilt as trained (``encoder_name`` config from the hub or a local
    path; its weights come from the checkpoint), LoRA adapters are merged, and the merged weights
    are strict-loaded into the inference class before anything is written. ``graph=True`` adds a
    parity-gated ``torch.export`` program. The bundle is written to ``<out_dir>.partial`` with
    ``model.json`` last and replaces ``out_dir`` only once complete, so a failed export leaves
    any previous bundle at ``out_dir`` untouched. Weights are stored with the canonical
    (transformers 4.x) encoder key layout, and ``exported_with`` records this environment.
    """
    manifest = _read_manifest(Path(sidecar_path))
    state_dict, run_config = _load_checkpoint(Path(checkpoint_path))
    _check_checkpoint_matches_sidecar(manifest, run_config, Path(checkpoint_path))
    _reject_batchnorm_head(state_dict.keys())

    arch = manifest["architecture"]
    encoder_config = AutoConfig.from_pretrained(
        arch["encoder_name"], token=os.environ.get("HF_TOKEN")
    )
    trained_kwargs = _model_kwargs(manifest)
    trained_kwargs.update(arch.get("lora") or {})
    trained = getattr(mm_models, arch["trained_class"])(
        encoder_config=encoder_config, **trained_kwargs
    )
    trained.load_state_dict(align_peft_checkpoint_state_dict(state_dict, trained), strict=True)
    if TRAINED_CLASSES[arch["trained_class"]]:
        trained.encoder = trained.encoder.merge_and_unload()
    merged = _canonical_state_dict(
        {k: v.detach().contiguous() for k, v in trained.state_dict().items()}
    )

    bundle_manifest = {
        **manifest,
        "exported_with": _environment(),
        "encoder_config": _jsonable(encoder_config.to_dict()),
        "weights": {"checkpoint": WEIGHTS_NAME, "graph": None},
    }
    model = _build_inference_model(bundle_manifest, merged)

    out_dir = Path(out_dir)
    staging = out_dir.with_name(out_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        save_file(merged, staging / WEIGHTS_NAME)
        if graph:
            export_graph(model, bundle_manifest["input"]["size_hw"], staging / GRAPH_NAME)
            bundle_manifest["weights"]["graph"] = GRAPH_NAME
        (staging / MANIFEST_NAME).write_text(
            json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _replace_dir(staging, out_dir)
    return out_dir


def _replace_dir(src: Path, dst: Path) -> None:
    """Move the complete bundle ``src`` to ``dst``, setting any previous ``dst`` aside first.

    ``dst`` never holds a mix of old and new files: it is either the old bundle, absent, or
    the new bundle.
    """
    if dst.exists():
        previous = dst.with_name(dst.name + ".old")
        if previous.exists():
            shutil.rmtree(previous)
        dst.rename(previous)
        src.rename(dst)
        shutil.rmtree(previous)
    else:
        src.rename(dst)


def load_bundle(
    path: str | Path, device: torch.device | str = "cpu", format: str = "auto"
) -> ModelBundle:
    """Load a bundle directory as an eager model (``"checkpoint"``) or graph (``"graph"``).

    ``"auto"`` picks the graph when the bundle has one. The graph is single-image serving only:
    it takes a batch of one, has no concept logits or internal features, and refuses ``amp``.
    Evaluation (batched forwards, linear-probe features) must request ``format="checkpoint"``.
    The manifest is validated against the weights either way; a mismatch raises
    ``ManifestError``. Checkpoint weights are remapped to the running transformers' encoder key
    layout.
    """
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}; got {format!r}")
    path = Path(path)
    manifest = _read_manifest(path / MANIFEST_NAME)
    weights_path = path / manifest["weights"]["checkpoint"]
    graph_name = manifest["weights"].get("graph")
    if format == "auto":
        format = "graph" if graph_name and (path / graph_name).exists() else "checkpoint"
    device = torch.device(device)

    if format == "checkpoint":
        model: torch.nn.Module | torch.export.ExportedProgram = _build_inference_model(
            manifest, load_file(weights_path, device="cpu")
        ).to(device)
    else:
        if not graph_name:
            raise ManifestError(f"{path} has no exported graph; load it with format='checkpoint'.")
        with safe_open(weights_path, framework="pt") as weights:
            keys = list(weights.keys())
            if "concept_classifier.weight" not in keys:
                raise ManifestError("Weights have no 'concept_classifier.weight'.")
            shape = weights.get_slice("concept_classifier.weight").get_shape()
        _validate_against_weights(manifest, keys, shape)
        model = torch.export.load(path / graph_name)

    return ModelBundle(path=path, metadata=manifest, model=model, format=format, device=device)
