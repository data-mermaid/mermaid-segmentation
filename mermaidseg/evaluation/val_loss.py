"""Validation loss of a checkpoint on the val splits of a data config.

Rebuilds the training-time registry and model (``mermaidseg.run_setup``) on the ``val``
split of every dataset in the data config, at the evaluation resolution, and averages
the ``ConceptBottleneckLoss`` components over batches:

    classification              target-class cross-entropy
    kingdom ... genus           per taxonomic rank
    multi_hot                   all binary concepts together
    multi_hot/<concept>         each binary concept on its own (e.g. multi_hot/bleached)
    concepts                    ranks + multi_hot
    loss                        classification + concepts

Every run is scored with the same pixel/image reduction (``per_pixel_loss_weight`` and
``per_image_loss_weight``, 0.5/0.5 by default), whatever weights it trained with, so runs
of the loss-weight sweep stay comparable. Each dataset contributes a fixed, seeded subset
of at most ``max_images_per_dataset`` val images. A component is averaged over the
batches in which it has labelled pixels. Results are reported per dataset and overall
(all batches pooled).

With ``apply_source_mask``, the run's ``training.concept_source_mask`` zeros the masked
source x concept targets as in the training loss (used for the CoralNet train sources).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mermaidseg.dataset_reconciliation.concept_source_mask import (
    apply_concept_source_mask,
    attach_source_rows,
)
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.datasets import build_datasets, make_worker_init_fn, setup_local_cache
from mermaidseg.evaluation.reporting import write_json
from mermaidseg.io import setup_config
from mermaidseg.model.data_parallel import apply_concept_bottleneck_loss
from mermaidseg.run_setup import (
    DEFAULT_SCHEMA_CONFIG,
    build_meta_model,
    build_registry,
    load_model_weights,
    load_run_concept_source_mask,
    resolve_repo_path,
)

logger = logging.getLogger(__name__)

TOTAL_KEY = "loss"


def binary_concept_names(concept_value2id: dict[str, Any], channel_names: list[str]) -> list[str]:
    """Names of the binary channels, which follow the taxonomic ranks (as in the slices)."""
    offset = 0
    for rank in TAXONOMIC_CONCEPTS:
        if rank in concept_value2id:
            offset += len(next(iter(concept_value2id[rank].values())))
    return channel_names[offset:]


class _Accumulator:
    """Per-key running sums and batch counts."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, components: dict[str, float]) -> None:
        """Add one batch's components."""
        for key, value in components.items():
            self.sums[key] += float(value)
            self.counts[key] += 1

    def merge(self, other: _Accumulator) -> None:
        for key, value in other.sums.items():
            self.sums[key] += value
            self.counts[key] += other.counts[key]

    def means(self) -> dict[str, float]:
        return {key: self.sums[key] / self.counts[key] for key in sorted(self.sums)}


def resolve_concept_mapping_path(training_cfg: Any) -> None:
    """Rewrite a relative ``concept_mapping_path`` to its path under the repository root."""
    path = training_cfg.get("concept_mapping_path")
    if path:
        training_cfg.concept_mapping_path = str(resolve_repo_path(path))


def _check_layout(checkpoint: Path, registry: Any, channel_names: list[str]) -> None:
    """Compare against the layouts the training run wrote next to its checkpoints."""
    run_dir = Path(checkpoint).parent
    id2label_path = run_dir / "id2label.json"
    if id2label_path.is_file():
        saved = {int(k): v for k, v in json.loads(id2label_path.read_text()).items() if int(k) > 0}
        if saved != dict(registry.target_id2label):
            raise ValueError(f"target classes differ from {id2label_path}")
    concept_path = run_dir / "concept_id2name.json"
    if concept_path.is_file():
        saved_names = [
            name
            for _, name in sorted(
                json.loads(concept_path.read_text()).items(), key=lambda kv: int(kv[0])
            )
        ]
        if saved_names != channel_names:
            raise ValueError(f"concept channels differ from {concept_path}")


def evaluate_val_loss(
    *,
    checkpoint: str | Path,
    model_config: str | Path,
    data_config: str | Path,
    training_config: str | Path,
    resolution: int,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 8,
    amp: bool = False,
    schema_config: str | Path = DEFAULT_SCHEMA_CONFIG,
    max_images_per_dataset: int | None = 1000,
    per_pixel_loss_weight: float = 0.5,
    per_image_loss_weight: float = 0.5,
    seed: int = 0,
    name: str = "val_loss",
    apply_source_mask: bool = False,
) -> dict:
    """Validation loss components of ``checkpoint``, also written to ``<output_dir>/<name>/``."""
    t0 = time.time()
    cfg = setup_config(
        {"data": str(data_config), "training": str(training_config), "model": str(model_config)},
        resolution=resolution,
    )
    cfg.run_name = name
    cfg.training.pop("taxonomic_loss", None)
    cfg.training.distributed = False
    cfg.training.loss.per_pixel_loss_weight = per_pixel_loss_weight
    cfg.training.loss.per_image_loss_weight = per_image_loss_weight
    resolve_concept_mapping_path(cfg.training)

    cache_stats = setup_local_cache(cfg.data)
    dataset_dict = build_datasets(cfg.data, padding=cfg.training.padding, splits=["val"])
    if not dataset_dict:
        raise ValueError(f"{data_config} has no val split")
    registry, schema = build_registry(
        dataset_dict, cfg.training, schema_config=schema_config, device=device
    )
    channel_names = [name for _, name in sorted(schema.channel_id2name().items())]
    _check_layout(Path(checkpoint), registry, channel_names)
    cfg.training.loss.binary_concept_names = binary_concept_names(
        cast("dict[str, Any]", registry.concept_value2id), channel_names
    )

    source_mask = None
    if apply_source_mask:
        source_mask = load_run_concept_source_mask(cfg.training, schema)
        if source_mask is None:
            logger.info("[%s] training config has no concept_source_mask; targets unmasked", name)
        else:
            logger.info(
                "[%s] concept source mask %s: %d %s source x concept pairs",
                name,
                source_mask.path,
                source_mask.num_masked,
                source_mask.dataset,
            )
            dataset_dict = dict(
                zip(
                    dataset_dict,
                    attach_source_rows(list(dataset_dict.values()), source_mask),
                    strict=True,
                )
            )

    meta = build_meta_model(cfg, registry, device)
    load_model_weights(meta, checkpoint, device)
    meta.model.eval()
    use_amp = amp and device.type == "cuda"

    per_dataset: dict[str, dict] = {}
    overall = _Accumulator()
    n_images: dict[str, int] = {}
    for (dataset_name, _split), dataset in sorted(dataset_dict.items()):
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
        if max_images_per_dataset is not None:
            indices = indices[:max_images_per_dataset]
        image_indices: list[int] = sorted(indices.tolist())
        n_images[dataset_name] = len(image_indices)
        loader = DataLoader(
            Subset(dataset, image_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=make_worker_init_fn(cache_stats) if num_workers > 0 else None,
        )
        acc = _Accumulator()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{name} {dataset_name}"):
                images, source_labels = batch[0], batch[1]
                if images.numel() == 0:
                    continue
                images = images.to(device, non_blocking=True).float()
                source_labels = source_labels.to(device, non_blocking=True).long()
                target_labels = meta._to_target_labels(source_labels)
                target_concepts = meta._to_concept_labels(source_labels)
                if source_mask is not None:
                    target_concepts = apply_concept_source_mask(
                        target_concepts, source_mask.keep, batch[2]
                    )
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    outputs = meta.model(images)
                loss, _, _, components = apply_concept_bottleneck_loss(
                    outputs,
                    cast("torch.nn.Module", meta.loss),
                    None,
                    target_labels,
                    target_concepts,
                )
                acc.add({TOTAL_KEY: loss.item(), **components})
        per_dataset[dataset_name] = acc.means()
        overall.merge(acc)
        logger.info(
            "[%s] %s: loss %.4f over %d images",
            name,
            dataset_name,
            per_dataset[dataset_name].get(TOTAL_KEY, float("nan")),
            len(image_indices),
        )

    del meta
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "overall": overall.means(),
        "per_dataset": per_dataset,
        "n_images": n_images,
        "settings": {
            "data_config": str(data_config),
            "training_config": str(training_config),
            "resolution": resolution,
            "max_images_per_dataset": max_images_per_dataset,
            "per_pixel_loss_weight": per_pixel_loss_weight,
            "per_image_loss_weight": per_image_loss_weight,
            "seed": seed,
            "concept_source_mask": source_mask.path if source_mask is not None else None,
        },
        "seconds": time.time() - t0,
    }
    write_json(Path(output_dir) / name / "metrics.json", result)
    return result
