"""Validation loss of an exported bundle on the val splits of a data config.

Rebuilds the training-time registry (``mermaidseg.run_setup``) on the ``val`` block of every
dataset in the data config, read from the synced cache, at the bundle's input size, around the bundle's model, with the
training and model sections of the bundle's run config, and averages
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

import logging
import time
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mermaidseg.data.config import DataSection
from mermaidseg.data.factory import build_datasets, load_benthic_hierarchy
from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    apply_concept_source_mask,
    attach_source_rows,
)
from mermaidseg.evaluation.reporting import write_json
from mermaidseg.inference.bundle import ModelBundle
from mermaidseg.io import REPO_ROOT, ConfigDict, setup_config
from mermaidseg.model.data_parallel import apply_concept_bottleneck_loss
from mermaidseg.model.distributed import components_to_floats
from mermaidseg.run_setup import (
    DEFAULT_SCHEMA_CONFIG,
    build_meta_model,
    build_registry,
    load_run_concept_source_mask,
)
from mermaidseg.seeding import make_worker_seeder

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


def _first_difference(expected: Sequence[str], actual: Sequence[str]) -> str:
    """Describe the first position where two name lists differ."""
    for i, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return f"position {i}: bundle {left!r} vs registry {right!r}"
    return f"lengths: bundle {len(expected)} vs registry {len(actual)}"


def _check_layout(bundle: ModelBundle, registry: Any, channel_names: list[str]) -> None:
    """Raise ValueError naming the first class or concept the bundle and registry disagree on."""
    classes = {k: v for k, v in bundle.id2label.items() if k > 0}
    target = dict(registry.target_id2label)
    if classes != target:
        ids = sorted(classes.keys() | target.keys())
        differing = next(i for i in ids if classes.get(i) != target.get(i))
        raise ValueError(
            f"target classes differ from the bundle at id {differing}: "
            f"bundle {classes.get(differing)!r} vs registry {target.get(differing)!r}"
        )
    names = [name for _, name in sorted(bundle.concept_id2name.items())]
    if names != channel_names:
        raise ValueError(
            f"concept channels differ from the bundle at {_first_difference(names, channel_names)}"
        )


def val_blocks_only(section: DataSection) -> DataSection:
    """Return ``section`` restricted to the datasets with a val block, each without its other blocks."""
    return section.model_copy(
        update={
            "datasets": {
                name: entry.model_copy(update={"train": None, "test": None})
                for name, entry in section.datasets.items()
                if entry.val is not None
            }
        }
    )


def _bundle_config(data_config: str | Path, bundle: ModelBundle) -> ConfigDict:
    """Run config: the data section from ``data_config``, model and training from ``bundle``."""
    cfg = setup_config({"data": str(data_config)}, resolution=bundle.input_size[0])
    run = ConfigDict(bundle.config)
    cfg.model = run.model.copy()
    cfg.model.input_size = list(bundle.input_size)
    cfg.training = run.training.copy()
    return cfg


def evaluate_val_loss(
    *,
    bundle: ModelBundle,
    data_config: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 8,
    amp_dtype: torch.dtype | None = None,
    schema_config: str | Path = DEFAULT_SCHEMA_CONFIG,
    max_images_per_dataset: int | None = 1000,
    per_pixel_loss_weight: float = 0.5,
    per_image_loss_weight: float = 0.5,
    seed: int = 0,
    name: str = "val_loss",
    apply_source_mask: bool = False,
    image_ids: Mapping[tuple[str, str], Collection[str]] | None = None,
) -> dict:
    """Validation loss components of the loaded ``bundle``, also written to ``<output_dir>/<name>/``.

    ``cache_dir`` is the directory ``scripts/sync_data.py`` synced ``data_config`` into, and
    ``image_ids`` keeps, per ``(dataset, "val")``, only those images (see
    :func:`mermaidseg.data.factory.build_datasets`).
    """
    t0 = time.time()
    cfg = _bundle_config(data_config, bundle)
    resolution = bundle.input_size[0]
    cfg.run_name = name
    cfg.training.pop("taxonomic_loss", None)
    cfg.training.distributed = False
    cfg.training.loss.per_pixel_loss_weight = per_pixel_loss_weight
    cfg.training.loss.per_image_loss_weight = per_image_loss_weight

    section = DataSection.model_validate(cfg.data)
    dataset_dict = build_datasets(
        val_blocks_only(section),
        Path(cache_dir),
        repo_root=REPO_ROOT,
        padding=cfg.training.padding,
        input_size=cfg.model.input_size,
        image_ids=image_ids,
    )
    if not dataset_dict:
        raise ValueError(f"{data_config} has no val split")
    registry, schema = build_registry(
        dataset_dict,
        cfg.training,
        schema_config=schema_config,
        benthic_hierarchy=load_benthic_hierarchy(section, Path(cache_dir)),
        device=device,
    )
    channel_names = [name for _, name in sorted(schema.channel_id2name().items())]
    _check_layout(bundle, registry, channel_names)
    cfg.training.loss.binary_concept_names = binary_concept_names(
        cast("dict[str, Any]", registry.concept_value2id), channel_names
    )

    source_mask = None
    if apply_source_mask:
        source_mask = load_run_concept_source_mask(cfg.training, schema)
        if source_mask is None:
            logger.info("[%s] the bundle config has no concept_source_mask; targets unmasked", name)
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

    meta = build_meta_model(cfg, registry, device, model=bundle.model)
    meta.model.eval()

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
            worker_init_fn=make_worker_seeder(0) if num_workers > 0 else None,
            generator=torch.Generator().manual_seed(seed),
        )
        acc = _Accumulator()
        with torch.inference_mode():
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
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    outputs = meta.model(images)
                loss, _, _, components = apply_concept_bottleneck_loss(
                    outputs,
                    cast("torch.nn.Module", meta.loss),
                    None,
                    target_labels,
                    target_concepts,
                )
                acc.add(components_to_floats({TOTAL_KEY: loss, **components}))
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
            "bundle": str(bundle.path),
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
