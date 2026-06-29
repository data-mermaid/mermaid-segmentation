"""Visualize the first N samples from the CBM train and val DataLoaders.

Mirrors the data setup in Concept_Bottleneck_Pipeline_vitL_lora_dpt.py and saves
per-sample figures (RGB + target mask + source mask) under ./tmp/train/ and
./tmp/val/.

Run from the nbs/ directory::

    uv run python visualize_dataloader_samples.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nb_setup import check_aws_session, check_env, check_mlflow_version
from torch.utils.data import ConcatDataset, DataLoader

from mermaidseg.dataset_reconciliation import (
    ConceptSchema,
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.datasets import (
    BenthosYuvalCoralsDataset,
    CatlinSeaviewDataset,
    CoralNetDataset,
    CoralscapesV2Dataset,
    MermaidDataset,
    MooreaLabeledCoralsDataset,
    PacificLabeledCoralsDataset,
    UCSDMosaicsDataset,
    make_worker_init_fn,
    setup_local_cache,
)
from mermaidseg.io import get_parser, setup_config, update_config_with_args

VITL_ENCODER_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SEED = 0
NUM_SAMPLES = 100
OUTPUT_DIR = Path("./tmp")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Single-process loading avoids pymp temp-dir cleanup errors on shared NFS scratch.
NUM_WORKERS = 0


def denormalize_image(image_chw: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to an RGB float array in [0, 1]."""
    rgb = image_chw.numpy().transpose(1, 2, 0)
    rgb = rgb * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(rgb, 0.0, 1.0)


def source_id_to_name(registry: SourceLabelRegistry, class_id: int) -> str:
    if class_id == 0:
        return "background"
    source_name, label_name = registry.global_id2source[class_id]
    return f"{source_name}/{label_name}"


def target_id_to_name(registry: SourceLabelRegistry, class_id: int) -> str:
    if class_id == 0:
        return "ignore"
    return registry.target_id2label[class_id]


def present_class_ids(mask: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(mask))


def mask_colormap(class_ids: list[int]) -> tuple[ListedColormap, dict[int, int]]:
    """Map present class IDs to consecutive colormap indices."""
    base = plt.get_cmap("tab20")
    id_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    colors = [base(idx % base.N) for idx in range(len(class_ids))]
    return ListedColormap(colors), id_to_index


def indexed_mask(mask: np.ndarray, id_to_index: dict[int, int]) -> np.ndarray:
    indexed = np.zeros_like(mask, dtype=np.int32)
    for class_id, idx in id_to_index.items():
        indexed[mask == class_id] = idx
    return indexed


def legend_patches(
    class_ids: list[int],
    id_to_index: dict[int, int],
    name_fn,
) -> list[Patch]:
    base = plt.get_cmap("tab20")
    patches: list[Patch] = []
    for class_id in class_ids:
        color = base(id_to_index[class_id] % base.N)
        patches.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                label=f"{class_id}: {name_fn(class_id)}",
            )
        )
    return patches


def save_sample_figure(
    sample_idx: int,
    image: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    registry: SourceLabelRegistry,
    output_dir: Path,
    rgb_title: str = "RGB (after augmentation)",
) -> None:
    rgb = denormalize_image(image)
    source_np = source_mask.numpy()
    target_np = target_mask.numpy()

    source_ids = present_class_ids(source_np)
    target_ids = present_class_ids(target_np)
    source_cmap, source_index = mask_colormap(source_ids)
    target_cmap, target_index = mask_colormap(target_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb)
    axes[0].set_title(rgb_title)
    axes[0].axis("off")

    axes[1].imshow(indexed_mask(target_np, target_index), cmap=target_cmap, vmin=0, vmax=max(len(target_ids) - 1, 0))
    axes[1].set_title("Target label class")
    axes[1].axis("off")
    axes[1].legend(
        handles=legend_patches(target_ids, target_index, lambda cid: target_id_to_name(registry, cid)),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    axes[2].imshow(indexed_mask(source_np, source_index), cmap=source_cmap, vmin=0, vmax=max(len(source_ids) - 1, 0))
    axes[2].set_title("Source label class")
    axes[2].axis("off")
    axes[2].legend(
        handles=legend_patches(source_ids, source_index, lambda cid: source_id_to_name(registry, cid)),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout()
    out_path = output_dir / f"sample_{sample_idx:03d}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def collect_samples(
    loader: DataLoader,
    registry: SourceLabelRegistry,
    num_samples: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Draw the first ``num_samples`` items from ``loader``."""
    lookup = registry.source_to_target.cpu()
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for images, source_masks in loader:
        target_masks = lookup[source_masks.long()]
        batch_size = images.shape[0]
        for batch_idx in range(batch_size):
            samples.append(
                (
                    images[batch_idx].cpu(),
                    source_masks[batch_idx].cpu(),
                    target_masks[batch_idx].cpu(),
                )
            )
            if len(samples) >= num_samples:
                return samples

    return samples


def visualize_loader(
    loader: DataLoader,
    registry: SourceLabelRegistry,
    output_dir: Path,
    num_samples: int,
    rgb_title: str,
) -> int:
    """Collect and save up to ``num_samples`` figures from ``loader``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(loader, registry, num_samples)
    for sample_idx, (image, source_mask, target_mask) in enumerate(samples):
        save_sample_figure(
            sample_idx,
            image,
            source_mask,
            target_mask,
            registry,
            output_dir,
            rgb_title=rgb_title,
        )
    return len(samples)


def main() -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = (
            "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-3546X3USYJNZ"
        )

    check_env()
    check_aws_session()
    check_mlflow_version()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    cfg = setup_config(
        {
            "data": "../configs/data_config_512.yaml",
            "training": "../configs/training_config_cbm_512.yaml",
            "model": "../configs/model_config_cbm_dpt_lora_vitl.yaml",
            "logger": "../configs/logger_config.yaml",
        }
    )
    args = get_parser().parse_args(["--run-name=mermaid_base_run_dinov3_lora_dpt_512"])
    cfg = update_config_with_args(cfg, args)
    cfg.model.encoder_name = VITL_ENCODER_NAME
    cfg.training.iterations_per_train_epoch = 4000
    cfg.training.iterations_per_val_epoch = 400
    cfg.training.batch_size = 25

    cache_stats = setup_local_cache(cfg.data)

    DATASET_CLASSES = {
        "pacific_labeled_corals": PacificLabeledCoralsDataset,
        "moorea_labeled_corals": MooreaLabeledCoralsDataset,
        "catlin_seaview": CatlinSeaviewDataset,
        "mermaid": MermaidDataset,
        "coralnet": CoralNetDataset,
        "coralscapes_v2": CoralscapesV2Dataset,
        "benthos_yuval": BenthosYuvalCoralsDataset,
        "ucsd_mosaics": UCSDMosaicsDataset,
    }

    def _build(name, split_cfg):
        cls = DATASET_CLASSES[name]
        if name in ("coralscapes_v2", "ucsd_mosaics"):
            return cls(**split_cfg)
        return cls(**split_cfg, padding=cfg.training.padding)

    dataset_dict: dict[tuple[str, str], object] = {}
    for name in DATASET_CLASSES:
        for split, split_cfg in cfg.data[name].items():
            if split_cfg is None or split_cfg == "None":
                continue
            dataset_dict[(name, split)] = _build(name, split_cfg)
            print(f"{name:>24s} - {split:<5s}: {len(dataset_dict[(name, split)]):>7d} samples")

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = make_worker_init_fn(cache_stats)

    concept_mapping_path = (
        cfg.training.get("concept_mapping_path")
        or "/data/vision/beery/scratch/sauder/mermaid-segmentation/configs/class_to_concepts.csv"
    )

    _, registry_datasets = prepare_splits_for_registry(dataset_dict)
    run_sources = {ds.SOURCE_NAME for ds in registry_datasets}
    schema = ConceptSchema.from_csv(concept_mapping_path, sources=run_sources)

    registry = SourceLabelRegistry(
        registry_datasets,
        target_label_subset=cfg.training.class_subset,
        compute_concepts=cfg.training.training_mode != "standard",
        concept_mapping_path=concept_mapping_path,
        concept_schema=schema,
        label_roll_up=cfg.training.get("label_roll_up", False),
    ).to(device)

    attach_registry(registry, dataset_dict.values())

    train_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "train"] + [
        ds for (name, split), ds in dataset_dict.items() if name == "coralscapes_v2" and split == "train"
    ] * 5
    val_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "val"]

    train_loader = DataLoader(ConcatDataset(train_datasets), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(ConcatDataset(val_datasets), shuffle=True, **loader_kwargs)

    print(f"train batches: {len(train_loader)}   val batches: {len(val_loader)}")

    train_count = visualize_loader(
        train_loader,
        registry,
        OUTPUT_DIR / "train",
        NUM_SAMPLES,
        rgb_title="RGB (after augmentation)",
    )
    val_count = visualize_loader(
        val_loader,
        registry,
        OUTPUT_DIR / "val",
        NUM_SAMPLES,
        rgb_title="RGB (validation transform)",
    )

    print(
        f"Saved {train_count} train figures to {(OUTPUT_DIR / 'train').resolve()} "
        f"and {val_count} val figures to {(OUTPUT_DIR / 'val').resolve()}"
    )


if __name__ == "__main__":
    main()
