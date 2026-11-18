"""CoralNet evaluation on the train, val or test sources (per source id + aggregate)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset
from mermaidseg.evaluation.gt_mapping import build_class_lookup, build_concept_lookup
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from mermaidseg.evaluation.point_datasets import run_point_evaluation
from mermaidseg.inference.predictor import CBMPredictor

logger = logging.getLogger(__name__)

SOURCE_NAME = "coralnet"


def sample_images_per_source(df_images: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """At most ``n`` images per ``source_id``, drawn with ``seed``; small sources keep all."""
    if n < 1:
        raise ValueError(f"images per source must be positive, got {n}")
    parts = [
        group.sample(n=min(n, len(group)), random_state=seed)
        for _, group in df_images.groupby("source_id", sort=True)
    ]
    if not parts:
        return df_images.iloc[:0].copy()
    return pd.concat(parts).sort_values(["source_id", "image_id"]).reset_index(drop=True)


def write_coralnet_sample(
    *,
    path: str | Path,
    whitelist_sources: list[int] | None,
    image_blacklist: str | list[int | str] | None = None,
    annotation_overrides: str | None = None,
    images_per_source: int,
    seed: int = 0,
) -> Path:
    """Write an ``image_keep`` parquet of a seeded per-source image sample."""
    base = CoralNetDataset(
        whitelist_sources=cast("list[int | str] | None", whitelist_sources),
        image_blacklist=image_blacklist,
        annotation_overrides=annotation_overrides,
        transform=None,
    )
    sample = sample_images_per_source(base.df_images, images_per_source, seed)
    counts = sample["source_id"].value_counts()
    logger.info(
        "CoralNet sample: %d of %d images over %d sources (<=%d per source, seed %d); "
        "%d sources have fewer",
        len(sample),
        len(base.df_images),
        len(counts),
        images_per_source,
        seed,
        int((counts < images_per_source).sum()),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": SOURCE_NAME,
            "source_id": sample["source_id"].astype(str),
            "image_id": sample["image_id"].astype(str),
        }
    ).to_parquet(path, index=False)
    return path


def evaluate_coralnet(
    *,
    predictor: CBMPredictor,
    whitelist_sources: list[int] | None,
    image_blacklist: str | list[int | str] | None = None,
    annotation_overrides: str | None = None,
    image_keep: str | Path | None = None,
    id2label: dict[int, str],
    concept_names: list[str],
    taxonomy_csv: str | Path,
    hierarchy: dict[str, str | None] | None,
    output_dir: str | Path,
    split: str = "val",
    batch_size: int = 8,
    num_workers: int = 8,
    max_images_per_unit: int | None = None,
    removed_concepts: Sequence[str] = (),
) -> dict:
    """Evaluate the model on CoralNet sources, per source id and pooled.

    Args:
        predictor: Inference wrapper around the model.
        whitelist_sources: CoralNet source ids to evaluate, or ``None`` for the split default.
        image_blacklist: Images to exclude.
        annotation_overrides: Path to annotation overrides.
        image_keep: Parquet of images to keep.
        id2label: Target class id to label name.
        concept_names: Model concept channel names.
        taxonomy_csv: Taxonomy CSV the checkpoint was trained with.
        hierarchy: Benthic hierarchy for label roll-up, or ``None`` to disable it.
        output_dir: Directory receiving the per-unit and summary outputs.
        split: Dataset split to evaluate.
        batch_size: Images per forward batch.
        num_workers: DataLoader workers.
        max_images_per_unit: Cap on images per source.
        removed_concepts: Concepts removed from the taxonomy during training.

    Returns:
        The evaluation result dict, with ``whitelist_sources`` (and ``image_keep``) added.
    """
    logger.info(
        "Building CoralNet %s dataset (whitelist_sources=%s, "
        "image_blacklist=%s, annotation_overrides=%s, image_keep=%s) ...",
        split,
        whitelist_sources,
        image_blacklist,
        annotation_overrides,
        image_keep,
    )
    base = CoralNetDataset(
        whitelist_sources=cast("list[int | str] | None", whitelist_sources),
        image_blacklist=image_blacklist,
        annotation_overrides=annotation_overrides,
        image_keep=image_keep,
        transform=None,
        split=split,
    )
    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    logger.info(
        "CoralNet %s: %d images, %d annotations, %d distinct labels",
        split,
        len(base.df_images),
        len(base.df_annotations),
        len(names),
    )

    class_lookup = build_class_lookup(SOURCE_NAME, names, id2label, hierarchy)
    concept_lookup = build_concept_lookup(SOURCE_NAME, names, concept_names, taxonomy_csv)

    result = run_point_evaluation(
        eval_name=f"coralnet_{split}",
        base_ds=base,
        unit_col="source_id",
        key_cols=["image_id"],
        predictor=predictor,
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
        concept_names=concept_names,
        id2label=id2label,
        output_dir=output_dir,
        binary_concepts=REPORTED_BINARY_CONCEPTS,
        taxonomy_csv=taxonomy_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images_per_unit=max_images_per_unit,
        removed_concepts=removed_concepts,
    )
    result["whitelist_sources"] = sorted(whitelist_sources or [])
    if image_keep is not None:
        result["image_keep"] = str(image_keep)
    return result
