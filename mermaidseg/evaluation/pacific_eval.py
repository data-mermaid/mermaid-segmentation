"""Pacific Labeled Corals validation evaluation (per region + aggregate).

Units are the Pacific *sites* (regions): ``heron_reef``, ``line_islands``,
``nanwan_bay``. The dataset default annotator column (``host`` with ``archived``
fallback) is used.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from mermaidseg.datasets.pacific_labeled_corals.pacific_labeled_corals_dataset import (
    PacificLabeledCoralsDataset,
)
from mermaidseg.evaluation.gt_mapping import build_class_lookup, build_concept_lookup
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from mermaidseg.evaluation.point_datasets import run_point_evaluation
from mermaidseg.inference.predictor import CBMPredictor

logger = logging.getLogger(__name__)

SOURCE_NAME = "pacific_labeled_corals"


def evaluate_pacific(
    *,
    predictor: CBMPredictor,
    whitelist_subsets: list[str] | None,
    whitelist_sites: list[str] | None = None,
    id2label: dict[int, str],
    concept_names: list[str],
    taxonomy_csv: str | Path,
    hierarchy: dict[str, str | None] | None,
    output_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 8,
    max_images_per_unit: int | None = None,
    removed_concepts: Sequence[str] = (),
) -> dict:
    """Evaluate the model on Pacific Labeled Corals, per site and pooled.

    Args:
        predictor: Inference wrapper around the model.
        whitelist_subsets: Dataset subsets to evaluate, or ``None`` for the default.
        whitelist_sites: Sites to evaluate, or ``None`` for all.
        id2label: Target class id to label name.
        concept_names: Model concept channel names.
        taxonomy_csv: Taxonomy CSV the checkpoint was trained with.
        hierarchy: Benthic hierarchy for label roll-up, or ``None`` to disable it.
        output_dir: Directory receiving the per-unit and summary outputs.
        batch_size: Images per forward batch.
        num_workers: DataLoader workers.
        max_images_per_unit: Cap on images per site.
        removed_concepts: Concepts removed from the taxonomy during training.

    Returns:
        The evaluation result dict.
    """
    logger.info(
        "Building Pacific Labeled Corals dataset (subsets=%s sites=%s) ...",
        whitelist_subsets,
        whitelist_sites,
    )
    base = PacificLabeledCoralsDataset(
        whitelist_subsets=whitelist_subsets,
        whitelist_sites=whitelist_sites,
        transform=None,
        split="val",
    )
    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    logger.info(
        "Pacific: %d images, %d annotations, %d distinct labels",
        len(base.df_images),
        len(base.df_annotations),
        len(names),
    )

    class_lookup = build_class_lookup(SOURCE_NAME, names, id2label, hierarchy)
    concept_lookup = build_concept_lookup(SOURCE_NAME, names, concept_names, taxonomy_csv)

    return run_point_evaluation(
        eval_name="pacific_labeled_corals",
        base_ds=base,
        unit_col="site",
        key_cols=["site", "subset", "image_id"],
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
