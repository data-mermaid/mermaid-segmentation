"""Pacific Labeled Corals validation evaluation (per region + aggregate).

Units are the Pacific *sites* (regions), the image groups of the version: ``heron_reef``,
``line_islands``, ``nanwan_bay``. The images, and the annotators in priority order, are those of
the data config's ``pacific_labeled_corals`` val block.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from mermaidseg.data.config import DataSection, entry_split_groups
from mermaidseg.data.factory import open_dataset_version
from mermaidseg.data.point_dataset import PointDataset
from mermaidseg.evaluation.gt_mapping import build_class_lookup, build_concept_lookup
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from mermaidseg.evaluation.point_datasets import reader_annotations, run_point_evaluation
from mermaidseg.inference.predictor import CBMPredictor

logger = logging.getLogger(__name__)

SOURCE_NAME = "pacific_labeled_corals"
UNIT_COL = "group"


def pacific_reader(section: DataSection, cache_dir: Path, repo_root: Path) -> PointDataset:
    """Build the Pacific reader of the section's val block.

    Raises:
        KeyError: The section has no ``pacific_labeled_corals`` dataset.
        ValueError: The entry has no val block, or the reader refuses its selection.
    """
    entry = section.datasets[SOURCE_NAME]
    groups = entry_split_groups(entry, "val", repo_root)
    assert entry.val is not None  # entry_split_groups raised otherwise
    return PointDataset(
        open_dataset_version(section, SOURCE_NAME, cache_dir),
        split="val",
        groups=groups,
        on_missing_images=entry.on_missing_images,
        annotators=entry.val.annotators,
    )


def evaluate_pacific(
    *,
    predictor: CBMPredictor,
    reader: PointDataset,
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
        reader: Pacific reader (see :func:`pacific_reader`).
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
    annotations = reader_annotations(reader)
    names = annotations["source_label_name"].unique().tolist()
    logger.info(
        "Pacific: %d images, %d annotations, %d distinct labels",
        len(reader.images),
        len(annotations),
        len(names),
    )

    class_lookup = build_class_lookup(reader.label_targets, names, id2label, hierarchy)
    concept_lookup = build_concept_lookup(SOURCE_NAME, names, concept_names, taxonomy_csv)

    return run_point_evaluation(
        eval_name="pacific_labeled_corals",
        reader=reader,
        unit_col=UNIT_COL,
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
