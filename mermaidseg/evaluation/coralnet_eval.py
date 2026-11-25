"""CoralNet evaluation on the train, val or test sources (per source group + aggregate).

The scored groups (``s<source id>``) are the groups of the requested split in the CoralNet split
file, read from the ``coralnet`` version the data config pins, in the synced cache.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path

import pandas as pd

from mermaidseg.data.config import DataSection, Split
from mermaidseg.data.factory import open_dataset_version
from mermaidseg.data.point_dataset import PointDataset
from mermaidseg.evaluation.gt_mapping import build_class_lookup, build_concept_lookup
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from mermaidseg.evaluation.point_datasets import reader_annotations, run_point_evaluation
from mermaidseg.inference.predictor import CBMPredictor
from segdata_spec.splits import read_split_file

logger = logging.getLogger(__name__)

SOURCE_NAME = "coralnet"
UNIT_COL = "group"


def split_groups(split_file: str | Path, split: str) -> list[str]:
    """Return the sorted groups of ``split_file`` whose split is ``split``.

    Raises:
        ValueError: The split file is malformed, or no group has the split.
    """
    table = read_split_file(Path(split_file)).to_pandas()
    groups = sorted(table.loc[table["split"] == split, "group"].astype(str))
    if not groups:
        raise ValueError(f"{split_file} has no group in split {split!r}")
    return groups


def coralnet_reader(
    section: DataSection,
    cache_dir: Path,
    *,
    groups: Collection[str],
    split: Split,
    image_ids: Collection[str] | None = None,
) -> PointDataset:
    """Build the CoralNet reader over ``groups``, keeping only ``image_ids`` when given.

    Raises:
        ValueError: A group is unknown or not synced, or an image id is outside the selection.
    """
    return PointDataset(
        open_dataset_version(section, SOURCE_NAME, cache_dir),
        split=split,
        global_offset=0,
        groups=groups,
        image_ids=image_ids,
    )


def sample_images_per_source(images: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """At most ``n`` images per ``group`` (source), drawn with ``seed``; small sources keep all."""
    if n < 1:
        raise ValueError(f"images per source must be positive, got {n}")
    parts = [
        group.sample(n=min(n, len(group)), random_state=seed)
        for _, group in images.groupby(UNIT_COL, sort=True)
    ]
    if not parts:
        return images.iloc[:0].copy()
    return pd.concat(parts).sort_values([UNIT_COL, "image_id"]).reset_index(drop=True)


def write_coralnet_sample(
    *,
    path: str | Path,
    reader: PointDataset,
    images_per_source: int,
    seed: int = 0,
) -> list[str]:
    """Draw a seeded per-source image sample of ``reader``, write it and return its image ids.

    The parquet has the columns ``dataset``, ``source_id`` (the group without its ``s``) and
    ``image_id``.
    """
    sample = sample_images_per_source(reader.images, images_per_source, seed)
    counts = sample[UNIT_COL].value_counts()
    logger.info(
        "CoralNet sample: %d of %d images over %d sources (<=%d per source, seed %d); "
        "%d sources have fewer",
        len(sample),
        len(reader.images),
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
            "source_id": sample[UNIT_COL].astype(str).str.removeprefix("s"),
            "image_id": sample["image_id"].astype(str),
        }
    ).to_parquet(path, index=False)
    return sample["image_id"].astype(str).tolist()


def evaluate_coralnet(
    *,
    predictor: CBMPredictor,
    reader: PointDataset,
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
    """Evaluate the model on CoralNet sources, per source group and pooled.

    Args:
        predictor: Inference wrapper around the model.
        reader: CoralNet reader over the scored groups (see :func:`coralnet_reader`).
        id2label: Target class id to label name.
        concept_names: Model concept channel names.
        taxonomy_csv: Taxonomy CSV the checkpoint was trained with.
        hierarchy: Benthic hierarchy for label roll-up, or ``None`` to disable it.
        output_dir: Directory receiving the per-unit and summary outputs.
        split: Split name used in the output keys.
        batch_size: Images per forward batch.
        num_workers: DataLoader workers.
        max_images_per_unit: Cap on images per source.
        removed_concepts: Concepts removed from the taxonomy during training.

    Returns:
        The evaluation result dict, with the scored ``groups`` added.
    """
    annotations = reader_annotations(reader)
    names = annotations["source_label_name"].unique().tolist()
    logger.info(
        "CoralNet %s: %d images, %d annotations, %d distinct labels",
        split,
        len(reader.images),
        len(annotations),
        len(names),
    )

    class_lookup = build_class_lookup(reader.label_targets, names, id2label, hierarchy)
    concept_lookup = build_concept_lookup(SOURCE_NAME, names, concept_names, taxonomy_csv)

    result = run_point_evaluation(
        eval_name=f"coralnet_{split}",
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
    result["groups"] = sorted(reader.images[UNIT_COL].astype(str).unique())
    return result
