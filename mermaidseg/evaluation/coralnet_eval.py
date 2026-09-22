"""CoralNet validation evaluation (per source id + aggregate)."""

from __future__ import annotations

import logging
from pathlib import Path

from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset
from mermaidseg.evaluation.gt_mapping import build_class_lookup, build_concept_lookup
from mermaidseg.evaluation.point_datasets import run_point_evaluation
from mermaidseg.evaluation.predictor import CBMPredictor

logger = logging.getLogger(__name__)

SOURCE_NAME = "coralnet"


def evaluate_coralnet(
    *,
    predictor: CBMPredictor,
    whitelist_sources: list[int] | None,
    id2label: dict[int, str],
    concept_names: list[str],
    taxonomy_csv: str | Path,
    hierarchy: dict[str, str | None] | None,
    output_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 8,
    max_images_per_unit: int | None = None,
) -> dict:
    logger.info("Building CoralNet dataset (whitelist_sources=%s) ...", whitelist_sources)
    base = CoralNetDataset(
        whitelist_sources=whitelist_sources,
        transform=None,
        split="val",
    )
    names = base.df_annotations["source_label_name"].astype(str).str.lower().unique().tolist()
    logger.info("CoralNet: %d images, %d annotations, %d distinct labels",
                len(base.df_images), len(base.df_annotations), len(names))

    class_lookup = build_class_lookup(SOURCE_NAME, names, id2label, hierarchy)
    concept_lookup = build_concept_lookup(SOURCE_NAME, names, concept_names, taxonomy_csv)

    return run_point_evaluation(
        eval_name="coralnet",
        base_ds=base,
        unit_col="source_id",
        key_cols=["image_id"],
        predictor=predictor,
        class_lookup=class_lookup,
        concept_lookup=concept_lookup,
        concept_names=concept_names,
        id2label=id2label,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images_per_unit=max_images_per_unit,
    )
