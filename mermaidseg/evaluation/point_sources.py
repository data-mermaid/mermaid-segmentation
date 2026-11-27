"""Point-evaluation sources: the CoralNet and Pacific readers, split-file groups and sampling.

CoralNet scores the groups (``s<source id>``) of one split of the split file, read from the
``coralnet`` version the data config pins, in the synced cache. Pacific Labeled Corals scores
its *sites* (``heron_reef``, ``line_islands``, ``nanwan_bay``), the image groups of the version,
with the images and the annotator priority of the data config's val block.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from mermaidseg.data.config import DataSection, Split, entry_split_groups
from mermaidseg.data.factory import open_dataset_version
from mermaidseg.data.point_dataset import PointDataset
from mermaidseg.dataset_reconciliation.artifact_meta import source_id_of_group
from mermaidseg.evaluation.point_datasets import UNIT_COL
from segdata_spec.splits import read_split_file

logger = logging.getLogger(__name__)

CORALNET_SOURCE = "coralnet"
PACIFIC_SOURCE = "pacific_labeled_corals"


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
        open_dataset_version(section, CORALNET_SOURCE, cache_dir),
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
            "dataset": CORALNET_SOURCE,
            "source_id": sample[UNIT_COL].map(source_id_of_group),
            "image_id": sample["image_id"].astype(str),
        }
    ).to_parquet(path, index=False)
    return sample["image_id"].astype(str).tolist()


def pacific_reader(section: DataSection, cache_dir: Path, repo_root: Path) -> PointDataset:
    """Build the Pacific reader of the section's val block.

    Raises:
        KeyError: The section has no ``pacific_labeled_corals`` dataset.
        ValueError: The entry has no val block, or the reader refuses its selection.
    """
    entry = section.datasets[PACIFIC_SOURCE]
    groups = entry_split_groups(entry, "val", repo_root)
    assert entry.val is not None  # entry_split_groups raised otherwise
    return PointDataset(
        open_dataset_version(section, PACIFIC_SOURCE, cache_dir),
        split="val",
        global_offset=0,
        groups=groups,
        annotators=entry.val.annotators,
    )
