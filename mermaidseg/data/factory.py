"""Build the readers a data config selects, one per dataset and split."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from mermaidseg.data.base import VersionDataset
from mermaidseg.data.config import (
    DATASET_BUCKET,
    TAXONOMY_NAME,
    DataSection,
    Split,
    entry_split_groups,
)
from mermaidseg.data.dense_dataset import DenseDataset
from mermaidseg.data.point_dataset import PointDataset
from mermaidseg.io import build_eval_transform, compile_transform
from segdata_spec.manifest import Pin
from segdata_spec.read import DatasetVersion, open_taxonomy, open_version
from segdata_spec.taxonomy import hierarchy_from_table


def _transform_for(
    split: Split, spec: Mapping[str, Any] | None, input_size: Sequence[int]
) -> Callable[..., dict[str, Any]]:
    """Return the compiled train transform, or the eval resize for val and test."""
    if split == "train":
        return compile_transform(spec or {})
    return build_eval_transform(input_size)


def _reader(
    version: DatasetVersion,
    split: Split,
    shared: dict[str, Any],
    padding: int,
    annotators: list[str] | None,
) -> VersionDataset:
    if version.manifest.kind != "points" and annotators is not None:
        raise ValueError(
            f"dataset {version.manifest.dataset!r}: annotators apply to point datasets only"
        )
    if version.manifest.kind == "points":
        return PointDataset(version, split=split, annotators=annotators, padding=padding, **shared)
    return DenseDataset(version, split=split, **shared)


def open_dataset_version(section: DataSection, name: str, cache_dir: Path) -> DatasetVersion:
    """Open the synced version the section pins for dataset ``name``, with the entry's ``resize``.

    Raises:
        KeyError: The section has no dataset ``name``.
        segdata_spec.errors.NotSynced: The version was not synced with the entry's ``resize``.
    """
    entry = section.datasets[name]
    pin = Pin(name=name, version=entry.version, manifest_sha256=entry.manifest_sha256)
    return open_version(cache_dir, DATASET_BUCKET, pin, resize=entry.resize)


def build_datasets(
    section: DataSection,
    cache_dir: Path,
    *,
    repo_root: Path,
    padding: int,
    input_size: Sequence[int],
    image_ids: Mapping[tuple[str, str], Collection[str]] | None = None,
) -> dict[tuple[str, str], VersionDataset]:
    """Open each dataset version and build a reader per defined split block.

    Args:
        section: The validated ``data`` section.
        cache_dir: The directory ``scripts/sync_data.py`` synced into.
        repo_root: The directory ``split_file`` paths are relative to.
        padding: Half-size of the square a point paints; used by point versions only.
        input_size: ``(height, width)`` the val and test images resize to.
        image_ids: Per ``(dataset, split)``, the images to keep (an evaluation subset); a key
            absent from the mapping keeps every selected image.

    Returns:
        The readers keyed by ``(dataset name, split)``. A block's ``repeat`` is not applied.

    Raises:
        segdata_spec.errors.NotSynced: A version was not synced with the entry's ``resize``.
        ValueError: A reader refuses its selection (see :class:`VersionDataset`), or a dense
            dataset's block sets ``annotators``.
    """
    subsets = image_ids or {}
    datasets: dict[tuple[str, str], VersionDataset] = {}
    for name, entry in section.datasets.items():
        version = open_dataset_version(section, name, cache_dir)
        for split in get_args(Split):
            block = getattr(entry, split)
            if block is None:
                continue
            shared: dict[str, Any] = {
                "groups": entry_split_groups(entry, split, repo_root),
                "transform": _transform_for(split, block.transform, input_size),
                "on_missing_images": entry.on_missing_images,
                "image_ids": subsets.get((name, split)),
            }
            datasets[(name, split)] = _reader(version, split, shared, padding, block.annotators)
    return datasets


def load_benthic_hierarchy(section: DataSection, cache_dir: Path) -> dict[str, str | None] | None:
    """Return the benthic-attribute hierarchy of the section's pinned taxonomy, or None unpinned.

    Args:
        section: The validated ``data`` section.
        cache_dir: The directory ``scripts/sync_data.py`` synced into.

    Returns:
        Lower-cased name -> parent name (see ``segdata_spec.taxonomy.hierarchy_from_table``).

    Raises:
        FileNotFoundError: The pinned taxonomy is not in the cache.
        segdata_spec.errors.PinMismatch: The cached taxonomy differs from the pin.
    """
    if section.taxonomy is None:
        return None
    pin = Pin(name=TAXONOMY_NAME, **section.taxonomy.model_dump())
    return hierarchy_from_table(open_taxonomy(cache_dir, DATASET_BUCKET, pin).benthic_attributes)
