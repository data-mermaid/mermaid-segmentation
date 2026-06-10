"""Canonical source vocabularies and registry wiring across train/val/test splits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry


@dataclass(frozen=True)
class SourceVocabulary:
    """Shared local source-label space for one SOURCE_NAME."""

    source_name: str
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    num_source_classes: int


def _root_dataset(ds: Any) -> Any:
    """Unwrap torch.utils.data.Subset chains."""
    while hasattr(ds, "dataset") and not hasattr(ds, "SOURCE_NAME"):
        ds = ds.dataset
    return ds


def _require_source_name(ds: Any) -> str:
    root = _root_dataset(ds)
    name = getattr(root, "SOURCE_NAME", None)
    if not name:
        raise TypeError(
            f"{type(root).__name__} has no SOURCE_NAME; cannot wire into SourceLabelRegistry."
        )
    return name


def _apply_source_vocabulary(
    root: Any,
    source_id2name: dict[int, str],
    source_name2id: dict[str, int],
    num_source_classes: int,
) -> None:
    """Apply canonical source maps, rebuilding any dataset-local id lookups."""
    if hasattr(root, "set_source_vocabulary"):
        root.set_source_vocabulary(source_id2name, source_name2id, num_source_classes)
        return
    root.source_id2name = dict(source_id2name)
    root.source_name2id = dict(source_name2id)
    root.num_source_classes = int(num_source_classes)


def group_splits(
    dataset_dict: Mapping[tuple[str, str], Any],
) -> dict[str, list[tuple[str, str, Any]]]:
    """Group ``{(config_name, split): dataset}`` by ``SOURCE_NAME``."""
    grouped: dict[str, list[tuple[str, str, Any]]] = defaultdict(list)
    for (config_name, split), ds in dataset_dict.items():
        source_name = _require_source_name(ds)
        grouped[source_name].append((config_name, split, ds))
    return dict(grouped)


def build_source_vocabularies(
    dataset_dict: Mapping[tuple[str, str], Any],
    *,
    sort_names: bool = True,
) -> dict[str, SourceVocabulary]:
    """Union of label names per SOURCE_NAME across all splits."""
    vocabularies: dict[str, SourceVocabulary] = {}
    for source_name, entries in group_splits(dataset_dict).items():
        names: set[str] = set()
        for _, _, ds in entries:
            root = _root_dataset(ds)
            names.update(root.source_name2id.keys())
        ordered = sorted(names) if sort_names else list(names)
        id2name = dict(enumerate(ordered, start=1))
        name2id = {name: local_id for local_id, name in id2name.items()}
        vocabularies[source_name] = SourceVocabulary(
            source_name=source_name,
            source_id2name=id2name,
            source_name2id=name2id,
            num_source_classes=len(id2name) + 1,
        )
    return vocabularies


def apply_vocabularies(
    dataset_dict: Mapping[tuple[str, str], Any],
    vocabularies: Mapping[str, SourceVocabulary],
) -> None:
    """Overwrite each dataset's local source maps (in-place)."""
    for source_name, entries in group_splits(dataset_dict).items():
        vocab = vocabularies[source_name]
        for _, _, ds in entries:
            root = _root_dataset(ds)
            _apply_source_vocabulary(
                root,
                vocab.source_id2name,
                vocab.source_name2id,
                vocab.num_source_classes,
            )


def select_registry_representatives(
    dataset_dict: Mapping[tuple[str, str], Any],
    *,
    prefer_split: str = "train",
) -> list[Any]:
    """One dataset per SOURCE_NAME for ``SourceLabelRegistry`` (train if present, else any)."""
    representatives: list[Any] = []
    for _source_name, entries in group_splits(dataset_dict).items():
        by_split = {split: ds for _, split, ds in entries}
        if prefer_split in by_split:
            representatives.append(_root_dataset(by_split[prefer_split]))
            continue
        representatives.append(_root_dataset(entries[0][2]))
    return representatives


def attach_registry(
    registry: SourceLabelRegistry,
    datasets: Iterable[Any],
) -> None:
    """Apply registered vocab + global offset to every split instance."""
    ref_by_source = {_root_dataset(ds).SOURCE_NAME: _root_dataset(ds) for ds in registry.datasets}
    for ds in datasets:
        root = _root_dataset(ds)
        source_name = root.SOURCE_NAME
        if source_name not in ref_by_source:
            raise KeyError(
                f"SOURCE_NAME '{source_name}' is not registered. "
                f"Registered: {sorted(ref_by_source)}. "
                "Add a representative for this source to SourceLabelRegistry."
            )
        ref = ref_by_source[source_name]
        _apply_source_vocabulary(
            root,
            ref.source_id2name,
            ref.source_name2id,
            ref.num_source_classes,
        )
        root.set_global_offset(registry.dataset_offsets[source_name])


def prepare_splits_for_registry(
    dataset_dict: Mapping[tuple[str, str], Any],
    *,
    sort_names: bool = True,
    prefer_split: str = "train",
) -> tuple[dict[str, SourceVocabulary], list[Any]]:
    """Canonicalize vocabularies and return registry input datasets.

    Returns:
        vocabularies: per-SOURCE_NAME canonical maps (already applied in-place).
        registry_datasets: pass to ``SourceLabelRegistry(...)``.
    """
    vocabularies = build_source_vocabularies(dataset_dict, sort_names=sort_names)
    apply_vocabularies(dataset_dict, vocabularies)
    registry_datasets = select_registry_representatives(dataset_dict, prefer_split=prefer_split)
    return vocabularies, registry_datasets
