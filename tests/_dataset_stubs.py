"""Minimal in-memory stubs for dataset-statistics tests.

Mirrors the post-PR-#96 architecture:
- ``StubDataset`` matches the BaseCoralDataset shape (df_annotations with
  source_label_name, df_images, source_id2name, source_name2id, global_offset,
  SOURCE_NAME).
- ``StubRegistry`` matches the SourceLabelRegistry surface used by the stats
  helpers: target_id2label, num_target_classes, dataset_offsets, and a
  source_to_target tensor.
- ``ConcatStub`` mirrors CombinedCoralDataset.__init__'s ``_datasets`` attr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


@dataclass
class StubDataset:
    df_annotations: pd.DataFrame
    df_images: pd.DataFrame
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    global_offset: int
    SOURCE_NAME: str = "stub"

    def __len__(self) -> int:
        return len(self.df_images)


@dataclass
class ConcatStub:
    """Stands in for CombinedCoralDataset."""

    _datasets: list[Any] = field(default_factory=list)


@dataclass
class StubRegistry:
    target_id2label: dict[int, str]
    dataset_offsets: dict[str, int]
    source_to_target: torch.Tensor  # LongTensor of shape (num_global_source + 1,)

    @property
    def num_target_classes(self) -> int:
        return len(self.target_id2label) + 1  # +1 for background


def make_mermaid_stub(
    *,
    image_to_classes: dict[str, list[str]],
    image_to_region: dict[str, str],
    source_id2name: dict[int, str],
    global_offset: int = 0,
) -> StubDataset:
    rows: list[dict] = []
    for image_id, names in image_to_classes.items():
        region = image_to_region[image_id]
        for name in names:
            rows.append(
                {
                    "image_id": image_id,
                    "source_label_name": name,
                    "region_id": region,
                    "region_name": region,
                }
            )
    df_annotations = pd.DataFrame(rows)
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates(subset=["image_id"])
        .reset_index(drop=True)
    )
    source_name2id = {v: k for k, v in source_id2name.items()}
    return StubDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        source_id2name=source_id2name,
        source_name2id=source_name2id,
        global_offset=global_offset,
        SOURCE_NAME="mermaid",
    )


def make_coralnet_stub(
    *,
    image_to_classes: dict[str, list[str]],
    image_to_source: dict[str, int],
    source_id2name: dict[int, str],
    global_offset: int = 0,
) -> StubDataset:
    rows: list[dict] = []
    for image_id, names in image_to_classes.items():
        src = image_to_source[image_id]
        for name in names:
            rows.append(
                {
                    "image_id": image_id,
                    "source_label_name": name,
                    "source_id": src,
                }
            )
    df_annotations = pd.DataFrame(rows)
    df_images = (
        df_annotations[["source_id", "image_id"]]
        .drop_duplicates(subset=["image_id"])
        .reset_index(drop=True)
    )
    source_name2id = {v: k for k, v in source_id2name.items()}
    return StubDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        source_id2name=source_id2name,
        source_name2id=source_name2id,
        global_offset=global_offset,
        SOURCE_NAME="coralnet",
    )


def make_registry_stub(
    *,
    target_id2label: dict[int, str],
    source_to_target_pairs: list[tuple[int, int]],
    dataset_offsets: dict[str, int] | None = None,
    num_global_source: int | None = None,
) -> StubRegistry:
    """Build a StubRegistry.

    ``source_to_target_pairs`` is a list of ``(global_source_id, target_id)``. Missing entries
    default to 0 (background).
    """
    if num_global_source is None:
        num_global_source = max((g for g, _ in source_to_target_pairs), default=0) + 1
    arr = np.zeros(num_global_source + 1, dtype=np.int64)
    for global_id, target_id in source_to_target_pairs:
        arr[global_id] = target_id
    return StubRegistry(
        target_id2label=target_id2label,
        dataset_offsets=dataset_offsets or {},
        source_to_target=torch.from_numpy(arr).long(),
    )


def random_split_indices(stub: StubDataset, splits: dict[str, list[int]]) -> dict[str, Subset]:
    return {name: Subset(stub, indices) for name, indices in splits.items()}
