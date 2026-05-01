"""Minimal in-memory dataset stubs for logger statistics tests.

Avoids spinning up MermaidDataset / CoralNetDataset (which require S3 + the
MERMAID API). Each stub exposes only the attributes that ``_resolve_annotations``
and the ``_compute_*`` helpers read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from torch.utils.data import Subset


@dataclass
class StubDataset:
    """Stand-in for ``MermaidDataset`` / ``CoralNetDataset``.

    Only carries the four attributes the logger reads.
    """

    df_annotations: pd.DataFrame
    df_images: pd.DataFrame
    id2label: dict[int, str]
    num_classes: int

    def __len__(self) -> int:
        return len(self.df_images)


@dataclass
class ConcatStub:
    """Stand-in for combined / ``ConcatDataset``-style wrappers.

    Mirrors the duck-typing the existing ``log_datasets`` already handles:
    expose either ``_datasets`` or ``datasets``.
    """

    _datasets: list = field(default_factory=list)


def make_mermaid_stub(
    *,
    image_to_classes: dict[str, list[str]],
    image_to_region: dict[str, str],
    class_subset: list[str],
) -> StubDataset:
    """Build a Mermaid-shaped stub from a per-image class list.

    ``image_to_classes`` maps ``image_id -> [benthic_attribute_name, ...]``
    (one row per annotation). ``image_to_region`` maps ``image_id -> region_name``.
    """
    rows: list[dict] = []
    for image_id, classes in image_to_classes.items():
        region = image_to_region[image_id]
        for cls in classes:
            rows.append(
                {
                    "image_id": image_id,
                    "benthic_attribute_name": cls,
                    "region_id": region,
                    "region_name": region,
                }
            )
    df_annotations = pd.DataFrame(rows)
    df_images = df_annotations[["image_id", "region_id", "region_name"]].drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    id2label = dict(enumerate(class_subset, start=1))
    return StubDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        id2label=id2label,
        num_classes=len(class_subset) + 1,
    )


def make_coralnet_stub(
    *,
    image_to_classes: dict[str, list[str]],
    image_to_source: dict[str, int],
    class_subset: list[str],
) -> StubDataset:
    """CoralNet-shaped stub: ``source_id`` instead of ``region_*``."""
    rows: list[dict] = []
    for image_id, classes in image_to_classes.items():
        source = image_to_source[image_id]
        for cls in classes:
            rows.append(
                {
                    "image_id": image_id,
                    "benthic_attribute_name": cls,
                    "source_id": source,
                }
            )
    df_annotations = pd.DataFrame(rows)
    df_images = df_annotations[["source_id", "image_id"]].drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    id2label = dict(enumerate(class_subset, start=1))
    return StubDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        id2label=id2label,
        num_classes=len(class_subset) + 1,
    )


def random_split_indices(stub: StubDataset, splits: dict[str, list[int]]) -> dict[str, Subset]:
    """Wrap a stub in PyTorch ``Subset`` objects for testing the Subset path."""
    return {name: Subset(stub, indices) for name, indices in splits.items()}
