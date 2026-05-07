"""Wraps multiple registered source datasets into a single concatenated dataset."""

from __future__ import annotations

from typing import Any

import torch

from mermaidseg.datasets.utils import _joint_collate


def _root_dataset(ds: Any) -> Any:
    """Unwrap PyTorch ``Subset`` wrappers to reach the underlying dataset."""
    while hasattr(ds, "dataset") and not hasattr(ds, "SOURCE_NAME"):
        ds = ds.dataset
    return ds


class CombinedCoralDataset:
    """Wraps multiple source datasets into a single dataset over the joint global source-label space.

    All constituent datasets must have been registered with the same
    :class:`SourceLabelRegistry` so their emitted source-label masks share a
    consistent global integer space (and offsets are non-overlapping).

    Indexing and length are delegated to ``torch.utils.data.ConcatDataset``.
    The ``_datasets`` attribute is recognised by ``Logger.log_datasets`` for
    per-source MLflow logging.
    """

    def __init__(self, datasets: list):
        if not datasets:
            raise ValueError("datasets must be non-empty.")

        offsets = []
        names = []
        for ds in datasets:
            root = _root_dataset(ds)
            if not hasattr(root, "global_offset"):
                raise TypeError(
                    f"Dataset {type(root).__name__} does not expose a global_offset attribute. "
                    "Register it with a SourceLabelRegistry before wrapping in CombinedCoralDataset."
                )
            offsets.append(int(root.global_offset))
            names.append(getattr(root, "SOURCE_NAME", type(root).__name__))

        if len(set(offsets)) != len(offsets):
            raise ValueError(
                f"Datasets do not have disjoint global offsets ({list(zip(names, offsets, strict=False))}); "
                "did you forget to call SourceLabelRegistry on them?"
            )

        self._datasets = datasets
        self._concat = torch.utils.data.ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int):
        return self._concat[idx]

    def collate_fn(self, batch):
        return _joint_collate(batch)
