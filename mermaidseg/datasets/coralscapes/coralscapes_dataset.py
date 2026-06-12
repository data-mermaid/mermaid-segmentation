"""Coralscapes PyTorch dataset.

Wraps the HuggingFace Coralscapes dataset and emits ``(image, source_labels)``
tuples where ``source_labels`` is in the **native Coralscapes 1..39 label
space**. The static mapping into the MERMAID benthic-attribute target space
lives in
:mod:`mermaidseg.dataset_reconciliation.label_mapping.coralscapes_to_mermaid`.
"""

from __future__ import annotations

import logging
from typing import Any

import albumentations as A
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from mermaidseg.datasets.utils import emit_dataset_warning

logger = logging.getLogger(__name__)


CORALSCAPES_ID2NAME: dict[int, str] = {
    1: "seagrass",
    2: "trash",
    3: "other coral dead",
    4: "other coral bleached",
    5: "sand",
    6: "other coral alive",
    7: "human",
    8: "transect tools",
    9: "fish",
    10: "algae covered substrate",
    11: "other animal",
    12: "unknown hard substrate",
    13: "background",
    14: "dark",
    15: "transect line",
    16: "massive/meandering bleached",
    17: "massive/meandering alive",
    18: "rubble",
    19: "branching bleached",
    20: "branching dead",
    21: "millepora",
    22: "branching alive",
    23: "massive/meandering dead",
    24: "clam",
    25: "acropora alive",
    26: "sea cucumber",
    27: "turbinaria",
    28: "table acropora alive",
    29: "sponge",
    30: "anemone",
    31: "pocillopora alive",
    32: "table acropora dead",
    33: "meandering bleached",
    34: "stylophora alive",
    35: "sea urchin",
    36: "meandering alive",
    37: "meandering dead",
    38: "crown of thorn",
    39: "dead clam",
}


class CoralscapesDataset(Dataset[tuple[torch.Tensor | NDArray[Any], Any]]):
    """A PyTorch Dataset wrapping the HuggingFace
    `EPFL-ECEO/coralscapes <https://huggingface.co/datasets/EPFL-ECEO/coralscapes>`_
    dataset.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is the native Coralscapes 1..39 integer mask (or shifted
    by ``global_offset`` once the dataset is registered with a
    :class:`SourceLabelRegistry`). Pixels labelled ``0`` in the HF dataset
    remain ``0`` (background).

    Attributes:
        SOURCE_NAME: ``"coralscapes"``.
        split: Optional dataset split identifier (``"train"`` / ``"validation"`` / ``"test"``).
        transform: Optional Albumentations transform applied to image and mask.
        class_subset: Optional list of source-space (Coralscapes) class names to retain.
        source_id2name: Mapping from local source IDs (1..N) to Coralscapes class names.
        source_name2id: Inverse of source_id2name.
        num_source_classes: ``len(source_id2name) + 1`` (incl. background).
    """

    SOURCE_NAME = "coralscapes"

    split: str | None
    transform: A.BasicTransform | None
    class_subset: list[str] | None
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    num_source_classes: int
    _global_offset: int

    def __init__(
        self,
        split: str | None = None,
        transform: A.BasicTransform | None = None,
        class_subset: list[str] | None = None,
    ):
        from datasets import concatenate_datasets, load_dataset

        self.split = split
        self.transform = transform
        self.class_subset = class_subset
        self._global_offset = 0

        self.dataset = load_dataset("EPFL-ECEO/coralscapes")
        if self.split is not None:
            self.dataset = self.dataset[self.split]
        else:
            self.dataset = concatenate_datasets(
                [
                    self.dataset["train"],
                    self.dataset["validation"],
                    self.dataset["test"],
                ]
            )

        if self.class_subset is not None:
            ordered = list(self.class_subset)
        else:
            ordered = [CORALSCAPES_ID2NAME[i] for i in sorted(CORALSCAPES_ID2NAME)]
        self.source_id2name = dict(enumerate(ordered, start=1))
        self.source_name2id = {v: k for k, v in self.source_id2name.items()}
        self.num_source_classes = len(self.source_id2name) + 1

        self._native_to_local = self._build_native_to_local()

    def _build_native_to_local(self) -> np.ndarray:
        """Build a vectorized lookup from native Coralscapes IDs to local source IDs.

        Native ID ``0`` and any class not present in ``class_subset`` map to ``0`` (background).
        """
        native_id2name = CORALSCAPES_ID2NAME
        max_native = max(native_id2name) + 1
        lookup = np.zeros(max_native, dtype=np.int64)
        for native_id, name in native_id2name.items():
            local_id = self.source_name2id.get(name, 0)
            lookup[native_id] = local_id
        return lookup

    def set_source_vocabulary(
        self,
        source_id2name: dict[int, str],
        source_name2id: dict[str, int],
        num_source_classes: int,
    ) -> None:
        """Replace local source maps and rebuild the native-to-local lookup.

        Called by :func:`mermaidseg.dataset_reconciliation.split_wiring.apply_vocabularies`
        when the registry canonicalizes vocabulary across splits. Rebuilding
        ``_native_to_local`` keeps emitted mask IDs aligned with registry lookup
        tables.
        """
        self.source_id2name = dict(source_id2name)
        self.source_name2id = dict(source_name2id)
        self.num_source_classes = int(num_source_classes)
        self._native_to_local = self._build_native_to_local()

    def set_global_offset(self, offset: int) -> None:
        """Set the global source-label offset assigned by the registry."""
        if offset < 0:
            raise ValueError(f"global offset must be non-negative, got {offset}")
        self._global_offset = int(offset)
        self._native_to_local = self._build_native_to_local()

    @property
    def global_offset(self) -> int:
        return self._global_offset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        """Return ``(image, source_labels)`` for ``idx``.

        On any internal load/transform error we emit a warning to logger + stdout + stderr and
        return ``(None, None)``. :meth:`collate_fn` filters out these placeholders, so a failed item
        drops out of the batch instead of crashing the loader.
        """
        try:
            return self._load_item(idx)
        except Exception as e:
            emit_dataset_warning(f"CoralscapesDataset: skipping idx={idx}: {type(e).__name__}: {e}")
            return None, None

    def _load_item(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        image = np.array(self.dataset[idx]["image"])
        native_mask = np.asarray(self.dataset[idx]["label"], dtype=np.int64)
        mask = self._native_to_local[native_mask]

        if self._global_offset:
            mask = np.where(mask > 0, mask + self._global_offset, mask).astype(
                mask.dtype, copy=False
            )

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].transpose(2, 0, 1)
            mask = transformed["mask"]

        return image, mask

    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter out ``(None, None)`` items (failed loads) and stack into batched tensors.

        :meth:`__getitem__` returns ``(None, None)`` for items it fails to load; this filter drops
        them so a failed item simply leaves the batch instead of crashing the loader.
        """
        batch_size = len(batch)
        filtered = [(img, msk) for img, msk in batch if img is not None and msk is not None]
        n_skipped = batch_size - len(filtered)
        if n_skipped > 0:
            logger.warning(
                "CoralscapesDataset.collate_fn: skipped %d/%d items in batch due to load errors",
                n_skipped,
                batch_size,
            )
        if len(filtered) == 0:
            logger.warning(
                "CoralscapesDataset.collate_fn: entire batch of %d items was empty, returning empty tensors",
                batch_size,
            )
            return torch.tensor([]), torch.tensor([])

        images, source_labels = zip(*filtered, strict=False)

        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
            source_labels = torch.stack(source_labels)
        else:
            images = torch.stack(
                [torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in images]
            )
            source_labels = torch.stack(
                [
                    torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
                    for mask in source_labels
                ]
            )

        return images, source_labels
