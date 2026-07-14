"""Coralscapes V2 PyTorch dataset.

Wraps the HuggingFace Coralscapes V2 dataset and emits ``(image, source_labels)``
tuples where ``source_labels`` is in the **native Coralscapes V2 1..95 label
space**. The static mapping into the MERMAID benthic-attribute target space
lives in
:mod:`mermaidseg.dataset_reconciliation.label_mapping.coralscapes_v2_to_mermaid`.
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

CORALSCAPES_V2_HF_REPO = "josauder/314d3951853dad8855bd06248987f626"

CORALSCAPES_V2_ID2NAME: dict[int, str] = {
    1: "acanthaster planci",
    2: "acropora alive",
    3: "acropora bleached",
    4: "acropora dead",
    5: "algae covered brain coral",
    6: "algae covered branching coral",
    7: "algae covered lobophylliidae",
    8: "algae covered massive coral",
    9: "algae covered pocillopora",
    10: "algae covered porites",
    11: "algae covered substrate",
    12: "algae covered table acropora",
    13: "anemone",
    14: "background",
    15: "brain coral alive",
    16: "brain coral bleached",
    17: "brain coral dead",
    18: "branching coral alive",
    19: "branching coral bleached",
    20: "branching coral dead",
    21: "cirrhipathes",
    22: "dark",
    23: "dead giant clam",
    24: "feather worm",
    25: "fish",
    26: "fungiidae alive",
    27: "fungiidae bleached",
    28: "fungiidae dead",
    29: "galaxea alive",
    30: "galaxea bleached",
    31: "galaxea dead",
    32: "giant clam",
    33: "goniopora alive",
    34: "goniopora dead",
    35: "hard coral alive",
    36: "hard coral bleached",
    37: "hard coral dead",
    38: "hard substrate",
    39: "human",
    40: "human-made structure",
    41: "leather coral alive",
    42: "lobophylliidae alive",
    43: "lobophylliidae bleached",
    44: "lobophylliidae dead",
    45: "massive coral alive",
    46: "massive coral bleached",
    47: "massive coral dead",
    48: "meandering coral alive",
    49: "meandering coral bleached",
    50: "meandering coral dead",
    51: "millepora",
    52: "millepora bleached",
    53: "millepora dead",
    54: "nephtheidae alive",
    55: "other animal",
    56: "other coral alive",
    57: "other coral bleached",
    58: "other coral dead",
    59: "pavona alive",
    60: "pavona bleached",
    61: "pavona dead",
    62: "pocillopora alive",
    63: "pocillopora bleached",
    64: "pocillopora dead",
    65: "porites alive",
    66: "porites bleached",
    67: "porites dead",
    68: "rubble",
    69: "sand",
    70: "sea cucumber",
    71: "sea urchin",
    72: "seagrass",
    73: "seriatopora alive",
    74: "seriatopora bleached",
    75: "seriatopora dead",
    76: "soft coral (malacalcyonacea) alive",
    77: "soft coral (malacalcyonacea) bleached",
    78: "sponge",
    79: "stylophora alive",
    80: "stylophora bleached",
    81: "stylophora dead",
    82: "table acropora alive",
    83: "table acropora bleached",
    84: "table acropora dead",
    85: "thin plate/encrusting coral alive",
    86: "thin plate/encrusting coral bleached",
    87: "thin plate/encrusting coral dead",
    88: "transect line",
    89: "transect tools",
    90: "trash",
    91: "turbinaria",
    92: "turbinaria dead",
    93: "water column",
    94: "water surface",
    95: "xeniidae alive",
}


class CoralscapesV2Dataset(Dataset[tuple[torch.Tensor | NDArray[Any], Any]]):
    """A PyTorch Dataset wrapping the HuggingFace
    `Coralscapes V2 <https://huggingface.co/datasets/josauder/314d3951853dad8855bd06248987f626>`_
    dataset.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is the native Coralscapes V2 1..95 integer mask (or shifted
    by ``global_offset`` once the dataset is registered with a
    :class:`SourceLabelRegistry`). Pixels labelled ``0`` in the HF dataset
    remain ``0`` (background).

    Attributes:
        SOURCE_NAME: ``"coralscapes_v2"``.
        split: Optional HuggingFace split(s). A single name (``"train"`` /
            ``"validation"`` / ``"test"``), a list of names to concatenate, or
            ``None`` for all three splits.
        transform: Optional Albumentations transform applied to image and mask.
        class_subset: Optional list of source-space (Coralscapes V2) class names to retain.
        source_id2name: Mapping from local source IDs (1..N) to Coralscapes V2 class names.
        source_name2id: Inverse of source_id2name.
        num_source_classes: ``len(source_id2name) + 1`` (incl. background).
    """

    SOURCE_NAME = "coralscapes_v2"

    split: str | list[str] | None
    transform: A.BasicTransform | None
    class_subset: list[str] | None
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    num_source_classes: int
    _global_offset: int

    def __init__(
        self,
        split: str | list[str] | None = None,
        transform: A.BasicTransform | None = None,
        class_subset: list[str] | None = None,
    ):
        from datasets import concatenate_datasets, load_dataset

        self.split = split
        self.transform = transform
        self.class_subset = class_subset
        self._global_offset = 0

        hf_dataset = load_dataset(CORALSCAPES_V2_HF_REPO)
        if self.split is None:
            self.dataset = concatenate_datasets(
                [
                    hf_dataset["train"],
                    hf_dataset["validation"],
                    hf_dataset["test"],
                ]
            )
        elif isinstance(self.split, str):
            self.dataset = hf_dataset[self.split]
        else:
            self.dataset = concatenate_datasets([hf_dataset[s] for s in self.split])

        if self.class_subset is not None:
            ordered = list(self.class_subset)
        else:
            ordered = [CORALSCAPES_V2_ID2NAME[i] for i in sorted(CORALSCAPES_V2_ID2NAME)]
        self.source_id2name = dict(enumerate(ordered, start=1))
        self.source_name2id = {v: k for k, v in self.source_id2name.items()}
        self.num_source_classes = len(self.source_id2name) + 1

        self._native_to_local = self._build_native_to_local()

    def _build_native_to_local(self) -> np.ndarray:
        """Build a vectorized lookup from native Coralscapes V2 IDs to local source IDs.

        Native ID ``0`` and any class not present in ``class_subset`` map to ``0`` (background).
        """
        native_id2name = CORALSCAPES_V2_ID2NAME
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
            emit_dataset_warning(
                f"CoralscapesV2Dataset: skipping idx={idx}: {type(e).__name__}: {e}"
            )
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
                "CoralscapesV2Dataset.collate_fn: skipped %d/%d items in batch due to load errors",
                n_skipped,
                batch_size,
            )
        if len(filtered) == 0:
            logger.warning(
                "CoralscapesV2Dataset.collate_fn: entire batch of %d items was empty, returning empty tensors",
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
