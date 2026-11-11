"""Coralscapes V2 PyTorch dataset.

Wraps the HuggingFace Coralscapes V2 dataset and emits ``(image, source_labels)``
tuples where ``source_labels`` is in the **native Coralscapes V2 1..95 label
space**. The static mapping into the MERMAID benthic-attribute target space
lives in
:mod:`mermaidseg.dataset_reconciliation.label_mapping.coralscapes_v2_to_mermaid`.
The published collapse back onto the original Coralscapes 39 classes is
:data:`CORALSCAPES_V2_TO_LEGACY_NAME`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from mermaidseg.datasets.coralscapes.coralscapes_dataset import CORALSCAPES_ID2NAME
from mermaidseg.datasets.utils import emit_dataset_warning

logger = logging.getLogger(__name__)

CORALSCAPES_V2_HF_REPO = "josauder/coralscapesV2"

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
    45: "massive alive",
    46: "massive bleached",
    47: "massive dead",
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

# Legacy column of the CoralscapesV2 dataset card (the paper's 95 -> 39 map).
# The card calls V1 class 12 "hard substrate"; this repo's Coralscapes ids name
# that class "unknown hard substrate".
CORALSCAPES_V2_TO_LEGACY_NAME: dict[str, str] = {
    "acanthaster planci": "crown of thorn",
    "acropora alive": "acropora alive",
    "acropora bleached": "branching bleached",
    "acropora dead": "branching dead",
    "algae covered brain coral": "algae covered substrate",
    "algae covered branching coral": "algae covered substrate",
    "algae covered lobophylliidae": "algae covered substrate",
    "algae covered massive coral": "algae covered substrate",
    "algae covered pocillopora": "algae covered substrate",
    "algae covered porites": "algae covered substrate",
    "algae covered substrate": "algae covered substrate",
    "algae covered table acropora": "algae covered substrate",
    "anemone": "anemone",
    "background": "background",
    "brain coral alive": "meandering alive",
    "brain coral bleached": "meandering bleached",
    "brain coral dead": "meandering dead",
    "branching coral alive": "branching alive",
    "branching coral bleached": "branching bleached",
    "branching coral dead": "branching dead",
    "cirrhipathes": "other coral alive",
    "dark": "dark",
    "dead giant clam": "dead clam",
    "feather worm": "other animal",
    "fish": "fish",
    "fungiidae alive": "other coral alive",
    "fungiidae bleached": "other coral bleached",
    "fungiidae dead": "other coral dead",
    "galaxea alive": "other coral alive",
    "galaxea bleached": "other coral bleached",
    "galaxea dead": "massive/meandering dead",
    "giant clam": "clam",
    "goniopora alive": "other coral alive",
    "goniopora dead": "other coral dead",
    "hard coral alive": "other coral alive",
    "hard coral bleached": "other coral bleached",
    "hard coral dead": "other coral dead",
    "hard substrate": "unknown hard substrate",
    "human": "human",
    "human-made structure": "unknown hard substrate",
    "leather coral alive": "other coral alive",
    "lobophylliidae alive": "meandering alive",
    "lobophylliidae bleached": "meandering bleached",
    "lobophylliidae dead": "meandering dead",
    "massive alive": "massive/meandering alive",
    "massive bleached": "massive/meandering bleached",
    "massive dead": "massive/meandering dead",
    "meandering coral alive": "meandering alive",
    "meandering coral bleached": "meandering bleached",
    "meandering coral dead": "meandering dead",
    "millepora": "millepora",
    "millepora bleached": "branching bleached",
    "millepora dead": "branching dead",
    "nephtheidae alive": "other coral alive",
    "other animal": "other animal",
    "other coral alive": "other coral alive",
    "other coral bleached": "other coral bleached",
    "other coral dead": "other coral dead",
    "pavona alive": "other coral alive",
    "pavona bleached": "other coral bleached",
    "pavona dead": "other coral dead",
    "pocillopora alive": "pocillopora alive",
    "pocillopora bleached": "branching bleached",
    "pocillopora dead": "branching dead",
    "porites alive": "massive/meandering alive",
    "porites bleached": "massive/meandering bleached",
    "porites dead": "massive/meandering dead",
    "rubble": "rubble",
    "sand": "sand",
    "sea cucumber": "sea cucumber",
    "sea urchin": "sea urchin",
    "seagrass": "seagrass",
    "seriatopora alive": "branching alive",
    "seriatopora bleached": "branching bleached",
    "seriatopora dead": "branching dead",
    "soft coral (malacalcyonacea) alive": "other coral alive",
    "soft coral (malacalcyonacea) bleached": "other coral bleached",
    "sponge": "sponge",
    "stylophora alive": "stylophora alive",
    "stylophora bleached": "branching bleached",
    "stylophora dead": "branching dead",
    "table acropora alive": "table acropora alive",
    "table acropora bleached": "branching bleached",
    "table acropora dead": "table acropora dead",
    "thin plate/encrusting coral alive": "other coral alive",
    "thin plate/encrusting coral bleached": "other coral bleached",
    "thin plate/encrusting coral dead": "other coral dead",
    "transect line": "transect line",
    "transect tools": "transect tools",
    "trash": "trash",
    "turbinaria": "turbinaria",
    "turbinaria dead": "other coral dead",
    "water column": "background",
    "water surface": "background",
    "xeniidae alive": "other coral alive",
}


def coralscapes_v2_to_legacy_lut() -> NDArray[np.uint8]:
    """Map a native CoralscapesV2 mask onto the original 39-class ids.

    Index 0 stays 0 (unannotated). Indexes 1..95 are the V1 ids of
    :data:`CORALSCAPES_V2_TO_LEGACY_NAME`.
    """
    legacy_ids = {name: class_id for class_id, name in CORALSCAPES_ID2NAME.items()}
    fine_names = set(CORALSCAPES_V2_ID2NAME.values())
    if set(CORALSCAPES_V2_TO_LEGACY_NAME) != fine_names:
        missing = sorted(fine_names - set(CORALSCAPES_V2_TO_LEGACY_NAME))
        extra = sorted(set(CORALSCAPES_V2_TO_LEGACY_NAME) - fine_names)
        raise RuntimeError(
            f"95-to-39 map does not match the V2 names: missing={missing} extra={extra}"
        )
    unknown = sorted(set(CORALSCAPES_V2_TO_LEGACY_NAME.values()) - set(legacy_ids))
    if unknown:
        raise RuntimeError(
            f"95-to-39 map uses names that are not Coralscapes V1 classes: {unknown}"
        )
    lut = np.zeros(len(CORALSCAPES_V2_ID2NAME) + 1, dtype=np.uint8)
    for class_id, name in CORALSCAPES_V2_ID2NAME.items():
        lut[class_id] = legacy_ids[CORALSCAPES_V2_TO_LEGACY_NAME[name]]
    return lut


class CoralscapesV2Dataset(Dataset[tuple[torch.Tensor | NDArray[Any], Any]]):
    """A PyTorch Dataset wrapping the HuggingFace
    `Coralscapes V2 <https://huggingface.co/datasets/josauder/coralscapesV2>`_
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
        hf_repo_id: HuggingFace dataset repo ID or local dataset directory (e.g. a sparsified
            copy from ``scripts/build_sparse_dense_datasets.py``) with the same columns.
        hf_revision: Optional hub revision. Required with ``image_keep`` on a hub repo so the
            allowlist cannot attach to a later version of the split.
        image_keep: Optional parquet allowlist of training rows. Masks are not rewritten.
        image_keep_n_rows: Split length the allowlist was built from, checked before selecting.
        source_id2name: Mapping from local source IDs (1..N) to Coralscapes V2 class names.
        source_name2id: Inverse of source_id2name.
        num_source_classes: ``len(source_id2name) + 1`` (incl. background).
    """

    SOURCE_NAME = "coralscapes_v2"

    split: str | list[str] | None
    transform: A.BasicTransform | None
    class_subset: list[str] | None
    hf_repo_id: str
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    num_source_classes: int
    _global_offset: int

    def __init__(
        self,
        split: str | list[str] | None = None,
        transform: A.BasicTransform | None = None,
        class_subset: list[str] | None = None,
        hf_repo_id: str = CORALSCAPES_V2_HF_REPO,
        hf_revision: str | None = None,
        image_keep: str | None = None,
        image_keep_n_rows: int | None = None,
    ):
        from datasets import concatenate_datasets, load_dataset

        from mermaidseg.datasets.data_size import apply_hf_image_keep

        self.split = split
        self.transform = transform
        self.class_subset = class_subset
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision
        self.image_keep = image_keep
        self.image_keep_n_rows = image_keep_n_rows
        self._global_offset = 0
        if image_keep is not None:
            if image_keep_n_rows is None:
                raise ValueError("image_keep requires image_keep_n_rows")
            if hf_revision is None and not Path(hf_repo_id).is_dir():
                raise ValueError("image_keep on a hub dataset requires hf_revision")

        revision_kwargs = {"revision": hf_revision} if hf_revision else {}
        hf_dataset = load_dataset(self.hf_repo_id, **revision_kwargs)
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

        if image_keep is not None:
            self.dataset = apply_hf_image_keep(
                self.dataset, image_keep, self.SOURCE_NAME, image_keep_n_rows
            )

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
        The global offset is folded directly into the foreground entries so ``_load_item`` does not
        need a separate ``np.where`` shift pass. ``int32`` keeps the emitted mask small.
        """
        native_id2name = CORALSCAPES_V2_ID2NAME
        max_native = max(native_id2name) + 1
        lookup = np.zeros(max_native, dtype=np.int32)
        offset = self._global_offset
        for native_id, name in native_id2name.items():
            local_id = self.source_name2id.get(name, 0)
            lookup[native_id] = local_id + offset if local_id else 0
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
        # Fetch the HF row once: ``datasets`` eagerly decodes every Image column on each row access,
        # so indexing twice would decode both the image and the label PNG twice.
        row = self.dataset[idx]
        image = np.array(row["image"])
        # The native label is uint8; index the (offset-folded, int32) lookup directly.
        native_mask = np.asarray(row["label"])
        mask = self._native_to_local[native_mask]

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
