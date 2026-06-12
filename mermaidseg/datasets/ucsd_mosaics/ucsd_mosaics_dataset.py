"""UCSD Mosaics PyTorch dataset.

Wraps the HuggingFace mirror at ``josauder/UCSD-mosaics-mirror`` (the GT-Clean
variant of the UCSD Mosaics dense semantic segmentation dataset; Edwards et al.
2017 + Alonso et al. 2019 + Raine et al. 2024) and emits
``(image, source_labels)`` tuples where ``source_labels`` is in the **native
UCSD Mosaics 1..34 label space** taken straight from the HF ``classes.json``.
Pixel value ``0`` in the source mask is the unlabeled / unidentified ignore
label and is mapped to background.

See ``README.md`` for dataset history and ``nbs/ucsd_mosaics_EDA.ipynb`` for
a hands-on tour of the HuggingFace mirror.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import albumentations as A
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class UCSDMosaicsDataset(Dataset[tuple[torch.Tensor | NDArray[Any], Any]]):
    """A PyTorch Dataset wrapping the HuggingFace
    `josauder/UCSD-mosaics-mirror <https://huggingface.co/datasets/josauder/UCSD-mosaics-mirror>`_
    dataset (Edwards et al. 2017 mosaics, Alonso et al. 2019 patch split,
    Raine et al. 2024 GT-Clean cleanup).

    Each item is ``(image, source_labels)`` where ``source_labels`` is an
    integer mask in the native UCSD Mosaics 1..34 label space (or shifted into
    the joint global source-label space once registered with a
    :class:`SourceLabelRegistry`). Pixel value ``0`` (``unlabeled /
    unidentified``) in the source mask is treated as background and stays
    ``0``.

    The HuggingFace mirror exposes one split per dive site (16 sites total,
    lowercased, e.g. ``"fr3"``, ``"palwave13"``). Each row carries
    ``image`` (PIL RGB 512x512), ``label`` (PIL ``L`` 512x512 with values in
    ``0..34``), ``filename``, ``site`` (original case, e.g. ``"PALWave13"``),
    and ``original_split`` (``"train"`` or ``"test"`` from the source GT-Clean
    release).

    Args:
        whitelist_sites: Optional list of HF split names (lowercased site IDs,
            e.g. ``["fr3", "palwave13"]``) to load. Mutually exclusive with
            ``blacklist_sites``. Defaults to all 16 sites.
        blacklist_sites: Optional list of HF split names to exclude.
        whitelist_original_splits: Optional subset of ``{"train", "test"}``
            (the source GT-Clean train/test field). Mutually exclusive with
            ``blacklist_original_splits``.
        blacklist_original_splits: Optional ``original_split`` values to drop.
        class_subset: Optional list of class names (from ``classes.json``) to
            retain. Non-subset classes (and the ignore label) map to ``0``.
        transform: Optional Albumentations transform applied to image and mask.
        hf_repo_id: HuggingFace dataset repo ID; override only for testing or
            to point at a fork of the mirror.

    Attributes:
        SOURCE_NAME: ``"ucsd_mosaics"``.
        class_table: List of ``{id, name, description, color_rgb, ...}`` dicts
            from ``classes.json``, sorted by ``id`` (1..34).
        source_id2name: Mapping from local source IDs (1..N) to class names.
            When ``class_subset`` is ``None`` the local IDs are kept aligned
            with the ``classes.json`` IDs.
        source_name2id: Inverse of ``source_id2name``.
        num_source_classes: ``len(source_id2name) + 1`` (incl. background).
    """

    SOURCE_NAME = "ucsd_mosaics"
    HF_REPO_ID = "josauder/UCSD-mosaics-mirror"
    IGNORE_LABEL_ID = 0  # "unlabeled / unidentified" -> background

    hf_repo_id: str
    whitelist_sites: list[str] | None
    blacklist_sites: list[str] | None
    whitelist_original_splits: list[str] | None
    blacklist_original_splits: list[str] | None
    transform: A.BasicTransform | None
    class_subset: list[str] | None
    class_table: list[dict[str, Any]]
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    num_source_classes: int
    sites: list[str]
    _global_offset: int

    def __init__(
        self,
        whitelist_sites: list[str] | None = None,
        blacklist_sites: list[str] | None = None,
        whitelist_original_splits: list[str] | None = None,
        blacklist_original_splits: list[str] | None = None,
        class_subset: list[str] | None = None,
        transform: A.BasicTransform | None = None,
        hf_repo_id: str = HF_REPO_ID,
    ):
        from datasets import concatenate_datasets, get_dataset_split_names, load_dataset
        from huggingface_hub import hf_hub_download

        if whitelist_sites is not None and blacklist_sites is not None:
            raise ValueError("Cannot specify both whitelist_sites and blacklist_sites.")
        if whitelist_original_splits is not None and blacklist_original_splits is not None:
            raise ValueError(
                "Cannot specify both whitelist_original_splits and blacklist_original_splits."
            )

        self.hf_repo_id = hf_repo_id
        self.whitelist_sites = whitelist_sites
        self.blacklist_sites = blacklist_sites
        self.whitelist_original_splits = whitelist_original_splits
        self.blacklist_original_splits = blacklist_original_splits
        self.transform = transform
        self.class_subset = class_subset
        self._global_offset = 0

        classes_path = hf_hub_download(
            repo_id=self.hf_repo_id, filename="classes.json", repo_type="dataset"
        )
        with open(classes_path) as f:
            self.class_table = sorted(json.load(f), key=lambda e: int(e["id"]))

        all_sites = list(get_dataset_split_names(self.hf_repo_id))
        self.sites = self._select_sites(all_sites)

        per_site = [load_dataset(self.hf_repo_id, split=site) for site in self.sites]
        self.dataset = concatenate_datasets(per_site) if len(per_site) > 1 else per_site[0]

        keep_original = self._resolved_original_splits()
        if keep_original is not None:
            self.dataset = self.dataset.filter(lambda ex: ex["original_split"] in keep_original)

        if self.class_subset is not None:
            ordered = list(self.class_subset)
        else:
            ordered = [entry["name"] for entry in self.class_table]
        self.source_id2name = dict(enumerate(ordered, start=1))
        self.source_name2id = {v: k for k, v in self.source_id2name.items()}
        self.num_source_classes = len(self.source_id2name) + 1

        self._native_to_local = self._build_native_to_local()

    def _select_sites(self, all_sites: list[str]) -> list[str]:
        """Resolve the effective list of HF site splits to load."""
        all_set = set(all_sites)
        if self.whitelist_sites is not None:
            requested = list(self.whitelist_sites)
            unknown = sorted(set(requested) - all_set)
            if unknown:
                raise ValueError(
                    f"whitelist_sites contains unknown site(s) {unknown}; "
                    f"valid sites are {sorted(all_set)}"
                )
            return [s for s in all_sites if s in set(requested)]
        if self.blacklist_sites is not None:
            unknown = sorted(set(self.blacklist_sites) - all_set)
            if unknown:
                raise ValueError(
                    f"blacklist_sites contains unknown site(s) {unknown}; "
                    f"valid sites are {sorted(all_set)}"
                )
            blacklist = set(self.blacklist_sites)
            return [s for s in all_sites if s not in blacklist]
        return list(all_sites)

    def _resolved_original_splits(self) -> set[str] | None:
        """Resolve the effective allowed ``original_split`` values, or ``None``."""
        if self.whitelist_original_splits is not None:
            return set(self.whitelist_original_splits)
        if self.blacklist_original_splits is not None:
            return {"train", "test"} - set(self.blacklist_original_splits)
        return None

    def _build_native_to_local(self) -> np.ndarray:
        """Build a vectorized lookup from native UCSD IDs (0..34) to local source IDs.

        Native ID ``0`` (ignore) and any class not present in ``class_subset``
        map to ``0`` (background).
        """
        max_native = max(int(entry["id"]) for entry in self.class_table) + 1
        lookup = np.zeros(max_native, dtype=np.int64)
        for entry in self.class_table:
            native_id = int(entry["id"])
            local_id = self.source_name2id.get(entry["name"], 0)
            lookup[native_id] = local_id
        return lookup

    def set_global_offset(self, offset: int) -> None:
        """Set the global source-label offset assigned by the registry."""
        if offset < 0:
            raise ValueError(f"global offset must be non-negative, got {offset}")
        self._global_offset = int(offset)

    @property
    def global_offset(self) -> int:
        return self._global_offset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        try:
            row = self.dataset[idx]
            image = np.array(row["image"])
            native_mask = np.asarray(row["label"], dtype=np.int64)
            mask = self._native_to_local[native_mask]
        except Exception as e:
            logger.warning(
                "UCSDMosaicsDataset: skipping idx=%d: %s: %s", idx, type(e).__name__, e
            )
            return None, None

        if self._global_offset:
            mask = np.where(mask > 0, mask + self._global_offset, mask).astype(
                mask.dtype, copy=False
            )

        if self.transform:
            try:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"].transpose(2, 0, 1)
                mask = transformed["mask"]
            except Exception as e:
                logger.warning(
                    "UCSDMosaicsDataset: transform failed for idx=%d: %s: %s",
                    idx,
                    type(e).__name__,
                    e,
                )
                return None, None

        return image, mask

    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter out failed loads and stack into batched tensors."""
        batch_size = len(batch)
        filtered = [(img, msk) for img, msk in batch if img is not None and msk is not None]
        n_skipped = batch_size - len(filtered)
        if n_skipped > 0:
            logger.warning(
                "UCSDMosaicsDataset.collate_fn: skipped %d/%d items in batch due to load errors",
                n_skipped,
                batch_size,
            )
        if len(filtered) == 0:
            logger.warning(
                "UCSDMosaicsDataset.collate_fn: entire batch of %d items was empty, "
                "returning empty tensors",
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
