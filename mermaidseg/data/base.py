"""Shared logic of the readers over synced segdata-spec dataset versions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import Dataset

from mermaidseg.data.dense_read import read_rgb
from segdata_spec.read import DatasetVersion

_MAX_LISTED_PATHS = 5


def _select_groups(images: pd.DataFrame, groups: Collection[str] | None) -> pd.DataFrame:
    """Return the rows whose ``group`` is in ``groups`` (all rows for ``None``)."""
    if groups is None:
        return images
    unknown = sorted(set(groups) - set(images["group"]))
    if unknown:
        raise ValueError(f"groups not in the dataset version: {unknown}")
    return images[images["group"].isin(list(groups))]


def _check_synced(images: pd.DataFrame, synced_groups: frozenset[str] | None) -> None:
    """Raise unless every group of ``images`` was synced (``None`` means all were)."""
    if synced_groups is None:
        return
    missing = sorted(set(images["group"]) - synced_groups)
    if missing:
        raise ValueError(
            f"groups not synced into the cache: {missing}; run scripts/sync_data.py "
            "with a data config that selects them"
        )


def _select_image_ids(images: pd.DataFrame, image_ids: Collection[str] | None) -> pd.DataFrame:
    """Return the rows whose ``image_id`` is in ``image_ids`` (all rows for ``None``)."""
    if image_ids is None:
        return images
    unknown = sorted(set(image_ids) - set(images["image_id"]))
    if unknown:
        raise ValueError(f"image_ids not in the reader's selection: {unknown}")
    return images[images["image_id"].isin(list(image_ids))]


def _next_random_idx(idx: int, n: int) -> int:
    """Return a uniform random index in ``[0, n)`` other than ``idx`` (``idx`` when ``n == 1``)."""
    if n <= 1:
        return idx
    next_idx = int(np.random.randint(0, n - 1))
    return next_idx + 1 if next_idx >= idx else next_idx


class VersionDataset(Dataset[tuple[Any, Any]]):
    """Source-space dataset over one synced segdata-spec version.

    Items are ``(image, mask)``; the mask holds 0 for background and ``global_offset +
    label_id`` for a label of ``labels.parquet``. Subclasses set ``KIND`` and supply the mask.

    ``__init__`` runs its steps in this order: group selection and the synced-groups check;
    ``_filter_images``, which may narrow subclass state (``PointDataset`` narrows ``self.points``),
    then the ``image_ids`` subset; the cache check, through ``_row_paths``; ``self.images`` set.

    Attributes:
        source_name: The dataset name from the manifest.
        global_offset: Added to every foreground ``label_id`` of the mask.
        images: The selected rows of ``images.parquet``, indexed ``0..len-1``.
        split: The split this reader serves.
    """

    KIND: ClassVar[Literal["points", "dense"]]

    source_name: str
    global_offset: int
    images: pd.DataFrame
    split: Literal["train", "val", "test"]
    transform: Callable[..., dict[str, Any]] | None

    def __init__(
        self,
        version: DatasetVersion,
        *,
        split: Literal["train", "val", "test"],
        global_offset: int,
        groups: Collection[str] | None = None,
        transform: Callable[..., dict[str, Any]] | None = None,
        image_ids: Collection[str] | None = None,
    ) -> None:
        """Select the images and verify the cache.

        Args:
            version: The opened version.
            split: The split; only ``"train"`` resamples a sample whose mask is empty.
            global_offset: Added to every foreground ``label_id`` of the mask (see
                :func:`mermaidseg.data.factory.source_offsets`).
            groups: Image groups to keep; ``None`` keeps all.
            transform: Called as ``transform(image=, mask=)`` and returning a dict with both.
            image_ids: Keeps only these images (an evaluation subset); ``None`` keeps all.

        Raises:
            ValueError: The reader kind differs from the manifest kind, ``labels.parquet`` repeats
                a name (compared case-folded) or holds a ``label_id`` below 1, a group is unknown or not in
                ``version.synced_groups``, or an image id is not in the selection.
            FileNotFoundError: A selected file is not cached.
        """
        manifest = version.manifest
        if manifest.kind != self.KIND:
            raise ValueError(
                f"{type(self).__name__} reads {self.KIND!r} versions; "
                f"{manifest.dataset} {manifest.version} is {manifest.kind!r}"
            )
        duplicated = sorted(
            n
            for n, count in Counter(
                name.casefold() for name in version.labels.column("name").to_pylist()
            ).items()
            if count > 1
        )
        if duplicated:
            raise ValueError(
                f"dataset {manifest.dataset!r}: labels.parquet repeats the names {duplicated}"
            )
        below_one = sorted(i for i in version.labels.column("label_id").to_pylist() if i < 1)
        if below_one:
            raise ValueError(
                f"dataset {manifest.dataset!r}: labels.parquet holds the label_ids {below_one}; "
                "0 is background and ids start at 1"
            )
        self.version = version
        self.source_name = manifest.dataset
        self.global_offset = global_offset
        self.split = split
        self.transform = transform
        self._source_rows: NDArray[np.int64] | None = None

        images = _select_groups(version.images.to_pandas(), groups)
        _check_synced(images, version.synced_groups)
        images = _select_image_ids(self._filter_images(images), image_ids)
        self._check_cache(images)
        self.images = images.reset_index(drop=True)

    def _filter_images(self, images: pd.DataFrame) -> pd.DataFrame:
        """Hook: narrow the group-selected rows before the cache check."""
        return images

    def _row_paths(self, row: Mapping[str, Any]) -> list[Path]:
        """Hook: return every cached file an image row needs."""
        return [self.version.image_path(row)]

    def _read_mask(self, row: Mapping[str, Any], image_shape: tuple[int, int]) -> NDArray[np.int32]:
        """Hook: return the int32 mask of an image, in offset-shifted label ids."""
        raise NotImplementedError

    def _check_cache(self, images: pd.DataFrame) -> None:
        """Raise ``FileNotFoundError`` naming the cached files the images need and lack."""
        missing_paths = [
            p for row in images.to_dict("records") for p in self._row_paths(row) if not p.is_file()
        ]
        if missing_paths:
            listed = ", ".join(str(p) for p in missing_paths[:_MAX_LISTED_PATHS])
            raise FileNotFoundError(
                f"{len(missing_paths)} cached file(s) missing, e.g. {listed}; run sync first"
            )

    def set_source_rows(self, rows: NDArray[np.int64] | None) -> None:
        """Set the concept-source-mask row of each image, or ``None`` to return pairs.

        With rows set, items are ``(image, mask, row)``, the row of the image actually returned.
        See :func:`mermaidseg.dataset_reconciliation.concept_source_mask.attach_source_rows`.

        Raises:
            ValueError: ``rows`` does not hold one row per image.
        """
        if rows is not None and len(rows) != len(self.images):
            raise ValueError(f"{len(rows)} source rows for {len(self.images)} images")
        self._source_rows = rows

    @property
    def label_targets(self) -> dict[str, str | None]:
        """Label name -> ``mermaid_benthic_attribute`` over ``labels.parquet``.

        A null or empty attribute is ``None``: the label maps to no target.
        """
        labels = self.version.labels
        return {
            name: attribute or None
            for name, attribute in zip(
                labels.column("name").to_pylist(),
                labels.column("mermaid_benthic_attribute").to_pylist(),
                strict=True,
            )
        }

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def _load(self, idx: int) -> tuple[Any, Any]:
        row = self.images.iloc[idx].to_dict()
        image = read_rgb(self.version.image_path(row))
        mask = self._read_mask(row, (image.shape[0], image.shape[1]))
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            return transformed["image"].transpose(2, 0, 1), transformed["mask"]
        return image, mask

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        """Return ``(image, mask)``, or ``(image, mask, source_row)`` once source rows are set.

        On the train split a sample whose mask has no label is replaced by another random image,
        up to ``len(self)`` loads.

        Raises:
            RuntimeError: The dataset is empty, or every train sample had an empty mask.
        """
        n = len(self)
        if n == 0:
            raise RuntimeError(f"{type(self).__name__}: dataset is empty")
        for _ in range(n):
            image, mask = self._load(idx)
            if self.split != "train" or _has_label(mask):
                if self._source_rows is None:
                    return image, mask
                return image, mask, int(self._source_rows[idx])
            idx = _next_random_idx(idx, n)
        raise RuntimeError(f"{type(self).__name__}: all {n} items were empty (no labels)")


def _has_label(mask: Any) -> bool:
    return bool(np.any(np.asarray(mask) > 0))
