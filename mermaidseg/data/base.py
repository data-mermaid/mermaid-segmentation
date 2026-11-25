"""Shared logic of the readers over synced segdata-spec dataset versions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

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


def _ordered_label_ids(counts: pd.Series) -> list[int]:
    """Return the label ids with a positive count, by count descending then id ascending."""
    present = counts[counts > 0]
    frame = pd.DataFrame({"label_id": present.index, "n": present.to_numpy()})
    ordered = frame.sort_values(["n", "label_id"], ascending=[False, True])
    return [int(label_id) for label_id in ordered["label_id"]]


def _next_random_idx(idx: int, n: int) -> int:
    """Return a uniform random index in ``[0, n)`` other than ``idx`` (``idx`` when ``n == 1``)."""
    if n <= 1:
        return idx
    next_idx = int(np.random.randint(0, n - 1))
    return next_idx + 1 if next_idx >= idx else next_idx


class VersionDataset(Dataset[tuple[Any, Any]]):
    """Source-space dataset over one synced segdata-spec version.

    Items are ``(image, mask)``; the mask holds 0 for background and local source ids ``1..N``
    (shifted by the global offset), where ``N`` is the number of labels present in the selection.
    Subclasses set ``KIND`` and supply the mask.

    ``__init__`` runs its steps in this order: group selection and the synced-groups check;
    ``_filter_images``, which may narrow subclass state (``PointDataset`` narrows ``self.points``),
    then the ``image_ids`` subset; the cache check, through
    ``_row_paths``; ``self.images`` set; ``_label_counts``; the vocabulary; ``_apply_offset``.

    Attributes:
        SOURCE_NAME: The dataset name from the manifest.
        images: The selected rows of ``images.parquet``, indexed ``0..len-1``.
        images_dropped: Images removed because a cached file is missing (``drop`` mode).
        split: The split this reader serves.
        source_id2name: Local source id to label name.
        source_name2id: Inverse of ``source_id2name``.
        source_id2label_id: Local source id to ``label_id`` in the version.
        num_source_classes: ``len(source_id2name) + 1`` (including background).
    """

    KIND: ClassVar[Literal["points", "dense"]]

    SOURCE_NAME: str
    images: pd.DataFrame
    images_dropped: int
    split: Literal["train", "val", "test"]
    transform: Callable[..., dict[str, Any]] | None
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    source_id2label_id: dict[int, int]
    num_source_classes: int

    def __init__(
        self,
        version: DatasetVersion,
        *,
        split: Literal["train", "val", "test"],
        groups: Collection[str] | None = None,
        transform: Callable[..., dict[str, Any]] | None = None,
        on_missing_images: Literal["fail", "drop"] = "fail",
        image_ids: Collection[str] | None = None,
    ) -> None:
        """Select the images, verify the cache and build the vocabulary.

        Args:
            version: The opened version.
            split: The split; only ``"train"`` resamples a sample whose mask is empty.
            groups: Image groups to keep; ``None`` keeps all.
            transform: Called as ``transform(image=, mask=)`` and returning a dict with both.
            on_missing_images: ``"fail"`` raises when a selected file is not cached; ``"drop"``
                removes those images and counts them in ``images_dropped``.
            image_ids: Keeps only these images (an evaluation subset); ``None`` keeps all. The
                vocabulary is built from the kept images.

        Raises:
            ValueError: The reader kind differs from the manifest kind, a group is unknown or
                not in ``version.synced_groups``, an image id is not in the selection, or
                two label ids in the selection's vocabulary share a name.
            FileNotFoundError: A selected file is not cached and ``on_missing_images`` is
                ``"fail"``.
        """
        manifest = version.manifest
        if manifest.kind != self.KIND:
            raise ValueError(
                f"{type(self).__name__} reads {self.KIND!r} versions; "
                f"{manifest.dataset} {manifest.version} is {manifest.kind!r}"
            )
        self.version = version
        self.SOURCE_NAME = manifest.dataset
        self.split = split
        self.transform = transform
        self._global_offset = 0
        self._source_rows: NDArray[np.int64] | None = None

        images = _select_groups(version.images.to_pandas(), groups)
        _check_synced(images, version.synced_groups)
        images = _select_image_ids(self._filter_images(images), image_ids)
        selected, self.images_dropped = self._check_cache(images, on_missing_images)
        self.images = selected.reset_index(drop=True)

        label_ids = _ordered_label_ids(self._label_counts())
        names = dict(
            zip(
                version.labels.column("label_id").to_pylist(),
                version.labels.column("name").to_pylist(),
                strict=True,
            )
        )
        unnamed = [i for i in label_ids if i not in names]
        if unnamed:
            raise ValueError(f"label ids missing from labels.parquet: {unnamed}")
        shared = sorted(n for n, k in Counter(names[i] for i in label_ids).items() if k > 1)
        if shared:
            raise ValueError(f"label names shared by several selected label ids: {shared}")
        self.source_id2label_id = dict(enumerate(label_ids, start=1))
        self.source_id2name = {i: names[label] for i, label in self.source_id2label_id.items()}
        self.source_name2id = {name: i for i, name in self.source_id2name.items()}
        self.num_source_classes = len(self.source_id2name) + 1
        self._apply_offset()

    def _filter_images(self, images: pd.DataFrame) -> pd.DataFrame:
        """Hook: narrow the group-selected rows before the cache check."""
        return images

    def _label_counts(self) -> pd.Series:
        """Hook: return the selection's count per ``label_id``, indexed by ``label_id``."""
        raise NotImplementedError

    def _row_paths(self, row: Mapping[str, Any]) -> list[Path]:
        """Hook: return every cached file an image row needs."""
        return [self.version.image_path(row)]

    def _read_mask(self, row: Mapping[str, Any], image_shape: tuple[int, int]) -> NDArray[np.int32]:
        """Hook: return the int32 mask of an image, in offset-shifted local ids."""
        raise NotImplementedError

    def _apply_offset(self) -> None:
        """Hook: refresh state derived from the global offset."""

    def _refresh_source_ids(self) -> None:
        """Hook: refresh state derived from ``source_id2label_id``."""
        self._apply_offset()

    def _check_cache(
        self, images: pd.DataFrame, mode: Literal["fail", "drop"]
    ) -> tuple[pd.DataFrame, int]:
        """Return the images whose files are all cached, and how many were dropped."""
        missing_paths: list[Path] = []
        present: list[bool] = []
        for row in images.to_dict("records"):
            absent = [p for p in self._row_paths(row) if not p.is_file()]
            missing_paths.extend(absent)
            present.append(not absent)
        if not missing_paths:
            return images, 0
        if mode == "fail":
            listed = ", ".join(str(p) for p in missing_paths[:_MAX_LISTED_PATHS])
            raise FileNotFoundError(
                f"{len(missing_paths)} cached file(s) missing, e.g. {listed}; run sync first"
            )
        kept = images[np.array(present, dtype=bool)]
        return kept, len(images) - len(kept)

    def set_global_offset(self, offset: int) -> None:
        """Shift foreground mask ids by ``offset``; background stays 0.

        Raises:
            ValueError: ``offset`` is negative.
        """
        if offset < 0:
            raise ValueError(f"global offset must be non-negative, got {offset}")
        self._global_offset = int(offset)
        self._apply_offset()

    def set_source_vocabulary(
        self,
        source_id2name: dict[int, str],
        source_name2id: dict[str, int],
        num_source_classes: int,
    ) -> None:
        """Replace the source maps with a shared vocabulary and remap mask ids to it.

        Args:
            source_id2name: Source id to label name.
            source_name2id: Inverse of ``source_id2name``.
            num_source_classes: ``len(source_id2name) + 1`` (including background).

        Raises:
            ValueError: A label name this reader serves is not in ``source_name2id``.
        """
        # Derived from the id maps, which stay consistent when a wider vocabulary is applied.
        served = {self.source_id2name[i]: label for i, label in self.source_id2label_id.items()}
        missing = sorted(set(served) - set(source_name2id))
        if missing:
            raise ValueError(f"label names missing from the new vocabulary: {missing}")
        self.source_id2name = dict(source_id2name)
        self.source_name2id = dict(source_name2id)
        self.num_source_classes = int(num_source_classes)
        self.source_id2label_id = {source_name2id[n]: label for n, label in served.items()}
        self._refresh_source_ids()

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

    @property
    def global_offset(self) -> int:
        """The current global offset (default 0)."""
        return self._global_offset

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def _load(self, idx: int) -> tuple[Any, Any]:
        row = self.images.iloc[idx].to_dict()
        with Image.open(self.version.image_path(row)) as handle:
            image = np.asarray(handle.convert("RGB"))
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
