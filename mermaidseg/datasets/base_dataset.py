"""Base dataset for source-space annotated coral reef images.

Subclasses populate a `df_annotations` DataFrame whose `source_label_name`
column holds labels in the **source dataset's own** label space (e.g. CoralNet
provider names, MERMAID benthic attribute names, Coralscapes class names).
Mapping into a unified target space (and optional concept space) is handled
externally by [`mermaidseg.dataset_reconciliation`](../dataset_reconciliation/).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from mermaidseg.datasets.utils import create_annotation_mask, emit_dataset_warning

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id: int) -> None:
    """Configure logging in DataLoader worker processes.

    Pass this as ``worker_init_fn`` to DataLoader when ``num_workers > 0`` so that warnings emitted
    in worker subprocesses are visible.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format=f"[worker-{worker_id}] %(levelname)s %(name)s: %(message)s",
    )


class BaseCoralDataset(Dataset[tuple[torch.Tensor | NDArray[Any], Any]]):
    """A base PyTorch Dataset for loading annotated coral-reef images.

    The dataset emits ``(image, source_labels)`` tuples where ``source_labels``
    is an integer mask in the **dataset's own** label space (0 = background,
    1..N = local source classes). Once a
    :class:`mermaidseg.dataset_reconciliation.SourceLabelRegistry` registers
    the dataset and calls :meth:`set_global_offset`, the emitted mask values
    are shifted into a global integer space jointly indexed across all
    registered datasets.

    Attributes:
        SOURCE_NAME: Identifier for this source dataset (e.g. ``"mermaid"``,
            ``"coralnet"``, ``"coralscapes"``). Used by
            :class:`SourceLabelRegistry` for global ID disambiguation.
        df_annotations: DataFrame with all annotation rows. Must contain
            columns ``image_id``, ``row``, ``col``, ``source_label_name``.
        df_images: DataFrame with one row per image, must contain at least
            ``image_id`` plus any columns required by :meth:`read_image`.
        split: Optional dataset split identifier (e.g. ``"train"``, ``"val"``,
            ``"test"``).
        transform: Optional Albumentations transform applied to image and mask.
        padding: Padding value (in pixels) for point annotations when forming
            the segmentation mask. ``None`` or ``0`` means single-pixel
            annotations.
        class_subset: Optional list of source-label names to retain. When set,
            both ``df_annotations`` and the source label space are filtered.
        source_id2name: Mapping from local source-label IDs (``1..N``) to
            source-label names.
        source_name2id: Inverse of ``source_id2name``.
        num_source_classes: ``len(source_id2name) + 1`` (including background).
    """

    SOURCE_NAME: str = "base"

    df_annotations: pd.DataFrame
    df_images: pd.DataFrame
    split: str | None
    transform: A.BasicTransform | None
    padding: int | None
    class_subset: list[str] | None
    source_id2name: dict[int, str]
    source_name2id: dict[str, int]
    num_source_classes: int
    _global_offset: int
    _load_failures: list[dict[str, Any]]
    _annotation_count_by_image: dict[str, int]
    _annotation_labels_by_image: dict[str, str]

    def __init__(
        self,
        df_annotations: pd.DataFrame,
        df_images: pd.DataFrame,
        split: str | None = None,
        transform: A.BasicTransform | None = None,
        padding: int | None = None,
        class_subset: list[str] | None = None,
    ):
        self.df_annotations = df_annotations
        self.df_images = df_images
        self.split = split
        self.transform = transform
        self.padding = padding
        self.class_subset = class_subset
        self._global_offset = 0

        if "source_label_name" not in self.df_annotations.columns:
            raise ValueError(
                "BaseCoralDataset expects df_annotations to contain a "
                "'source_label_name' column populated with the dataset's "
                "native (source-space) label names."
            )

        if self.class_subset is not None:
            self.df_annotations = self.df_annotations[
                self.df_annotations["source_label_name"].apply(lambda x: x in self.class_subset)
            ].reset_index(drop=True)
            self.df_images = self._derive_df_images_from_annotations(self.df_annotations)

        ordered_names = self.df_annotations["source_label_name"].value_counts().index.tolist()
        self.source_id2name = dict(enumerate(ordered_names, start=1))
        self.source_name2id = {v: k for k, v in self.source_id2name.items()}
        self.num_source_classes = len(self.source_id2name) + 1  # +1 for background

        self._annotation_count_by_image = self.df_annotations["image_id"].value_counts().to_dict()
        self._annotation_labels_by_image = (
            self.df_annotations.groupby("image_id")["source_label_name"]
            .apply(lambda values: ",".join(sorted({str(v) for v in values if pd.notna(v)})))
            .to_dict()
        )
        self._load_failures = []

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        """Re-derive ``df_images`` after filtering ``df_annotations``.

        The default implementation auto-detects column structure for the bundled MERMAID and
        CoralNet shapes. Subclasses are encouraged to override this when they have a fixed schema.
        """
        if "region_id" in df_annotations.columns:
            return (
                df_annotations[["image_id", "region_id", "region_name"]]
                .drop_duplicates(subset=["image_id"])
                .reset_index(drop=True)
            )
        if "source_id" in df_annotations.columns:
            return (
                df_annotations[["source_id", "image_id"]]
                .drop_duplicates(subset=["source_id", "image_id"])
                .reset_index(drop=True)
            )
        raise ValueError(
            "BaseCoralDataset._derive_df_images_from_annotations cannot "
            "auto-detect the df_images schema. Override this method in your "
            "subclass."
        )

    def set_global_offset(self, offset: int) -> None:
        """Set the global source-label offset assigned by the registry.

        After this is called, foreground mask values produced by
        :meth:`__getitem__` are shifted by ``offset`` so they are unique across
        all datasets registered with the same
        :class:`SourceLabelRegistry`.
        """
        if offset < 0:
            raise ValueError(f"global offset must be non-negative, got {offset}")
        self._global_offset = int(offset)

    @property
    def global_offset(self) -> int:
        """Current global offset assigned by the registry (default 0)."""
        return self._global_offset

    def __len__(self) -> int:
        return self.df_images.shape[0]

    def read_image(self, **row_kwargs: Any) -> NDArray[Any]:
        """Read an image given the row metadata.

        Implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        """Return ``(image, source_labels)`` for ``idx``.

        On any internal load/transform error we record the failure, emit a warning to logger +
        stdout + stderr, and return ``(None, None)``. The dataset's :meth:`collate_fn` filters out
        these placeholders, so a failed item drops out of the batch instead of crashing the loader.
        """
        try:
            return self._load_item(idx)
        except Exception as e:
            try:
                image_id = self.df_images.loc[idx, "image_id"]
                row_kwargs = self.df_images.loc[idx].to_dict()
            except Exception:
                image_id = None
                row_kwargs = {}
            self._record_load_failure(image_id=image_id, row_kwargs=row_kwargs, error=e)

            source_info = {k: v for k, v in row_kwargs.items() if k != "image_id"}
            emit_dataset_warning(
                f"{self.__class__.__name__}: skipping idx={idx} image_id={image_id} "
                f"(source={source_info}): {type(e).__name__}: {e}"
            )
            return None, None

    def _load_item(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        """Perform a single load (no error handling).

        Subclasses should override this rather than :meth:`__getitem__` so they inherit the
        recursive-on-failure behaviour for free.
        """
        image_id = self.df_images.loc[idx, "image_id"]
        row_kwargs = self.df_images.loc[idx].to_dict()

        image = self.read_image(**row_kwargs)

        annotations = self.df_annotations.loc[
            self.df_annotations["image_id"] == image_id,
            ["row", "col", "source_label_name"],
        ]

        local_mask = create_annotation_mask(
            annotations, image.shape, self.source_name2id, padding=self.padding
        )

        if self._global_offset:
            local_mask = np.where(
                local_mask > 0, local_mask + self._global_offset, local_mask
            ).astype(local_mask.dtype, copy=False)

        if self.transform:
            transformed = self.transform(image=image, mask=local_mask)
            image = transformed["image"].transpose(2, 0, 1)
            local_mask = transformed["mask"]

        return image, local_mask

    def _record_load_failure(
        self, image_id: Any, row_kwargs: dict[str, Any], error: Exception
    ) -> None:
        record = {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "dataset_class": self.__class__.__name__,
            "split": self.split,
            "image_id": image_id,
            "region_id": row_kwargs.get("region_id"),
            "region_name": row_kwargs.get("region_name"),
            "source_id": row_kwargs.get("source_id"),
            "annotation_count": int(self._annotation_count_by_image.get(image_id, 0)),
            "annotation_labels": self._annotation_labels_by_image.get(image_id, ""),
            "missing_annotations": image_id not in self._annotation_count_by_image,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "annotations_path": getattr(self, "annotations_path", None),
            "source_bucket": getattr(self, "source_bucket", None),
        }
        self._load_failures.append(record)

    def num_load_failures(self) -> int:
        """Return the number of recorded data-loading failures."""
        return len(self._load_failures)

    def load_failures_df(self) -> pd.DataFrame:
        """Return a DataFrame with all recorded data-loading failures."""
        return pd.DataFrame(self._load_failures)

    def save_load_failures(self, output_path: str | Path) -> Path:
        """Save recorded data-loading failures to a parquet report."""
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.load_failures_df().to_parquet(path, index=False)
        return path

    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function that filters out ``(None, None)`` items (failed loads).

        :meth:`__getitem__` returns ``(None, None)`` for items it fails to load (after
        recording the failure and emitting a warning); this filter drops them so a failed
        item simply leaves the batch instead of crashing the loader. If every item in the
        batch failed, empty tensors are returned and the training loop skips the step.

        Args:
            batch: List of ``(image, source_labels)`` tuples possibly containing
                ``(None, None)`` placeholders for items that failed to load.
        Returns:
            A 2-tuple ``(images, source_labels)`` of stacked tensors.
        """
        batch_size = len(batch)
        filtered = [(img, msk) for img, msk in batch if img is not None and msk is not None]
        n_skipped = batch_size - len(filtered)
        if n_skipped > 0:
            logger.warning(
                "collate_fn: skipped %d/%d items in batch due to load errors",
                n_skipped,
                batch_size,
            )

        if len(filtered) == 0:
            logger.warning(
                "collate_fn: entire batch of %d items was empty, returning empty tensors",
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
