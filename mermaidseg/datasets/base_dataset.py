"""Base dataset for source-space annotated coral reef images.

Subclasses populate a `df_annotations` DataFrame whose `source_label_name`
column holds labels in the **source dataset's own** label space (e.g. CoralNet
provider names, MERMAID benthic attribute names, Coralscapes class names).
Mapping into a unified target space (and optional concept space) is handled
externally by [`mermaidseg.dataset_reconciliation`](../dataset_reconciliation/).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import albumentations as A
import boto3
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from mermaidseg.datasets.utils import (
    create_annotation_mask_from_arrays,
    emit_dataset_warning,
    s3_training_config,
)

try:  # resource is Unix-only; the RSS diagnostic is optional and silently skipped without it
    import resource
except ImportError:  # pragma: no cover - non-Unix platforms
    resource = None

logger = logging.getLogger(__name__)

# Optional per-worker RSS logging for OOM investigations. Off unless MERMAIDSEG_LOG_WORKER_RSS
# is truthy. Read at import (each worker process re-evaluates it), consistent with the cache-dir
# env handling in utils.py.
_LOG_WORKER_RSS = os.environ.get("MERMAIDSEG_LOG_WORKER_RSS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_RSS_LOG_INTERVAL = 500
_rss_getitem_count = 0

# Cap on the number of full load-failure *records* retained per dataset. The failure *count*
# (`num_load_failures`) is always exact; only the rich per-failure dicts are bounded, so a
# systemic failure (which would otherwise append a dict every sample for the whole run) can't grow
# `_load_failures` without bound in each forked worker. 1000 records is plenty to diagnose a
# pattern; the count still drives the per-epoch failure-rate guard in model/train.py.
_LOAD_FAILURE_RECORD_CAP = 1000


def _maybe_log_worker_rss() -> None:
    """Periodically log this process's peak RSS when ``MERMAIDSEG_LOG_WORKER_RSS`` is
    set.

    Diagnostic for OOM investigations: SageMaker samples instance memory only once a minute, so
    a sub-minute spike (many large images decoded across workers at once) is invisible there.
    This surfaces per-worker memory from inside the load loop, every ``_RSS_LOG_INTERVAL`` items.
    """
    global _rss_getitem_count
    if not _LOG_WORKER_RSS or resource is None:
        return
    _rss_getitem_count += 1
    if _rss_getitem_count % _RSS_LOG_INTERVAL:
        return
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    worker = torch.utils.data.get_worker_info()
    wid = worker.id if worker is not None else "main"
    # ru_maxrss is bytes on macOS, kilobytes on Linux — report both readings.
    emit_dataset_warning(
        f"[rss] worker={wid} items={_rss_getitem_count} ru_maxrss={ru} "
        f"(~{ru / 1024:.0f}MB-if-KB / ~{ru / 1048576:.0f}MB-if-bytes)"
    )


def _reinit_s3_clients(dataset: object) -> None:
    """Drop any fork-inherited S3 client so each worker builds its own lazily.

    Walks ConcatDataset wrappers to reach each underlying BaseCoralDataset. Resetting to
    ``None`` (rather than eagerly constructing a client here) keeps creation lazy and
    per-process — the fresh client is built on first ``.s3`` access inside the worker.
    We gate on ``isinstance`` rather than ``hasattr(dataset, "s3")`` because ``s3`` is
    now a property: ``hasattr`` would invoke its getter and create a client in the wrong
    place.
    """
    if isinstance(dataset, BaseCoralDataset):
        dataset._s3 = None
    for child in getattr(dataset, "datasets", []):
        _reinit_s3_clients(child)


def worker_init_fn(worker_id: int) -> None:
    """Configure logging and reinitialize S3 clients in forked DataLoader workers."""
    logging.basicConfig(
        level=logging.WARNING,
        format=f"[worker-{worker_id}] %(levelname)s %(name)s: %(message)s",
    )
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        _reinit_s3_clients(worker_info.dataset)


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
    # Contiguous numpy annotation index (aligned to df_images positional idx), replacing the old
    # str-keyed dict + per-sample DataFrame slice on the hot path. None for datasets without
    # point-annotation row/col columns (e.g. dense-mask subclasses that override _load_item).
    _ann_offsets: np.ndarray | None
    _ann_row: np.ndarray | None
    _ann_col: np.ndarray | None
    _ann_label_id: np.ndarray | None

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
        # Lazily-created, per-process S3 client (see the ``s3`` property). Never construct it
        # here: a boto3 client is unpicklable and not fork-safe, so it must not be carried across
        # a DataLoader worker boundary. ``setdefault`` avoids clobbering a client a subclass or
        # test injected via ``self.s3 = ...`` before calling ``super().__init__()``.
        self.__dict__.setdefault("_s3", None)

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
        grouped_by_image = self.df_annotations.groupby("image_id")
        self._annotation_labels_by_image = (
            grouped_by_image["source_label_name"]
            .apply(lambda values: ",".join(sorted({str(v) for v in values if pd.notna(v)})))
            .to_dict()
        )

        # Precompute a numpy-native, per-image annotation index so __getitem__ never touches the
        # multi-million-row DataFrame or a str-keyed Python dict on the hot path. Under forked
        # DataLoader workers, per-sample access to pandas object columns / str-keyed dicts writes
        # CPython refcounts into their header pages, which copy-on-write into each worker's RSS and
        # accumulate for the whole run under persistent_workers (the dinov3-lora-qv-r8 OOM; see
        # scripts/diagnostics/dataloader_rss_findings.md). Contiguous numpy arrays carry no
        # per-element Python objects, so slicing them dirties nothing shared.
        self._ann_offsets = None
        self._ann_row = None
        self._ann_col = None
        self._ann_label_id = None
        if {"row", "col", "source_label_name"}.issubset(self.df_annotations.columns):
            self._build_annotation_index()

        self._load_failures = []
        self._load_failure_count = 0

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        """Re-derive ``df_images`` after filtering ``df_annotations``.

        The default implementation auto-detects column structure for the bundled MERMAID
        and CoralNet shapes. Subclasses are encouraged to override this when they have a
        fixed schema.
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

    def _build_annotation_index(self) -> None:
        """Precompute the contiguous numpy annotation index used by :meth:`_load_item`.

        Reorders all point annotations so image ``idx`` (positional in ``df_images``)
        owns the contiguous slice ``[_ann_offsets[idx]:_ann_offsets[idx + 1]]`` of
        ``_ann_row`` / ``_ann_col`` / ``_ann_label_id``. Built once in the main process;
        the per-sample hot path then touches only these numpy arrays (no pandas object
        columns, no str-keyed dict), keeping forked workers copy-on-write-clean. Images
        with no annotations get an empty slice. Annotations whose label is unmapped/NaN
        or whose image is absent from ``df_images`` are dropped (mirrors
        :func:`create_annotation_mask`'s unknown-label handling).
        """
        num_images = len(self.df_images)
        # image_id -> positional index in df_images. Built once here (main process); never touched
        # per sample. Positional (df_images order) so it matches _load_item's ``.iloc[idx]``.
        id_to_pos = {img_id: i for i, img_id in enumerate(self.df_images["image_id"].to_numpy())}

        ann = self.df_annotations
        label_id_raw = ann["source_label_name"].map(self.source_name2id).to_numpy()
        img_pos_raw = ann["image_id"].map(id_to_pos).to_numpy()
        keep = ~pd.isna(label_id_raw) & ~pd.isna(img_pos_raw)

        img_pos = img_pos_raw[keep].astype(np.int64)
        rows = ann["row"].to_numpy()[keep]
        cols = ann["col"].to_numpy()[keep]
        label_ids = label_id_raw[keep]

        # Stable sort groups each image's annotations contiguously while preserving their order.
        order = np.argsort(img_pos, kind="stable")
        self._ann_row = np.ascontiguousarray(rows[order], dtype=np.intp)
        self._ann_col = np.ascontiguousarray(cols[order], dtype=np.intp)
        self._ann_label_id = np.ascontiguousarray(label_ids[order], dtype=np.int64)

        counts = np.bincount(img_pos, minlength=num_images)
        offsets = np.zeros(num_images + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        self._ann_offsets = offsets

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

    @property
    def s3(self) -> boto3.client:
        """Lazily-created, per-process boto3 S3 client.

        A boto3 client is neither picklable nor fork-safe, so it must never be created
        at construction time and carried across a process boundary. Building it on first
        access means each process — the main process, or a spawned/forked DataLoader
        worker — gets its own. :meth:`__getstate__` drops it before pickling (``spawn``
        workers) and :func:`_reinit_s3_clients` resets it in ``worker_init_fn``
        (``fork`` workers), so a client is never shared across the boundary.
        """
        if getattr(self, "_s3", None) is None:
            self._s3 = boto3.client("s3", config=s3_training_config())
        return self._s3

    @s3.setter
    def s3(self, client: Any) -> None:
        # Retained so tests/helpers can inject a fake client via ``dataset.s3 = ...``.
        self._s3 = client

    def __getstate__(self) -> dict[str, Any]:
        """Drop the unpicklable S3 client so the dataset can be sent to a ``spawn``
        worker.

        The worker recreates it lazily on first :attr:`s3` access.
        """
        state = self.__dict__.copy()
        state["_s3"] = None
        return state

    def __len__(self) -> int:
        return self.df_images.shape[0]

    def read_image(self, **row_kwargs: Any) -> NDArray[Any]:
        """Read an image given the row metadata.

        Implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        """Return ``(image, source_labels)`` for ``idx``.

        On any internal load/transform error we record the failure, emit a warning to
        logger + stdout + stderr, and return ``(None, None)``. The dataset's
        :meth:`collate_fn` filters out these placeholders, so a failed item drops out of
        the batch instead of crashing the loader.
        """
        _maybe_log_worker_rss()
        try:
            return self._load_item(idx)
        except Exception as e:
            try:
                row_kwargs = self.df_images.iloc[idx].to_dict()
                image_id = row_kwargs.get("image_id")
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

        Subclasses should override this rather than :meth:`__getitem__` so they inherit
        the recursive-on-failure behaviour for free.
        """
        row_kwargs = self.df_images.iloc[idx].to_dict()

        image = self.read_image(**row_kwargs)

        # Slice the precomputed contiguous numpy arrays for this image — no DataFrame / str-dict
        # access on the hot path (keeps forked workers copy-on-write-clean; see __init__).
        start = int(self._ann_offsets[idx])
        end = int(self._ann_offsets[idx + 1])
        local_mask = create_annotation_mask_from_arrays(
            self._ann_row[start:end],
            self._ann_col[start:end],
            self._ann_label_id[start:end],
            image.shape,
            padding=self.padding,
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
        # Always count (the failure-rate guard in model/train.py reads num_load_failures()); only
        # the rich record list is capped, so a systemic failure can't grow RSS without bound in
        # each forked worker (see _LOAD_FAILURE_RECORD_CAP).
        self._load_failure_count += 1
        if len(self._load_failures) >= _LOAD_FAILURE_RECORD_CAP:
            return
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
        """Return the exact cumulative count of data-loading failures.

        This is the count, not ``len(load_failures_df())`` — the stored records are capped at
        ``_LOAD_FAILURE_RECORD_CAP`` but the count is always exact, so the per-epoch failure-rate
        guard stays correct even past the cap.

        Known limitation: failures recorded inside forked DataLoader workers do not propagate back
        to the main-process dataset copy, so with ``num_workers>0`` this under-counts (a
        pre-existing issue, tracked separately — see the plan's B3 note).
        """
        return self._load_failure_count

    def load_failures_df(self) -> pd.DataFrame:
        """Return a DataFrame of retained load-failure records.

        A *sample* (the first ``_LOAD_FAILURE_RECORD_CAP`` records) when
        :meth:`num_load_failures` exceeds the cap.
        """
        return pd.DataFrame(self._load_failures)

    def save_load_failures(self, output_path: str | Path) -> Path:
        """Save recorded data-loading failures to a parquet report."""
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.load_failures_df().to_parquet(path, index=False)
        return path

    @staticmethod
    def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
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
