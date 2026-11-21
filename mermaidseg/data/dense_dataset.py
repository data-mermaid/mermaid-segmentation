"""Reader over a ``dense`` version: mask PNGs of native label ids mapped to local ids."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from mermaidseg.data.base import VersionDataset
from segdata_spec.read import DatasetVersion


def build_lut(
    source_id2label_id: Mapping[int, int], max_label_id: int, offset: int
) -> NDArray[np.int32]:
    """Return an int32 table mapping a native ``label_id`` to ``local id + offset``.

    Entries outside the vocabulary, and 0, map to 0.
    """
    lut = np.zeros(max_label_id + 1, dtype=np.int32)
    for local_id, label_id in source_id2label_id.items():
        lut[label_id] = local_id + offset
    return lut


class DenseDataset(VersionDataset):
    """Dense-mask reader; the mask PNG holds native ``label_id`` values (0 is background).

    Raises:
        ValueError: ``label_id`` 0 appears in the selected label counts.
    """

    KIND: ClassVar[Literal["points", "dense"]] = "dense"

    def __init__(
        self,
        version: DatasetVersion,
        *,
        split: Literal["train", "val", "test"],
        groups: Collection[str] | None = None,
        transform: Callable[..., dict[str, Any]] | None = None,
        on_missing_images: Literal["fail", "drop"] = "fail",
    ) -> None:
        """Select images and build the label lookup; arguments as in :class:`VersionDataset`."""
        self._max_label_id = int(max(version.labels.column("label_id").to_pylist(), default=0))
        super().__init__(
            version,
            split=split,
            groups=groups,
            transform=transform,
            on_missing_images=on_missing_images,
        )
        if 0 in self.source_id2label_id.values():
            raise ValueError(
                "label_id 0 is counted in the selected images, but mask value 0 is unlabeled"
            )

    def _apply_offset(self) -> None:
        self._lut = build_lut(self.source_id2label_id, self._max_label_id, self._global_offset)

    def _label_counts(self) -> pd.Series:
        counts = self.version.label_counts.to_pandas()
        selected = counts[counts["image_id"].isin(self.images["image_id"])]
        return selected.groupby("label_id")["n"].sum()

    def _row_paths(self, row: Mapping[str, Any]) -> list[Path]:
        return [self.version.image_path(row), self.version.mask_path(row)]

    def _read_mask(self, row: Mapping[str, Any], image_shape: tuple[int, int]) -> NDArray[np.int32]:
        with Image.open(self.version.mask_path(row)) as handle:
            native = np.asarray(handle)
        if native.ndim == 3:
            native = native[..., 0]
        if native.size and int(native.max()) >= len(self._lut):
            raise ValueError(
                f"mask of image {row['image_id']} has label {int(native.max())}, "
                f"above the largest label_id {len(self._lut) - 1}"
            )
        return self._lut[native]
