"""Reader over a ``dense`` version: mask PNGs of native label ids shifted by the global offset."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from mermaidseg.data.base import VersionDataset
from mermaidseg.data.dense_read import read_label_mask
from segdata_spec.read import DatasetVersion


def build_lut(label_ids: Iterable[int], max_label_id: int, offset: int) -> NDArray[np.int32]:
    """Return an int32 table mapping each native ``label_id`` in ``label_ids`` to ``offset + id``.

    0, and an id up to ``max_label_id`` that is not in ``label_ids``, map to 0.
    """
    lut = np.zeros(max_label_id + 1, dtype=np.int32)
    for label_id in label_ids:
        lut[label_id] = offset + label_id
    return lut


class DenseDataset(VersionDataset):
    """Dense-mask reader; the mask PNG holds native ``label_id`` values (0 is background)."""

    KIND: ClassVar[Literal["points", "dense"]] = "dense"

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
        """Select images and build the label lookup; arguments as in :class:`VersionDataset`."""
        super().__init__(
            version,
            split=split,
            global_offset=global_offset,
            groups=groups,
            transform=transform,
            image_ids=image_ids,
        )
        label_ids = version.labels.column("label_id").to_pylist()
        self._lut = build_lut(label_ids, int(max(label_ids, default=0)), global_offset)

    def _row_paths(self, row: Mapping[str, Any]) -> list[Path]:
        return [self.version.image_path(row), self.version.mask_path(row)]

    def _read_mask(self, row: Mapping[str, Any], image_shape: tuple[int, int]) -> NDArray[np.int32]:
        native = read_label_mask(self.version.mask_path(row), len(self._lut))
        return self._lut[native]
