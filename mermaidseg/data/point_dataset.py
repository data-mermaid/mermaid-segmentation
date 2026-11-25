"""Reader over a ``points`` version: point annotations rasterised into a mask."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.data.base import VersionDataset
from segdata_spec.read import DatasetVersion

_MAX_LISTED_POINTS = 5
_POSITION = ["image_id", "row", "col"]


def rasterize_points(
    rows: NDArray[np.integer[Any]],
    cols: NDArray[np.integer[Any]],
    local_ids: NDArray[np.integer[Any]],
    shape: tuple[int, int],
    padding: int | None,
    offset: int,
) -> NDArray[np.int32]:
    """Paint points into an int32 mask; background is 0 and foreground ids gain ``offset``.

    A point paints ``[r - padding : r + padding, c - padding : c + padding]``, with the start
    clipped at 0; with no padding it paints one pixel. Later points overwrite earlier ones.

    Args:
        rows: Point rows.
        cols: Point columns.
        local_ids: Local source ids (``1..N``), one per point.
        shape: Mask ``(height, width)``.
        padding: Half-size of the painted square in pixels; ``None`` or 0 paints one pixel.
        offset: Added to every foreground id.

    Returns:
        The ``(height, width)`` mask.
    """
    mask = np.zeros(shape, dtype=np.int32)
    if len(rows) == 0:
        return mask
    rows = rows.astype(np.intp)
    cols = cols.astype(np.intp)
    ids = local_ids.astype(np.int32) + np.int32(offset)
    if padding is not None and padding > 0:
        r0 = np.maximum(0, rows - padding)
        c0 = np.maximum(0, cols - padding)
        for r0i, r1i, c0i, c1i, label in zip(
            r0, rows + padding, c0, cols + padding, ids, strict=True
        ):
            mask[r0i:r1i, c0i:c1i] = label
    else:
        mask[rows, cols] = ids
    return mask


def _select_annotators(points: pd.DataFrame, annotators: Sequence[str] | None) -> pd.DataFrame:
    """Keep, per ``(image_id, row, col)``, the rows of the highest-priority listed annotator."""
    if annotators is None:
        found = points["annotator"].unique()
        if len(found) > 1:
            raise ValueError(
                f"annotators=None needs a single annotator in the version; found {sorted(found)}"
            )
        return points
    priority = points["annotator"].map({name: i for i, name in enumerate(annotators)})
    listed = points[priority.notna()]
    priority = priority[priority.notna()]
    best = priority.groupby([listed[c] for c in _POSITION]).transform("min")
    return listed[priority == best]


def _check_annotators_exist(points: pd.DataFrame, annotators: Sequence[str]) -> None:
    """Raise if any listed annotator is absent from ``points["annotator"]``."""
    unknown = [name for name in dict.fromkeys(annotators) if name not in set(points["annotator"])]
    if unknown:
        raise ValueError(f"annotators not in the dataset version: {unknown}")


def _out_of_bounds(points: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Return the points outside ``[0, height) x [0, width)`` of their image."""
    sizes = images.set_index("image_id")[["height", "width"]]
    height = points["image_id"].map(sizes["height"])
    width = points["image_id"].map(sizes["width"])
    inside = (points["row"] >= 0) & (points["row"] < height)
    inside &= (points["col"] >= 0) & (points["col"] < width)
    return points[~inside]


def select_points(
    points: pd.DataFrame, images: pd.DataFrame, annotators: Sequence[str] | None
) -> pd.DataFrame:
    """Return the points of ``images``, one annotator's labels per point position.

    Args:
        points: Point rows (``image_id, row, col, label_id, annotator``).
        images: Image rows with ``image_id``, ``height`` and ``width``.
        annotators: Annotators in priority order. A point keeps the labels of the first one that
            annotated it, and a point with none of them is dropped. ``None`` requires a single
            annotator in ``points``.

    Returns:
        The selected point rows.

    Raises:
        ValueError: A listed annotator is absent from ``points``, ``annotators`` is ``None`` and
            the selected points have several annotators, or a selected point lies outside its
            image.
    """
    if annotators is not None:
        _check_annotators_exist(points, annotators)
    selected = points[points["image_id"].isin(images["image_id"])]
    selected = _select_annotators(selected, annotators)
    outside = _out_of_bounds(selected, images)
    if not outside.empty:
        listed = outside.head(_MAX_LISTED_POINTS).to_dict("records")
        raise ValueError(f"{len(outside)} point(s) lie outside their image, e.g. {listed}")
    return selected


class PointDataset(VersionDataset):
    """Point-annotation reader; each item's mask is the image's points, painted with ``padding``.

    Images with no selected point are not part of the dataset.

    Attributes:
        points: The selected point rows (``image_id, row, col, label_id, annotator``), only
            those of the images in ``images``.
        padding: Half-size in pixels of the square painted around each point; ``None`` or 0
            paints one pixel.
    """

    KIND: ClassVar[Literal["points", "dense"]] = "points"

    def __init__(
        self,
        version: DatasetVersion,
        *,
        split: Literal["train", "val", "test"],
        groups: Collection[str] | None = None,
        transform: Callable[..., dict[str, Any]] | None = None,
        on_missing_images: Literal["fail", "drop"] = "fail",
        image_ids: Collection[str] | None = None,
        annotators: Sequence[str] | None = None,
        padding: int | None = None,
    ) -> None:
        """Select images and points; see :class:`VersionDataset` for the shared arguments.

        Args:
            version: The opened version.
            split: The split; only ``"train"`` resamples a sample whose mask is empty.
            groups: Image groups to keep; ``None`` keeps all.
            transform: Called as ``transform(image=, mask=)`` and returning a dict with both.
            on_missing_images: ``"fail"`` or ``"drop"`` images whose files are not cached.
            image_ids: Keeps only these images of the selection; ``None`` keeps all.
            annotators: Annotators in priority order. A point keeps the labels of the first one
                that annotated it, and a point with none of them is dropped. ``None`` requires a
                single annotator in the version.
            padding: Half-size in pixels of the square painted around each point.

        Raises:
            ValueError: ``annotators`` is ``None`` and the version has several annotators, or a
                point lies outside its image.
        """
        self.padding = padding
        self._annotators = annotators
        self.points = version.points.to_pandas() if version.points is not None else pd.DataFrame()
        super().__init__(
            version,
            split=split,
            groups=groups,
            transform=transform,
            on_missing_images=on_missing_images,
            image_ids=image_ids,
        )
        # ``drop`` mode removes images after ``_filter_images``, so their points go here.
        self.points = self.points[self.points["image_id"].isin(self.images["image_id"])]
        points = self.points
        self._rows = points["row"].to_numpy(dtype=np.intp)
        self._cols = points["col"].to_numpy(dtype=np.intp)
        self._positions = points.reset_index(drop=True).groupby("image_id", sort=False).indices
        self._refresh_source_ids()

    def _refresh_source_ids(self) -> None:
        label_to_local = {label: i for i, label in self.source_id2label_id.items()}
        self._local_ids = self.points["label_id"].map(label_to_local).to_numpy(dtype=np.int32)

    def _filter_images(self, images: pd.DataFrame) -> pd.DataFrame:
        self.points = select_points(self.points, images, self._annotators)
        return images[images["image_id"].isin(self.points["image_id"])]

    def _label_counts(self) -> pd.Series:
        selected = self.points[self.points["image_id"].isin(self.images["image_id"])]
        return selected.groupby("label_id").size()

    def _read_mask(self, row: Mapping[str, Any], image_shape: tuple[int, int]) -> NDArray[np.int32]:
        positions = self._positions[row["image_id"]]
        return rasterize_points(
            self._rows[positions],
            self._cols[positions],
            self._local_ids[positions],
            image_shape,
            self.padding,
            self._global_offset,
        )
