"""One-off comparison of the old point loaders with a synced point version."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mermaidseg.data.config import DATASET_BUCKET
from mermaidseg.data.point_dataset import select_points
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
from mermaidseg.datasets.factory import build_datasets
from segdata_spec.manifest import Pin
from segdata_spec.read import DatasetVersion, open_version
from segdata_spec.sync import sync

# Val/test transform size for the old loader; only its annotations are read here.
_UNUSED_INPUT_SIZE = (256, 256)
_COLUMNS = ["image_id", "row", "col", "native_name", "mapped"]
_MAX_LISTED = 10

Point = tuple[str, int, int, str | None]
CountDiff = dict[str | None, tuple[int, int]]


@dataclass(frozen=True)
class ParityReport:
    """The differences between the old and the new points; empty fields mean parity.

    Attributes:
        old_only_images: Image ids with old points and none on the new side.
        new_only_images: Image ids with new points, in the compared groups, and none on the old
            side.
        old_only_points: ``(image_id, row, col, mapped)`` the old side has more often.
        new_only_points: ``(image_id, row, col, mapped)`` the new side has more often.
        native_diff: Native label -> ``(old count, new count)`` where the counts differ.
        mapped_diff: Mapped label -> ``(old count, new count)`` where the counts differ.
    """

    old_only_images: list[str] = field(default_factory=list)
    new_only_images: list[str] = field(default_factory=list)
    old_only_points: Counter[Point] = field(default_factory=Counter)
    new_only_points: Counter[Point] = field(default_factory=Counter)
    native_diff: CountDiff = field(default_factory=dict)
    mapped_diff: CountDiff = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no field reports a difference."""
        return not (
            self.old_only_images
            or self.new_only_images
            or self.old_only_points
            or self.new_only_points
            or self.native_diff
            or self.mapped_diff
        )


def _mapped_names(registry: SourceLabelRegistry, ds: Any) -> dict[str, str | None]:
    """Map each source label name of ``ds`` to its target name, ``None`` when unmapped."""
    offset = registry.dataset_offsets[ds.SOURCE_NAME]
    result: dict[str, str | None] = {}
    for name, local_id in ds.source_name2id.items():
        target_id = int(registry.source_to_target[local_id + offset])
        result[name] = registry.target_id2label[target_id] if target_id else None
    return result


def old_points(ds: Any) -> pd.DataFrame:
    """Return the points of an old point dataset with their mapped target label.

    Args:
        ds: An old ``BaseCoralDataset``; registering it sets its global offset.

    Returns:
        Columns ``image_id`` (str), ``row``, ``col``, ``native_name`` and ``mapped``.
    """
    names = _mapped_names(SourceLabelRegistry([ds]), ds)
    annotations = ds.df_annotations
    native = annotations["source_label_name"]
    mapped = native.map(names).astype(object)
    return pd.DataFrame(
        {
            "image_id": annotations["image_id"].astype(str),
            "row": annotations["row"],
            "col": annotations["col"],
            "native_name": native,
            "mapped": mapped.where(mapped.notna(), None),
        }
    ).reset_index(drop=True)[_COLUMNS]


def new_points(version: DatasetVersion, points: pd.DataFrame) -> pd.DataFrame:
    """Return ``points`` of ``version`` with the same columns as :func:`old_points`."""
    labels = version.labels.to_pandas().set_index("label_id")
    attribute = points["label_id"].map(labels["mermaid_benthic_attribute"])
    lowered = attribute.str.lower().astype(object)
    return pd.DataFrame(
        {
            "image_id": points["image_id"].astype(str),
            "row": points["row"],
            "col": points["col"],
            "native_name": points["label_id"].map(labels["name"]),
            "mapped": lowered.where(attribute.notna(), None),
        }
    ).reset_index(drop=True)[_COLUMNS]


def _none_if_na(value: Any) -> Any:
    return None if pd.isna(value) else value


def _point_counter(points: pd.DataFrame) -> Counter[Point]:
    keys = zip(
        points["image_id"],
        points["row"].astype(int),
        points["col"].astype(int),
        (_none_if_na(m) for m in points["mapped"]),
        strict=True,
    )
    return Counter(keys)


def _label_counts(points: pd.DataFrame, column: str) -> Counter[str | None]:
    return Counter(_none_if_na(v) for v in points[column])


def _count_diff(old: Counter[str | None], new: Counter[str | None]) -> CountDiff:
    labels = sorted(old.keys() | new.keys(), key=lambda x: (x is None, x or ""))
    return {label: (old[label], new[label]) for label in labels if old[label] != new[label]}


def _restrict_to_old_groups(
    old: pd.DataFrame, new: pd.DataFrame, new_groups: Mapping[str, str]
) -> pd.DataFrame:
    """Keep the new points of the groups that hold at least one old image."""
    groups = {new_groups[i] for i in set(old["image_id"]) if i in new_groups}
    return new[new["image_id"].map(new_groups).isin(groups)]


def compare(old: pd.DataFrame, new: pd.DataFrame, new_groups: Mapping[str, str]) -> ParityReport:
    """Compare old and new points over the groups of the new version that hold old images.

    Args:
        old: Frame from :func:`old_points`.
        new: Frame from :func:`new_points`.
        new_groups: ``image_id -> group`` of the new version's images.

    Returns:
        The report; ``ok`` when the points agree as multisets.
    """
    new = _restrict_to_old_groups(old, new, new_groups)
    old_images, new_images = set(old["image_id"]), set(new["image_id"])
    old_counter, new_counter = _point_counter(old), _point_counter(new)
    return ParityReport(
        old_only_images=sorted(old_images - new_images),
        new_only_images=sorted(new_images - old_images),
        old_only_points=old_counter - new_counter,
        new_only_points=new_counter - old_counter,
        native_diff=_count_diff(
            _label_counts(old, "native_name"), _label_counts(new, "native_name")
        ),
        mapped_diff=_count_diff(_label_counts(old, "mapped"), _label_counts(new, "mapped")),
    )


def _format_list(title: str, items: Collection[Any]) -> list[str]:
    if not items:
        return [f"{title}: none"]
    shown = ", ".join(str(i) for i in list(items)[:_MAX_LISTED])
    more = f" (+{len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else ""
    return [f"{title}: {len(items)}", f"  {shown}{more}"]


def _format_diff(title: str, diff: CountDiff) -> list[str]:
    if not diff:
        return [f"{title}: none"]
    return [f"{title}:", *(f"  {label}: old {o}, new {n}" for label, (o, n) in diff.items())]


def format_report(report: ParityReport) -> str:
    """Render a report as text, ending with ``PARITY OK`` or ``PARITY FAILED``."""
    lines = [
        *_format_list("old-only images", report.old_only_images),
        *_format_list("new-only images", report.new_only_images),
        *_format_list("points only in old", sorted(report.old_only_points.elements(), key=str)),
        *_format_list("points only in new", sorted(report.new_only_points.elements(), key=str)),
        *_format_diff("native label counts", report.native_diff),
        *_format_diff("mapped label counts", report.mapped_diff),
        "PARITY OK" if report.ok else "PARITY FAILED",
    ]
    return "\n".join(lines)


def run_parity(
    data_cfg: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    version: str,
    manifest_sha256: str,
    cache_dir: Path,
    client: Any,
    annotators: Sequence[str] | None,
) -> ParityReport:
    """Build the old dataset, sync the pinned version's tables, and compare their points.

    No image is downloaded: only the parquet tables are synced.

    Args:
        data_cfg: The ``data`` section of the old data config.
        dataset: Dataset name.
        split: Split to build on the old side; the new side compares every split.
        version: Version to compare against.
        manifest_sha256: The sha256 of that version's manifest.
        cache_dir: The local cache root.
        client: An S3 client.
        annotators: Annotators in priority order for selecting the new points.

    Returns:
        The parity report over all groups of the new version that hold old images.
    """
    built = build_datasets(
        data_cfg, input_size=_UNUSED_INPUT_SIZE, names=[dataset], splits=[split], verbose=False
    )
    old = old_points(built[(dataset, split)])
    pin = Pin(name=dataset, version=version, manifest_sha256=manifest_sha256)
    sync(
        client,
        {dataset: pin},
        cache_dir,
        bucket=DATASET_BUCKET,
        selection={dataset: set()},
    )
    version_data = open_version(cache_dir, DATASET_BUCKET, pin)
    if version_data.points is None:
        raise ValueError(f"{dataset} {version} has no points table")
    images = version_data.images.to_pandas()
    points = select_points(version_data.points.to_pandas(), images, annotators)
    kept = images[images["image_id"].isin(points["image_id"])]
    new_groups = dict(zip(kept["image_id"], kept["group"], strict=True))
    return compare(old, new_points(version_data, points), new_groups)
