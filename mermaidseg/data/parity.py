"""One-off comparison of the old loaders with a synced (or locally staged) dataset version."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from numpy.typing import NDArray
from PIL import Image

from mermaidseg.data.config import DATASET_BUCKET
from mermaidseg.data.point_dataset import select_points
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME
from mermaidseg.datasets.factory import build_datasets
from mermaidseg.io import ConfigDict, preprocess_data_config
from segdata_spec.manifest import Pin
from segdata_spec.read import DatasetVersion, open_version
from segdata_spec.sync import sync as sync_versions

# Val/test transform size for the old loader; only its annotations are read here.
_UNUSED_INPUT_SIZE = (256, 256)
_COLUMNS = ["image_id", "row", "col", "native_name", "mapped"]
_MAX_LISTED = 10

Point = tuple[str, int, int, str | None]
CountDiff = dict[str | None, tuple[int, int]]
LabelCountDiff = dict[int, tuple[int, int]]


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


def _build_old(data_cfg: Mapping[str, Any], dataset: str, split: str) -> Any:
    """Build one old dataset from the raw ``data`` section, expanded as training expands it."""
    expanded = preprocess_data_config(ConfigDict({"data": data_cfg})).data
    return build_datasets(
        expanded, input_size=_UNUSED_INPUT_SIZE, names=[dataset], splits=[split], verbose=False
    )


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


def _format_diff(title: str, diff: Mapping[Any, tuple[int, int]]) -> list[str]:
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


def sync_tables(client: Any, pin: Pin, cache_dir: Path) -> None:
    """Download and verify the parquet tables of ``pin`` into ``cache_dir``, and no image."""
    sync_versions(
        client,
        {pin.name: pin},
        cache_dir,
        bucket=DATASET_BUCKET,
        selection={pin.name: set()},
    )


def version_kind(cache_dir: Path, pin: Pin) -> str:
    """Return the ``kind`` of the cached version, re-verifying its manifest against ``pin``."""
    return open_version(cache_dir, DATASET_BUCKET, pin).manifest.kind


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
    sync: bool = True,
) -> ParityReport:
    """Build the old dataset, open the pinned version's tables, and compare their points.

    No image is downloaded: only the parquet tables are synced.

    Args:
        data_cfg: The ``data`` section of the old data config.
        dataset: Dataset name.
        split: Split to build on the old side; the new side compares every split.
        version: Version to compare against.
        manifest_sha256: The sha256 of that version's manifest.
        cache_dir: The local cache root.
        client: An S3 client; unused with ``sync=False``.
        annotators: Annotators in priority order for selecting the new points.
        sync: When False, S3 is not reached and the version is read from ``cache_dir`` as staged.

    Returns:
        The parity report over all groups of the new version that hold old images.
    """
    built = _build_old(data_cfg, dataset, split)
    old = old_points(built[(dataset, split)])
    pin = Pin(name=dataset, version=version, manifest_sha256=manifest_sha256)
    if sync:
        sync_tables(client, pin, cache_dir)
    version_data = open_version(cache_dir, DATASET_BUCKET, pin)
    if version_data.points is None:
        raise ValueError(f"{dataset} {version} has no points table")
    images = version_data.images.to_pandas()
    points = select_points(version_data.points.to_pandas(), images, annotators)
    kept = images[images["image_id"].isin(points["image_id"])]
    new_groups = dict(zip(kept["image_id"], kept["group"], strict=True))
    return compare(old, new_points(version_data, points), new_groups)


# --- dense ---------------------------------------------------------------------------------

IdImageMask = tuple[str, NDArray[Any], NDArray[Any]]


@dataclass(frozen=True)
class VocabMismatch:
    """One native label id whose old and new name or MERMAID mapping differ (``None`` if absent)."""

    label_id: int
    old_name: str | None
    new_name: str | None
    old_mapped: str | None
    new_mapped: str | None


@dataclass(frozen=True)
class DenseParityReport:
    """The differences between an old dense dataset and a dense version; empty means parity.

    Attributes:
        matched: Images present on both sides, matched by the sha256 of their RGB pixels.
        old_only: Old row ids with no new image.
        new_only: New image ids with no old row.
        mask_diffs: New image id -> differing native mask pixels, for matched images whose masks
            differ; a shape mismatch counts every pixel of the larger mask.
        label_pixel_diff: Native label id -> ``(old pixels, new pixels)`` where the totals differ.
        vocab_mismatches: Label ids whose name or MERMAID mapping differ.
        hf_revision_at_run: Head revision of the unpinned Hugging Face source, if it was reachable.
    """

    matched: int = 0
    old_only: list[str] = field(default_factory=list)
    new_only: list[str] = field(default_factory=list)
    mask_diffs: dict[str, int] = field(default_factory=dict)
    label_pixel_diff: LabelCountDiff = field(default_factory=dict)
    vocab_mismatches: list[VocabMismatch] = field(default_factory=list)
    hf_revision_at_run: str | None = None

    @property
    def ok(self) -> bool:
        """True when no field reports a difference."""
        return not (
            self.old_only
            or self.new_only
            or self.mask_diffs
            or self.label_pixel_diff
            or self.vocab_mismatches
        )


def _no_revision() -> str | None:
    return None


def _rgb_array(image: Any) -> NDArray[Any]:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    return np.asarray(image)


def _native_mask(mask: Any) -> NDArray[Any]:
    array = np.asarray(mask)
    return array[..., 0] if array.ndim == 3 else array


def _pixel_sha(array: NDArray[Any]) -> str:
    digest = sha256(repr(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _add_label_pixels(counter: Counter[int], mask: NDArray[Any]) -> None:
    values, counts = np.unique(mask, return_counts=True)
    counter.update(dict(zip(values.tolist(), counts.tolist(), strict=True)))


def _differing_pixels(old: NDArray[Any], new: NDArray[Any]) -> int:
    if old.shape != new.shape:
        return max(old.size, new.size)
    return int((old != new).sum())


def old_groups(ds: Any) -> set[str] | None:
    """Return the new-version groups an old dense dataset covers, ``None`` for every group.

    UCSD mosaics load one HF split per site, which is the group; Coralscapes' ``split`` is.
    """
    sites = getattr(ds, "sites", None)
    if sites is not None:
        return set(sites)
    if ds.split is None:
        return None
    return {ds.split} if isinstance(ds.split, str) else set(ds.split)


_HF_SPLITS = ("train", "validation", "test")


def _loaded_splits(ds: Any) -> list[str]:
    """Return the HF splits an old Coralscapes dataset concatenated, in order."""
    if ds.split is None:
        return list(_HF_SPLITS)
    return [ds.split] if isinstance(ds.split, str) else list(ds.split)


def _split_lengths(ds: Any, splits: Sequence[str]) -> list[int]:
    """Return the row count of each loaded split; the old dataset keeps only the concatenation."""
    if len(splits) == 1:
        return [len(ds.dataset)]
    hf_dataset = load_dataset(ds.hf_repo_id)
    return [len(hf_dataset[name]) for name in splits]


def _positional_id(index: int, splits: Sequence[str], lengths: Sequence[int]) -> str:
    """Name row ``index`` of the concatenation by its split and index within that split."""
    for name, length in zip(splits, lengths, strict=True):
        if index < length:
            return f"{name} row {index}"
        index -= length
    raise IndexError("row index beyond the loaded splits")


def old_dense_rows(ds: Any) -> Iterator[IdImageMask]:
    """Yield ``(row id, native RGB image, native mask)`` from the HF rows, before any transform.

    Rows without a ``filename`` are named ``"<split> row <index within the split>"``.
    """
    boundaries: tuple[list[str], list[int]] | None = None
    for index, row in enumerate(ds.dataset):
        row_id = row.get("filename")
        if not row_id:
            if boundaries is None:
                splits = _loaded_splits(ds)
                boundaries = (splits, _split_lengths(ds, splits))
            row_id = _positional_id(index, *boundaries)
        yield str(row_id), _rgb_array(row["image"]), _native_mask(row["label"])


def _native_id2name(ds: Any) -> dict[int, str]:
    """Return the native mask id -> class name of an old dense dataset (UCSD or Coralscapes)."""
    table = getattr(ds, "class_table", None)
    if table is not None:
        return {int(entry["id"]): entry["name"] for entry in table}
    return dict(CORALSCAPES_V2_ID2NAME)


def vocabulary_mismatches(
    old_names: Mapping[int, str],
    old_mapped: Mapping[str, str | None],
    labels: pd.DataFrame,
) -> list[VocabMismatch]:
    """List the label ids where the old and new name, or lowercased MERMAID mapping, differ.

    Args:
        old_names: Native id -> old class name.
        old_mapped: Old class name -> mapped target name, lowercase, ``None`` when unmapped.
        labels: ``labels.parquet`` as a frame with ``label_id``, ``name`` and
            ``mermaid_benthic_attribute``.
    """
    new = labels.set_index("label_id")
    mismatches = []
    for label_id in sorted(old_names.keys() | set(new.index)):
        old_name = old_names.get(label_id)
        new_name = _none_if_na(new["name"].get(label_id))
        attribute = _none_if_na(new["mermaid_benthic_attribute"].get(label_id))
        new_mapped = None if attribute is None else attribute.lower()
        old_target = old_mapped.get(old_name) if old_name is not None else None
        if (old_name, old_target) != (new_name, new_mapped):
            mismatches.append(VocabMismatch(label_id, old_name, new_name, old_target, new_mapped))
    return mismatches


def _index_new_images(
    version: DatasetVersion, images: pd.DataFrame
) -> tuple[dict[str, list[str]], Counter[int]]:
    """Decode the new images: ``sha256 of RGB pixels -> image ids`` and total pixels per label."""
    by_sha: dict[str, list[str]] = {}
    label_pixels: Counter[int] = Counter()
    for row in images.to_dict("records"):
        with Image.open(version.image_path(row)) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        by_sha.setdefault(_pixel_sha(rgb), []).append(row["image_id"])
        _add_label_pixels(label_pixels, _read_new_mask(version, row))
    return by_sha, label_pixels


def _read_new_mask(version: DatasetVersion, row: Mapping[str, Any]) -> NDArray[Any]:
    with Image.open(version.mask_path(row)) as handle:
        return _native_mask(handle)


def compare_dense_images(
    old_rows: Iterable[IdImageMask],
    new_by_sha: Mapping[str, Sequence[str]],
    load_new_mask: Callable[[str], NDArray[Any]],
) -> tuple[int, list[str], list[str], dict[str, int], Counter[int]]:
    """Match old rows to new images by pixel hash and compare the matched masks.

    Returns:
        ``(matched, old_only ids, new_only ids, mask_diffs, old label pixels)``.
    """
    remaining = {sha: list(ids) for sha, ids in new_by_sha.items()}
    old_only: list[str] = []
    mask_diffs: dict[str, int] = {}
    old_pixels: Counter[int] = Counter()
    matched = 0
    for row_id, image, mask in old_rows:
        _add_label_pixels(old_pixels, mask)
        candidates = remaining.get(_pixel_sha(image))
        if not candidates:
            old_only.append(row_id)
            continue
        new_id = candidates.pop(0)
        matched += 1
        differing = _differing_pixels(mask, load_new_mask(new_id))
        if differing:
            mask_diffs[new_id] = differing
    new_only = sorted(i for ids in remaining.values() for i in ids)
    return matched, old_only, new_only, mask_diffs, old_pixels


def _label_pixel_diff(old: Counter[int], new: Counter[int]) -> LabelCountDiff:
    return {k: (old[k], new[k]) for k in sorted(old.keys() | new.keys()) if old[k] != new[k]}


def _old_vocabulary(ds: Any) -> tuple[dict[int, str], dict[str, str | None]]:
    return _native_id2name(ds), _mapped_names(SourceLabelRegistry([ds]), ds)


def run_dense_parity(
    data_cfg: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    version: str,
    manifest_sha256: str,
    cache_dir: Path,
    client: Any,
    sync: bool = True,
    hf_revision_probe: Callable[[], str | None] = _no_revision,
) -> DenseParityReport:
    """Compare an old dense dataset with a dense version, image by image.

    Images are matched by the sha256 of the decoded RGB pixels; the native masks, the per-label
    pixel totals and the vocabulary are then compared.

    Args:
        data_cfg: The ``data`` section of the old data config.
        dataset: Dataset name.
        split: Split to build on the old side; the new side takes the same groups.
        version: Version to compare against.
        manifest_sha256: The sha256 of that version's manifest.
        cache_dir: The local cache root.
        client: An S3 client; unused with ``sync=False``.
        sync: When False, S3 is not reached and the version is read from ``cache_dir`` as staged.
        hf_revision_probe: Returns the head revision of the old loader's HF source, or ``None``.

    Raises:
        ValueError: The version is not dense.
    """
    built = _build_old(data_cfg, dataset, split)
    old = built[(dataset, split)]
    groups = old_groups(old)
    pin = Pin(name=dataset, version=version, manifest_sha256=manifest_sha256)
    if sync:
        sync_versions(
            client, {dataset: pin}, cache_dir, bucket=DATASET_BUCKET, selection={dataset: groups}
        )
    version_data = open_version(cache_dir, DATASET_BUCKET, pin)
    if version_data.manifest.kind != "dense":
        raise ValueError(f"{dataset} {version} is not a dense version")
    images = version_data.images.to_pandas()
    if groups is not None:
        images = images[images["group"].isin(groups)]
    rows_by_id = {row["image_id"]: row for row in images.to_dict("records")}
    new_by_sha, new_pixels = _index_new_images(version_data, images)
    matched, old_only, new_only, mask_diffs, old_pixels = compare_dense_images(
        old_dense_rows(old),
        new_by_sha,
        lambda image_id: _read_new_mask(version_data, rows_by_id[image_id]),
    )
    old_names, old_mapped = _old_vocabulary(old)
    labels = version_data.labels.to_pandas()
    return DenseParityReport(
        matched=matched,
        old_only=old_only,
        new_only=new_only,
        mask_diffs=mask_diffs,
        label_pixel_diff=_label_pixel_diff(old_pixels, new_pixels),
        vocab_mismatches=vocabulary_mismatches(old_names, old_mapped, labels),
        hf_revision_at_run=hf_revision_probe(),
    )


def format_dense_report(report: DenseParityReport) -> str:
    """Render a dense report as text, ending with ``PARITY OK`` or ``PARITY FAILED``."""
    mismatches = [
        f"id {m.label_id}: name old {m.old_name!r}, new {m.new_name!r}; "
        f"mapped old {m.old_mapped!r}, new {m.new_mapped!r}"
        for m in report.vocab_mismatches
    ]
    mask_diffs = [f"{image}: {n} pixels" for image, n in report.mask_diffs.items()]
    lines = [
        f"matched images: {report.matched}",
        *_format_list("old-only images", report.old_only),
        *_format_list("new-only images", report.new_only),
        *_format_list("images with differing masks", mask_diffs),
        *_format_diff("native label pixel counts", report.label_pixel_diff),
        *_format_list("vocabulary mismatches", mismatches),
        f"HF revision at run: {report.hf_revision_at_run or 'unknown'}",
        "PARITY OK" if report.ok else "PARITY FAILED",
    ]
    return "\n".join(lines)
