"""One-off comparison of the old loaders with a synced (or locally staged) dataset version."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from segdata_spec._s3 import cache_path, dataset_prefix
from segdata_spec.errors import ChecksumMismatch
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
Tile = tuple[str, str]
TileCounts = dict[Tile, Counter[int]]


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
        points_matched: Points present on both sides, counted as a multiset intersection.
        groups_compared: New groups that hold at least one old image.
        excluded_groups: New group without old images -> ``(images, points)``; informational,
            not part of ``ok``.
    """

    old_only_images: list[str] = field(default_factory=list)
    new_only_images: list[str] = field(default_factory=list)
    old_only_points: Counter[Point] = field(default_factory=Counter)
    new_only_points: Counter[Point] = field(default_factory=Counter)
    native_diff: CountDiff = field(default_factory=dict)
    mapped_diff: CountDiff = field(default_factory=dict)
    points_matched: int = 0
    groups_compared: int = 0
    excluded_groups: dict[str, tuple[int, int]] = field(default_factory=dict)

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


def _excluded_groups(excluded: pd.DataFrame, groups: pd.Series) -> dict[str, tuple[int, int]]:
    """Return group -> ``(images, points)`` of the new points outside the compared groups."""
    by_group = excluded["image_id"].groupby(groups[excluded.index])
    images, points = by_group.nunique(), by_group.size()
    return {str(g): (int(images[g]), int(points[g])) for g in images.index}


def compare(old: pd.DataFrame, new: pd.DataFrame, new_groups: Mapping[str, str]) -> ParityReport:
    """Compare old and new points over the groups of the new version that hold old images.

    Args:
        old: Frame from :func:`old_points`.
        new: Frame from :func:`new_points`.
        new_groups: ``image_id -> group`` of the new version's images.

    Returns:
        The report; ``ok`` when the points agree as multisets. New groups without old images are
        listed in ``excluded_groups`` and left out of the comparison.
    """
    compared = {new_groups[i] for i in set(old["image_id"]) if i in new_groups}
    groups = new["image_id"].map(new_groups)
    kept = groups.isin(compared)
    excluded = _excluded_groups(new[~kept], groups)
    new = new[kept]
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
        points_matched=sum((old_counter & new_counter).values()),
        groups_compared=len(compared),
        excluded_groups=excluded,
    )


def _format_list(title: str, items: Collection[Any]) -> list[str]:
    if not items:
        return [f"{title}: none"]
    shown = ", ".join(str(i) for i in list(items)[:_MAX_LISTED])
    more = f" (+{len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else ""
    return [f"{title}: {len(items)}", f"  {shown}{more}"]


def _format_diff(
    title: str, diff: Mapping[Any, tuple[Any, Any]], limit: int | None = None
) -> list[str]:
    if not diff:
        return [f"{title}: none"]
    lines = [f"{title}:", *(f"  {k}: old {o}, new {n}" for k, (o, n) in list(diff.items())[:limit])]
    if limit is not None and len(diff) > limit:
        lines.append(f"  (+{len(diff) - limit} more)")
    return lines


def _format_excluded(
    title: str, groups: Mapping[str, tuple[int, int]], units: tuple[str, str]
) -> list[str]:
    if not groups:
        return [f"{title}: none"]
    first, second = units
    lines = [f"{title}: {len(groups)}"]
    for group, (a, b) in list(groups.items())[:_MAX_LISTED]:
        lines.append(f"  {group}: {a} {first}, {b} {second}")
    if len(groups) > _MAX_LISTED:
        lines.append(f"  (+{len(groups) - _MAX_LISTED} more)")
    return lines


def _verdict(ok: bool) -> str:
    return "PARITY OK" if ok else "PARITY FAILED"


def format_report(report: ParityReport) -> str:
    """Render a report as text, ending with ``PARITY OK`` or ``PARITY FAILED``."""
    lines = [
        *_format_list("old-only images", report.old_only_images),
        *_format_list("new-only images", report.new_only_images),
        *_format_list("points only in old", sorted(report.old_only_points.elements(), key=str)),
        *_format_list("points only in new", sorted(report.new_only_points.elements(), key=str)),
        *_format_diff("native label counts", report.native_diff),
        *_format_diff("mapped label counts", report.mapped_diff),
        f"points matched: {report.points_matched}",
        f"groups compared: {report.groups_compared}",
        *_format_excluded("new groups excluded", report.excluded_groups, ("images", "points")),
        _verdict(report.ok),
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


def _verify_cached_tables(version: DatasetVersion, pin: Pin) -> None:
    """Check every cached table of ``version`` against the sha256 its manifest records.

    Raises:
        ChecksumMismatch: A cached table differs from its manifest entry, as ``sync`` raises.
    """
    prefix = dataset_prefix(pin.name, pin.version)
    for name, entry in version.manifest.files.items():
        path = cache_path(version.cache_dir, DATASET_BUCKET, prefix + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.sha256:
            raise ChecksumMismatch(f"cached file {path} does not match its expected sha256")


def _open_cached(cache_dir: Path, pin: Pin) -> DatasetVersion:
    """Open a staged version without S3, verifying its manifest and every table it lists."""
    version = open_version(cache_dir, DATASET_BUCKET, pin)
    _verify_cached_tables(version, pin)
    return version


def _require_old(items: Collection[Any], what: str) -> None:
    """Raise ``ValueError`` when the old side yields nothing, which would compare as parity."""
    if len(items) == 0:
        raise ValueError(f"the old split has no {what}; there is nothing to compare")


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

    No image is downloaded: only the parquet tables are synced. The manifest is verified against
    ``manifest_sha256`` and every table against the manifest's sha256, with or without ``sync``.

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

    Raises:
        ValueError: The old split has no points, or the version has no points table.
        PinMismatch: The manifest differs from ``manifest_sha256``.
        ChecksumMismatch: A table differs from its manifest entry.
    """
    built = _build_old(data_cfg, dataset, split)
    old = old_points(built[(dataset, split)])
    _require_old(old, "points")
    pin = Pin(name=dataset, version=version, manifest_sha256=manifest_sha256)
    if sync:
        sync_tables(client, pin, cache_dir)
        version_data = open_version(cache_dir, DATASET_BUCKET, pin)
    else:
        version_data = _open_cached(cache_dir, pin)
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

    Two old-side adapters fill it. The Hugging Face adapter (Coralscapes, UCSD mosaics) fills
    ``matched`` through ``hf_revision_at_run``; the Benthos adapter fills ``old_only_tiles``
    through ``excluded_sites``. The fields the other adapter fills stay empty.

    Attributes:
        matched: HF: images present on both sides, matched by the sha256 of their RGB pixels.
        old_only: HF: old row ids with no new image.
        new_only: HF: new image ids with no old row.
        mask_diffs: HF: new image id -> differing native mask pixels, for matched images whose
            masks differ; a shape mismatch counts every pixel of the larger mask.
        label_pixel_diff: HF: native label id -> ``(old pixels, new pixels)`` where the totals
            differ.
        vocab_mismatches: HF: label ids whose name or MERMAID mapping differ.
        hf_revision_at_run: HF: head revision of the unpinned Hugging Face source, if reachable.
        old_only_tiles: Benthos: ``(site, tile)`` with an old mask and no new tile.
        new_only_tiles: Benthos: ``(site, tile)`` of the compared sites with a new tile and no
            old mask.
        tile_diff: Benthos: ``(site, tile)`` -> label id -> ``(old pixels, new pixels)`` where
            they differ.
        mapped_diff: Benthos: mapped label -> ``(old pixels, new pixels)`` over the matched tiles.
        label_map_diff: Benthos: label id -> ``(old mapped name, new mapped name)`` where they
            differ, for every id with pixels on either side.
        native_name_diff: Benthos: label id -> ``(old name, new name)``, lowercased, where the
            ``classes.json`` name differs from the ``labels`` name, for every id with pixels on
            either side.
        mask_mismatches: Benthos: matched tiles whose stored new mask differs from the old mask.
        label_count_mismatches: Benthos: ``(site, tile)`` -> label id -> ``(label_counts pixels,
            decoded mask pixels)`` where they differ, over every new tile.
        tiles_matched: Benthos: tiles present on both sides.
        sites_compared: Benthos: new sites that hold at least one old tile.
        excluded_sites: Benthos: new site without old tiles -> ``(tiles, labeled pixels)``;
            informational, not part of ``ok``.
    """

    matched: int = 0
    old_only: list[str] = field(default_factory=list)
    new_only: list[str] = field(default_factory=list)
    mask_diffs: dict[str, int] = field(default_factory=dict)
    label_pixel_diff: LabelCountDiff = field(default_factory=dict)
    vocab_mismatches: list[VocabMismatch] = field(default_factory=list)
    hf_revision_at_run: str | None = None
    old_only_tiles: list[Tile] = field(default_factory=list)
    new_only_tiles: list[Tile] = field(default_factory=list)
    tile_diff: dict[Tile, dict[int, tuple[int, int]]] = field(default_factory=dict)
    mapped_diff: CountDiff = field(default_factory=dict)
    label_map_diff: dict[int, tuple[str | None, str | None]] = field(default_factory=dict)
    native_name_diff: dict[int, tuple[str | None, str | None]] = field(default_factory=dict)
    mask_mismatches: list[Tile] = field(default_factory=list)
    label_count_mismatches: dict[Tile, dict[int, tuple[int, int]]] = field(default_factory=dict)
    tiles_matched: int = 0
    sites_compared: int = 0
    excluded_sites: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no field reports a difference."""
        return not (
            self.old_only
            or self.new_only
            or self.mask_diffs
            or self.label_pixel_diff
            or self.vocab_mismatches
            or self.old_only_tiles
            or self.new_only_tiles
            or self.tile_diff
            or self.mapped_diff
            or self.label_map_diff
            or self.native_name_diff
            or self.mask_mismatches
            or self.label_count_mismatches
        )


def _no_revision() -> str | None:
    return None


def _native_mask(mask: Any) -> NDArray[Any]:
    array = np.asarray(mask)
    return array[..., 0] if array.ndim == 3 else array


def _read_new_mask(version: DatasetVersion, row: Mapping[str, Any]) -> NDArray[Any]:
    """Decode a synced mask as the dense loaders do: the first channel of a multi-channel PNG."""
    with Image.open(version.mask_path(row)) as handle:
        return _native_mask(handle)


# --- dense: Hugging Face adapter (Coralscapes, UCSD mosaics) -------------------------------


def _rgb_array(image: Any) -> NDArray[Any]:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    return np.asarray(image)


def _pixel_sha(array: NDArray[Any]) -> str:
    digest = hashlib.sha256(repr(array.shape).encode())
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


def _restrict_to_original_splits(ds: Any, images: pd.DataFrame) -> pd.DataFrame:
    """Keep the images whose ``original_split`` the old loader kept, when it filters on one."""
    resolve = getattr(ds, "_resolved_original_splits", None)
    allowed = None if resolve is None else resolve()
    if allowed is None:
        return images
    kept = [json.loads(attrs)["original_split"] in allowed for attrs in images["attrs"]]
    return images[kept]


def _subset_vocabulary(
    ds: Any,
    old_names: Mapping[int, str],
    labels: pd.DataFrame,
) -> tuple[dict[int, str], pd.DataFrame]:
    """Limit the native names and the new labels to the ids of ``ds.class_subset``, if set."""
    if getattr(ds, "class_subset", None) is None:
        return dict(old_names), labels
    ids = {i for i, name in old_names.items() if name in ds.source_name2id}
    return {i: old_names[i] for i in ids}, labels[labels["label_id"].isin(ids)]


def _hf_dense_report(
    old: Any,
    version_data: DatasetVersion,
    groups: set[str] | None,
    hf_revision_probe: Callable[[], str | None],
) -> DenseParityReport:
    """Match HF rows to new images by pixel hash, then compare masks, pixels and vocabulary."""
    images = version_data.images.to_pandas()
    if groups is not None:
        images = images[images["group"].isin(groups)]
    images = _restrict_to_original_splits(old, images)
    rows_by_id = {row["image_id"]: row for row in images.to_dict("records")}
    new_by_sha, new_pixels = _index_new_images(version_data, images)
    matched, old_only, new_only, mask_diffs, old_pixels = compare_dense_images(
        old_dense_rows(old),
        new_by_sha,
        lambda image_id: _read_new_mask(version_data, rows_by_id[image_id]),
    )
    old_names, old_mapped = _old_vocabulary(old)
    old_names, labels = _subset_vocabulary(old, old_names, version_data.labels.to_pandas())
    return DenseParityReport(
        matched=matched,
        old_only=old_only,
        new_only=new_only,
        mask_diffs=mask_diffs,
        label_pixel_diff=_label_pixel_diff(old_pixels, new_pixels),
        vocab_mismatches=vocabulary_mismatches(old_names, old_mapped, labels),
        hf_revision_at_run=hf_revision_probe(),
    )


# --- dense: Benthos adapter ----------------------------------------------------------------


def _labeled_pixels(mask: NDArray[Any]) -> Counter[int]:
    """Count the pixels per label id of a mask, excluding the unlabeled class 0."""
    pixels = np.bincount(np.asarray(mask).ravel())
    return Counter({int(i): int(n) for i, n in enumerate(pixels) if i and n})


def old_dense_counts(ds: Any) -> tuple[TileCounts, dict[int, str | None]]:
    """Count the labeled pixels per tile of an old dense dataset, and map its label ids.

    Args:
        ds: An old ``BenthosYuvalCoralsDataset``; registering it sets its global offset.

    Returns:
        ``(site, tile) -> label id -> pixels`` read from the raw masks, with the unlabeled class
        0 excluded, and ``classes.json`` label id -> mapped target name (``None`` when unmapped).
    """
    names = _mapped_names(SourceLabelRegistry([ds]), ds)
    mapped = {int(i): names.get(name) for name, i in ds._classes_global.items()}
    counts: TileCounts = {}
    for site, tile in zip(ds.df_images["site"], ds.df_images["image_id"], strict=True):
        counts[(str(site), str(tile))] = _labeled_pixels(ds.read_label(image_id=tile, site=site))
    return counts, mapped


def _tile_rows(version: DatasetVersion) -> dict[Tile, dict[str, Any]]:
    """Key each ``images`` row of a dense version by ``(group, source_name stem)``.

    Raises:
        ValueError: A tile has no ``source_name``, or two tiles share a key.
    """
    rows = version.images.to_pylist()
    if any(row["source_name"] is None for row in rows):
        raise ValueError("a dense tile has no source_name; it cannot be matched to the old tiles")
    keys = [(str(row["group"]), Path(row["source_name"]).stem) for row in rows]
    colliding = sorted(key for key, n in Counter(keys).items() if n > 1)
    if colliding:
        raise ValueError(f"dense tiles share a (group, source_name stem) key: {colliding}")
    return dict(zip(keys, rows, strict=True))


def new_dense_counts(version: DatasetVersion) -> tuple[TileCounts, dict[int, str | None]]:
    """Return the labeled pixels per tile of a dense version, and its label ids' mapped names.

    Tiles are keyed by ``(group, source_name stem)``; the counts come from ``label_counts`` and
    exclude the unlabeled 0.

    Raises:
        ValueError: A tile has no ``source_name``, or two tiles share a key.
    """
    tiles = {row["image_id"]: tile for tile, row in _tile_rows(version).items()}
    counts: TileCounts = {tile: Counter() for tile in tiles.values()}
    for row in version.label_counts.to_pandas().itertuples():
        if row.label_id:
            counts[tiles[row.image_id]][int(row.label_id)] += int(row.n)
    labels = version.labels.to_pandas()
    attribute = labels["mermaid_benthic_attribute"]
    mapped = {
        int(i): (None if pd.isna(a) else str(a).lower())
        for i, a in zip(labels["label_id"], attribute, strict=True)
    }
    return counts, mapped


def _decoded_new_counts(version: DatasetVersion) -> TileCounts:
    """Count the labeled pixels of every synced mask of a dense version, keyed like its tiles."""
    return {
        tile: _labeled_pixels(_read_new_mask(version, row))
        for tile, row in _tile_rows(version).items()
    }


def _mask_mismatches(old_ds: Any, version: DatasetVersion, tiles: Collection[Tile]) -> list[Tile]:
    """Return the ``tiles`` whose synced new mask is not pixel-equal to the old raw mask."""
    rows = _tile_rows(version)
    return [
        (site, tile)
        for site, tile in sorted(tiles)
        if not np.array_equal(
            _read_new_mask(version, rows[(site, tile)]),
            np.asarray(old_ds.read_label(image_id=tile, site=site)),
        )
    ]


def _tile_count_diff(
    left: TileCounts, right: TileCounts, tiles: Collection[Tile]
) -> dict[Tile, dict[int, tuple[int, int]]]:
    """Return tile -> label id -> ``(left pixels, right pixels)`` where they differ."""
    result = {}
    for tile in sorted(tiles):
        a, b = left.get(tile, Counter()), right.get(tile, Counter())
        diff = {label: (a[label], b[label]) for label in sorted(a.keys() | b.keys())}
        diff = {label: pair for label, pair in diff.items() if pair[0] != pair[1]}
        if diff:
            result[tile] = diff
    return result


def _mapped_totals(
    counts: TileCounts, tiles: Collection[Tile], mapped: Mapping[int, str | None]
) -> Counter[str | None]:
    totals: Counter[str | None] = Counter()
    for tile in tiles:
        for label, n in counts[tile].items():
            totals[mapped.get(label)] += n
    return totals


def _label_map_diff(
    counts: Collection[Counter[int]],
    old_mapped: Mapping[int, str | None],
    new_mapped: Mapping[int, str | None],
) -> dict[int, tuple[str | None, str | None]]:
    """Return label id -> ``(old name, new name)`` where they differ, over the ids with pixels."""
    ids = sorted({label for c in counts for label, n in c.items() if n})
    pairs = {i: (old_mapped.get(i), new_mapped.get(i)) for i in ids}
    return {i: pair for i, pair in pairs.items() if pair[0] != pair[1]}


def _old_native_names(ds: Any) -> dict[int, str]:
    """Return the ``classes.json`` label id -> lowercased name of an old dense dataset."""
    return {int(i): str(name).lower() for name, i in ds._classes_global.items()}


def _new_native_names(version: DatasetVersion) -> dict[int, str]:
    """Return the ``labels`` label id -> lowercased name of a dense version."""
    labels = version.labels.to_pandas()
    return {int(i): str(n).lower() for i, n in zip(labels["label_id"], labels["name"], strict=True)}


def _excluded_sites(counts: TileCounts) -> dict[str, tuple[int, int]]:
    """Return site -> ``(tiles, labeled pixels)`` of ``counts``."""
    tiles: Counter[str] = Counter()
    pixels: Counter[str] = Counter()
    for (site, _), c in counts.items():
        tiles[site] += 1
        pixels[site] += sum(c.values())
    return {site: (tiles[site], pixels[site]) for site in sorted(tiles)}


def compare_dense(
    old: TileCounts,
    new: TileCounts,
    old_mapped: Mapping[int, str | None],
    new_mapped: Mapping[int, str | None],
) -> DenseParityReport:
    """Compare per-tile pixel counts over the sites of the new version that hold old tiles.

    Args:
        old: Counts from :func:`old_dense_counts`.
        new: Counts from :func:`new_dense_counts`.
        old_mapped: Label id -> mapped name on the old side.
        new_mapped: Label id -> mapped name on the new side.

    Returns:
        The report; ``ok`` when every tile matches, the mapped pixel totals agree and every label
        id with pixels maps to the same name on both sides. New sites without old tiles are
        listed in ``excluded_sites`` and left out of the comparison.
    """
    sites = {site for site, _ in old}
    excluded = {tile: c for tile, c in new.items() if tile[0] not in sites}
    new = {tile: c for tile, c in new.items() if tile[0] in sites}
    both = sorted(old.keys() & new.keys())
    return DenseParityReport(
        old_only_tiles=sorted(old.keys() - new.keys()),
        new_only_tiles=sorted(new.keys() - old.keys()),
        tile_diff=_tile_count_diff(old, new, both),
        mapped_diff=_count_diff(
            _mapped_totals(old, both, old_mapped), _mapped_totals(new, both, new_mapped)
        ),
        label_map_diff=_label_map_diff([*old.values(), *new.values()], old_mapped, new_mapped),
        tiles_matched=len(both),
        sites_compared=len({site for site, _ in new}),
        excluded_sites=_excluded_sites(excluded),
    )


def _benthos_dense_report(
    old_ds: Any,
    old_counts: TileCounts,
    old_mapped: Mapping[int, str | None],
    version_data: DatasetVersion,
) -> DenseParityReport:
    """Compare the counts, the label names, then the synced masks and ``label_counts``."""
    new_counts, new_mapped = new_dense_counts(version_data)
    report = compare_dense(old_counts, new_counts, old_mapped, new_mapped)
    matched = old_counts.keys() & new_counts.keys()
    decoded = _decoded_new_counts(version_data)
    return replace(
        report,
        mask_mismatches=_mask_mismatches(old_ds, version_data, matched),
        label_count_mismatches=_tile_count_diff(new_counts, decoded, new_counts.keys()),
        native_name_diff=_label_map_diff(
            [*old_counts.values(), *new_counts.values()],
            _old_native_names(old_ds),
            _new_native_names(version_data),
        ),
    )


# --- dense: dispatch -----------------------------------------------------------------------


def _is_benthos(ds: Any) -> bool:
    """True for an old dataset with the Benthos loader interface (raw masks + ``classes.json``)."""
    return hasattr(ds, "read_label") and hasattr(ds, "_classes_global")


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
    """Compare an old dense dataset with a dense version, through the adapter of the old loader.

    A Benthos-shaped old dataset (``read_label`` and ``_classes_global``) is compared tile by
    tile: per-tile pixel counts, raw mask equality, ``label_counts`` against the decoded masks,
    and label names and mappings; every group is synced. Any other is read as Hugging Face rows,
    matched to the new images by the sha256 of the RGB pixels, and its masks, per-label pixel
    totals and vocabulary are compared; only the groups it covers are synced.

    Args:
        data_cfg: The ``data`` section of the old data config.
        dataset: Dataset name.
        split: Split to build on the old side.
        version: Version to compare against.
        manifest_sha256: The sha256 of that version's manifest.
        cache_dir: The local cache root.
        client: An S3 client; unused with ``sync=False``.
        sync: When False, S3 is not reached: the manifest and tables are verified in
            ``cache_dir`` and the images and masks are read from it as staged.
        hf_revision_probe: Returns the head revision of the old loader's HF source, or ``None``;
            called by the Hugging Face adapter only.

    Raises:
        ValueError: The old split has no tiles or images, or the version is not dense.
        PinMismatch: The manifest differs from ``manifest_sha256``.
        ChecksumMismatch: A table, or with ``sync`` an image or mask, differs from its record.
        FileNotFoundError: With ``sync=False``, an image or mask the check reads is not staged.
    """
    built = _build_old(data_cfg, dataset, split)
    old = built[(dataset, split)]
    benthos = _is_benthos(old)
    if benthos:
        old_counts, old_mapped = old_dense_counts(old)
        _require_old(old_counts, "tiles")
        groups = None
    else:
        _require_old(old.dataset, "images")
        groups = old_groups(old)
    pin = Pin(name=dataset, version=version, manifest_sha256=manifest_sha256)
    if sync:
        sync_versions(
            client, {dataset: pin}, cache_dir, bucket=DATASET_BUCKET, selection={dataset: groups}
        )
        version_data = open_version(cache_dir, DATASET_BUCKET, pin)
    else:
        version_data = _open_cached(cache_dir, pin)
    if version_data.manifest.kind != "dense":
        raise ValueError(f"{dataset} {version} is not a dense version")
    if benthos:
        return _benthos_dense_report(old, old_counts, old_mapped, version_data)
    return _hf_dense_report(old, version_data, groups, hf_revision_probe)


def _format_tile_diff(
    title: str, diff: Mapping[Tile, Mapping[int, tuple[int, int]]], sides: tuple[str, str]
) -> list[str]:
    if not diff:
        return [f"{title}: none"]
    left, right = sides
    lines = [f"{title}: {len(diff)} tiles"]
    for (site, tile), labels in list(diff.items())[:_MAX_LISTED]:
        counts = ", ".join(f"label {k} {left} {a}, {right} {b}" for k, (a, b) in labels.items())
        lines.append(f"  {site}/{tile}: {counts}")
    if len(diff) > _MAX_LISTED:
        lines.append(f"  (+{len(diff) - _MAX_LISTED} more)")
    return lines


def _format_hf_fields(report: DenseParityReport) -> list[str]:
    mismatches = [
        f"id {m.label_id}: name old {m.old_name!r}, new {m.new_name!r}; "
        f"mapped old {m.old_mapped!r}, new {m.new_mapped!r}"
        for m in report.vocab_mismatches
    ]
    mask_diffs = [f"{image}: {n} pixels" for image, n in report.mask_diffs.items()]
    return [
        f"matched images: {report.matched}",
        *_format_list("old-only images", report.old_only),
        *_format_list("new-only images", report.new_only),
        *_format_list("images with differing masks", mask_diffs),
        *_format_diff("native label pixel counts", report.label_pixel_diff),
        *_format_list("vocabulary mismatches", mismatches),
        f"HF revision at run: {report.hf_revision_at_run or 'unknown'}",
    ]


def _format_benthos_fields(report: DenseParityReport) -> list[str]:
    return [
        *_format_list("old-only tiles", [f"{s}/{t}" for s, t in report.old_only_tiles]),
        *_format_list("new-only tiles", [f"{s}/{t}" for s, t in report.new_only_tiles]),
        *_format_tile_diff("tile label counts", report.tile_diff, ("old", "new")),
        *_format_list("mask pixel mismatches", [f"{s}/{t}" for s, t in report.mask_mismatches]),
        *_format_tile_diff(
            "label_counts vs mask", report.label_count_mismatches, ("table", "mask")
        ),
        *_format_diff("mapped label pixels", report.mapped_diff),
        *_format_diff("label id mapping", report.label_map_diff),
        *_format_diff("label id native names", report.native_name_diff, _MAX_LISTED),
        f"tiles matched: {report.tiles_matched}",
        f"sites compared: {report.sites_compared}",
        *_format_excluded("new sites excluded", report.excluded_sites, ("tiles", "pixels")),
    ]


def format_dense_report(report: DenseParityReport) -> str:
    """Render a dense report as text, ending with ``PARITY OK`` or ``PARITY FAILED``."""
    lines = [*_format_hf_fields(report), *_format_benthos_fields(report), _verdict(report.ok)]
    return "\n".join(lines)
