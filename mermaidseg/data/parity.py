"""One-off comparison of the old loaders with a synced dataset version (points or dense)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from mermaidseg.data.config import DATASET_BUCKET
from mermaidseg.data.point_dataset import select_points
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
from mermaidseg.datasets.factory import build_datasets
from mermaidseg.io import ConfigDict, preprocess_data_config
from segdata_spec.manifest import Pin
from segdata_spec.read import DatasetVersion, open_version
from segdata_spec.sync import sync

# Val/test transform size for the old loader; only its annotations are read here.
_UNUSED_INPUT_SIZE = (256, 256)
_COLUMNS = ["image_id", "row", "col", "native_name", "mapped"]
_MAX_LISTED = 10

Point = tuple[str, int, int, str | None]
CountDiff = dict[str | None, tuple[int, int]]
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


@dataclass(frozen=True)
class DenseParityReport:
    """The differences between the old and the new dense tiles; empty fields mean parity.

    Attributes:
        old_only_tiles: ``(site, tile)`` with an old mask and no new tile.
        new_only_tiles: ``(site, tile)`` of the compared sites with a new tile and no old mask.
        tile_diff: ``(site, tile)`` -> label id -> ``(old pixels, new pixels)`` where they differ.
        mapped_diff: Mapped label -> ``(old pixels, new pixels)`` over the matched tiles.
        label_map_diff: Label id -> ``(old mapped name, new mapped name)`` where they differ, for
            every id with pixels on either side.
        native_name_diff: Label id -> ``(old name, new name)``, lowercased, where the
            ``classes.json`` name differs from the ``labels`` name, for every id with pixels on
            either side.
        mask_mismatches: Matched tiles whose stored new mask differs from the old mask.
        label_count_mismatches: ``(site, tile)`` -> label id -> ``(label_counts pixels, decoded
            mask pixels)`` where they differ, over every new tile.
        tiles_matched: Tiles present on both sides.
        sites_compared: New sites that hold at least one old tile.
        excluded_sites: New site without old tiles -> ``(tiles, labeled pixels)``;
            informational, not part of ``ok``.
    """

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
            self.old_only_tiles
            or self.new_only_tiles
            or self.tile_diff
            or self.mapped_diff
            or self.label_map_diff
            or self.native_name_diff
            or self.mask_mismatches
            or self.label_count_mismatches
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


def _read_new_mask(version: DatasetVersion, row: Mapping[str, Any]) -> NDArray[Any]:
    """Decode a synced mask as the dense loader does: the first channel of a multi-channel PNG."""
    with Image.open(version.mask_path(row)) as handle:
        mask = np.asarray(handle)
    return mask[..., 0] if mask.ndim == 3 else mask


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


def _format_dense_report(report: DenseParityReport) -> str:
    lines = [
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
        _verdict(report.ok),
    ]
    return "\n".join(lines)


def format_report(report: ParityReport | DenseParityReport) -> str:
    """Render a report as text, ending with ``PARITY OK`` or ``PARITY FAILED``."""
    if isinstance(report, DenseParityReport):
        return _format_dense_report(report)
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


def _require_old(items: Collection[Any], what: str) -> None:
    """Raise ``ValueError`` when the old side yields nothing, which would compare as parity."""
    if len(items) == 0:
        raise ValueError(f"the old split has no {what}; there is nothing to compare")


def _compare_points(
    old_ds: Any, version_data: DatasetVersion, annotators: Sequence[str] | None
) -> ParityReport:
    if version_data.points is None:
        raise ValueError("the version has no points table")
    old = old_points(old_ds)
    _require_old(old, "points")
    images = version_data.images.to_pandas()
    points = select_points(version_data.points.to_pandas(), images, annotators)
    kept = images[images["image_id"].isin(points["image_id"])]
    new_groups = dict(zip(kept["image_id"], kept["group"], strict=True))
    return compare(old, new_points(version_data, points), new_groups)


def _compare_dense(
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


class _OfflineClient:
    """The S3 surface ``sync`` uses, for a cache where every object must already be staged."""

    def get_object(self, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        raise FileNotFoundError(f"offline: s3://{Bucket}/{Key} is not staged")

    def head_object(self, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        raise FileNotFoundError(f"offline: s3://{Bucket}/{Key} is not staged")


def _sync_version(
    client: Any, pin: Pin, cache_dir: Path, groups: Collection[str] | None
) -> DatasetVersion:
    """Sync ``pin`` (its tables, plus the images and masks of ``groups``) and open it."""
    sync(client, {pin.name: pin}, cache_dir, bucket=DATASET_BUCKET, selection={pin.name: groups})
    return open_version(cache_dir, DATASET_BUCKET, pin)


def run_parity(
    data_cfg: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    version: str,
    manifest_sha256: str,
    cache_dir: Path,
    client: Any | None,
    annotators: Sequence[str] | None,
    offline: bool = False,
) -> ParityReport | DenseParityReport:
    """Build the old dataset, sync and verify the pinned version, and compare with the old side.

    The sync verifies the manifest against ``manifest_sha256`` and every table against the
    manifest's sha256. A points version syncs no image and is compared point by point. A dense
    version also syncs every image and mask, each verified by size, ETag and CRC64NVME; each tile
    is then compared by pixel counts, its decoded mask must equal the old mask pixel for pixel,
    and its ``label_counts`` rows must equal its decoded mask's counts.

    Args:
        data_cfg: The ``data`` section of the old data config; its ``default`` block is expanded
            as the training path does.
        dataset: Dataset name.
        split: Split to build on the old side; the new side compares every split.
        version: Version to compare against.
        manifest_sha256: The sha256 of that version's manifest.
        cache_dir: The local cache root.
        client: An S3 client; unused, and may be ``None``, when ``offline``.
        annotators: Annotators in priority order for selecting the new points.
        offline: Verify the version already staged in ``cache_dir`` exactly as a sync does, with
            no S3 access; a file that is not staged raises ``FileNotFoundError``.

    Returns:
        The parity report over all groups of the new version that hold old images.

    Raises:
        ValueError: ``client`` is ``None`` and ``offline`` is false, or the old split has no
            points or tiles.
        FileNotFoundError: ``offline`` and a file the check needs is not staged.
    """
    if client is None and not offline:
        raise ValueError("an S3 client is required unless offline")
    cfg = preprocess_data_config(ConfigDict({"data": dict(data_cfg)})).data
    built = build_datasets(
        cfg, input_size=_UNUSED_INPUT_SIZE, names=[dataset], splits=[split], verbose=False
    )
    old_ds = built[(dataset, split)]
    pin = Pin(name=dataset, version=version, manifest_sha256=manifest_sha256)
    source = _OfflineClient() if offline else client
    version_data = _sync_version(source, pin, cache_dir, set())
    if version_data.manifest.kind != "dense":
        return _compare_points(old_ds, version_data, annotators)
    old_counts, old_mapped = old_dense_counts(old_ds)
    _require_old(old_counts, "tiles")
    version_data = _sync_version(source, pin, cache_dir, None)
    return _compare_dense(old_ds, old_counts, old_mapped, version_data)
