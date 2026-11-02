"""Build a fixed CoralscapesV2 linear-probe point set.

The builder reads a Cityscapes-style local mirror of ``josauder/coralscapesV2``
(``leftImg8bit`` + ``gtFine/train/*/*_gtFine_labelIds.png``) and writes a JSON
file that pins the probe pixels used to train the linear SVM. Evaluation locates
those images by file stem in the mirror, or by the md5 of the label array in the
HuggingFace dataset. Nothing in this script is random: label paths are sorted,
``scipy.ndimage.label`` is deterministic, and the interior pixel is
``numpy.argmax`` of the Euclidean distance transform (first maximum, row-major).

Algorithm (full train census, target ``K`` points per class, default ``K=10``):

1. Scan every training label. For each class id 1..95, find 8-connected
   components (3x3 structuring element) and keep those with area at least
   ``--min-component-size`` (default 4 px).
2. Let ``N`` be the number of kept components for a class.
   - ``N == 0``: raise. The JSON is not written.
   - ``N >= K``: take ``K`` different components. Group by image. Within an
     image, order components by size descending, then component index. Order
     images by their largest component size descending, then stem, site, and
     image index. Round-robin one component per image until ``K`` are chosen.
     Each chosen component contributes one pixel: the distance-transform
     maximum. No exclusion disk.
   - ``N < K``: use every component. Order them by size descending, then stem,
     site, component index, and image index. Each component's quota is
     ``K // N``, and the first ``K % N`` components (the largest) get one extra.
     Four components and ``K=10`` therefore get quotas 3, 3, 2, 2.
3. A quota ``q > 1`` is placed by repeating: take the distance-transform maximum
   of the remaining mask, then set False every pixel with
   ``dy**2 + dx**2 <= radius**2`` (default radius 16, clipped to the image).
   If the mask is empty before ``q`` pixels have been placed, the pixels already
   placed are kept. The unused quota is not moved onto another component, so
   that class is stored with fewer than ``K`` points. Each such component is
   listed in ``placement_shortfalls`` (class, image, component index, size,
   requested quota, and how many pixels were placed). A component with fewer
   pixels than its quota, or one that yields no pixel, still raises, and the
   JSON is not written.
4. After selection, every chosen image must be 1024x2048, and every class must
   have at least one distinct ``(image, row, col)`` pixel. Duplicate pixels
   raise before the JSON is written. A class total below ``K`` is written only
   when it is explained by those exclusion-disk shortfalls.
5. The image list is exactly the images that own at least one point, sorted by
   site then stem.

Usage:
    uv run python scripts/build_coralscapes_v2_probe_points.py \
        --local-root /Users/jonathan/mit/coralscapes_v2 \
        --output configs/eval/coralscapes_v2_probe_points.json \
        --per-class 10 \
        --min-component-size 4 \
        --exclusion-radius 16

On that local mirror (1790 train masks) three components can hold only one
pixel once a radius-16 disk is removed, so those classes are stored short of
the target of 10:

- seriatopora dead (id 75), site30/site30_000148_000236, component 1, 251 px,
  quota 3, placed 1
- seriatopora dead (id 75), same image, component 2, 156 px, quota 3, placed 1
- turbinaria dead (id 92), site33/site33_000058_033600, component 6, 65 px,
  quota 2, placed 1

seriatopora dead therefore has 6 probe pixels and turbinaria dead has 9. Every
other class has 10. The written set has 663 images and 945 points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from scipy import ndimage

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME

logger = logging.getLogger(__name__)

NUM_CLASSES = 95  # native ids 1..95 (0 = unlabeled / ignore)
EXPECTED_HEIGHT = 1024
EXPECTED_WIDTH = 2048
_STRUCT8 = np.ones((3, 3), dtype=bool)


class ProbeSelectionError(RuntimeError):
    """Raised when a class cannot be given the requested probe pixels."""


class Component(NamedTuple):
    image_index: int
    component_index: int
    size: int
    site: str
    stem: str


class ImageRecord(NamedTuple):
    path: str
    site: str
    stem: str
    height: int
    width: int


def _label_paths(root: Path, split: str) -> list[Path]:
    return sorted((root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))


def _stem_from_label_path(path: Path) -> tuple[str, str]:
    site = path.parent.name
    stem = path.name.replace("_gtFine_labelIds.png", "")
    return site, stem


def _load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def _class_name(class_id: int) -> str:
    name = CORALSCAPES_V2_ID2NAME.get(class_id)
    if name is None:
        raise ProbeSelectionError(f"class id {class_id} is outside CoralscapesV2 ids 1..95")
    return name


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """Return ``(row, col)`` at the first maximum of the Euclidean distance transform."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        raise ProbeSelectionError("distance transform requested on an empty mask")
    distance = ndimage.distance_transform_edt(mask)
    flat_index = int(np.argmax(distance))
    row, col = np.unravel_index(flat_index, mask.shape)
    return int(row), int(col)


def exclude_disk(mask: np.ndarray, row: int, col: int, radius: int) -> None:
    """Set False on ``mask`` for every pixel with ``dy**2 + dx**2 <= radius**2``."""
    if radius < 0:
        raise ValueError(f"exclusion radius must be >= 0, got {radius}")
    height, width = mask.shape
    row0 = max(0, row - radius)
    row1 = min(height, row + radius + 1)
    col0 = max(0, col - radius)
    col1 = min(width, col + radius + 1)
    ys = np.arange(row0, row1, dtype=np.int32)
    xs = np.arange(col0, col1, dtype=np.int32)
    dy = ys[:, None] - row
    dx = xs[None, :] - col
    disk = dy * dy + dx * dx <= radius * radius
    hit_rows, hit_cols = np.nonzero(disk)
    mask[hit_rows + row0, hit_cols + col0] = False


def pick_points_in_component(
    component_mask: np.ndarray, quota: int, exclusion_radius: int
) -> list[tuple[int, int]]:
    """Place ``quota`` interior pixels inside one component.

    Quota 1 is the distance-transform maximum and does not punch a disk.
    Quota > 1 repeats that maximum, then removes an exclusion disk. If the
    component runs out of pixels after at least one point, the points already
    placed are returned and the caller records the shortfall. Fewer raw pixels
    than ``quota``, or a component that yields no pixel, still raises.
    """
    if quota < 1:
        raise ValueError(f"quota must be >= 1, got {quota}")
    if component_mask.dtype != bool:
        component_mask = component_mask.astype(bool)
    n_pixels = int(component_mask.sum())
    if n_pixels < quota:
        raise ProbeSelectionError(
            f"component has {n_pixels} pixels, which is fewer than quota {quota}"
        )

    work = component_mask.copy()
    points: list[tuple[int, int]] = []
    for _placed in range(quota):
        if not work.any():
            break
        row, col = interior_point(work)
        if not component_mask[row, col]:
            raise ProbeSelectionError(
                f"interior point {(row, col)} is outside the original component"
            )
        points.append((row, col))
        if quota > 1:
            exclude_disk(work, row, col, exclusion_radius)

    if not points:
        raise ProbeSelectionError(
            f"exclusion disk of radius {exclusion_radius}px left no probe pixel "
            f"(component has {n_pixels} pixels, quota {quota})"
        )
    if len(set(points)) != len(points):
        raise ProbeSelectionError(f"duplicate probe pixels inside one component: {points}")
    return points


def assign_quotas(n_components: int, per_class: int) -> list[int]:
    """Even quotas for ``n_components < per_class``.

    The caller must pass components ordered largest-first. The first
    ``per_class % n_components`` entries receive one extra point, so 4 components
    and ``per_class=10`` return ``[3, 3, 2, 2]``.
    """
    if per_class < 1:
        raise ValueError(f"per_class must be >= 1, got {per_class}")
    if n_components <= 0:
        raise ProbeSelectionError("cannot assign quotas with zero components")
    if n_components >= per_class:
        raise ValueError(
            f"assign_quotas is only for N < per_class (N={n_components}, per_class={per_class})"
        )
    base, remainder = divmod(per_class, n_components)
    return [base + (1 if i < remainder else 0) for i in range(n_components)]


def select_components_round_robin(components: list[Component], per_class: int) -> list[Component]:
    """Pick ``per_class`` components, spreading them across images, largest first."""
    if per_class < 1:
        raise ValueError(f"per_class must be >= 1, got {per_class}")
    by_image: dict[int, list[Component]] = defaultdict(list)
    for component in components:
        by_image[component.image_index].append(component)
    for group in by_image.values():
        group.sort(key=lambda component: (-component.size, component.component_index))

    image_order = sorted(
        by_image.keys(),
        key=lambda image_index: (
            -by_image[image_index][0].size,
            by_image[image_index][0].stem,
            by_image[image_index][0].site,
            image_index,
        ),
    )

    picked: list[Component] = []
    while len(picked) < per_class:
        progressed = False
        for image_index in image_order:
            group = by_image[image_index]
            if not group:
                continue
            picked.append(group.pop(0))
            progressed = True
            if len(picked) >= per_class:
                break
        if not progressed:
            break
    return picked


def select_class_assignments(
    components: list[Component], per_class: int
) -> list[tuple[Component, int]]:
    """Return ``(component, quota)`` pairs that together supply ``per_class`` points.

    ``N >= per_class`` uses one point on each of ``per_class`` different components.
    ``N < per_class`` uses every component and splits the points as evenly as possible.
    """
    n_components = len(components)
    if n_components == 0:
        raise ProbeSelectionError("class has no connected components")
    if n_components >= per_class:
        picked = select_components_round_robin(components, per_class)
        if len(picked) != per_class:
            raise ProbeSelectionError(
                f"round-robin selected {len(picked)} components, expected {per_class} "
                f"from {n_components} available"
            )
        return [(component, 1) for component in picked]

    ordered = sorted(
        components,
        key=lambda component: (
            -component.size,
            component.stem,
            component.site,
            component.component_index,
            component.image_index,
        ),
    )
    quotas = assign_quotas(n_components, per_class)
    return list(zip(ordered, quotas, strict=True))


def components_in_label(
    label: np.ndarray, min_component_size: int
) -> dict[int, list[tuple[int, int]]]:
    """Return ``class_id -> [(component_index, size), ...]`` for one label mask.

    ``component_index`` is the ``ndimage.label`` id (1-based). Re-running this
    function on the same array yields the same ids.
    """
    if label.ndim != 2:
        raise ProbeSelectionError(f"label must be 2D, got shape {label.shape}")
    max_id = int(label.max()) if label.size else 0
    if max_id > NUM_CLASSES:
        raise ProbeSelectionError(
            f"label contains class id {max_id}, but CoralscapesV2 ids only go to {NUM_CLASSES}"
        )
    present = [int(class_id) for class_id in np.unique(label) if 1 <= int(class_id) <= NUM_CLASSES]
    found: dict[int, list[tuple[int, int]]] = {}
    for class_id in present:
        labeled, n_components = ndimage.label(label == class_id, structure=_STRUCT8)
        if n_components == 0:
            continue
        sizes = np.bincount(labeled.reshape(-1))
        kept: list[tuple[int, int]] = []
        for component_index in range(1, n_components + 1):
            size = int(sizes[component_index])
            if size < min_component_size:
                continue
            kept.append((component_index, size))
        if kept:
            found[class_id] = kept
    return found


def census_train_components(
    records: list[ImageRecord], min_component_size: int
) -> dict[int, list[Component]]:
    """Collect kept components for every class across the training split."""
    by_class: dict[int, list[Component]] = defaultdict(list)
    n_images = len(records)
    for image_index, record in enumerate(records):
        label = _load_label(Path(record.path))
        if label.shape != (record.height, record.width):
            raise ProbeSelectionError(
                f"{record.site}/{record.stem} label shape {label.shape} "
                f"does not match the recorded size {(record.height, record.width)}"
            )
        for class_id, comps in components_in_label(label, min_component_size).items():
            for component_index, size in comps:
                by_class[class_id].append(
                    Component(
                        image_index=image_index,
                        component_index=component_index,
                        size=size,
                        site=record.site,
                        stem=record.stem,
                    )
                )
        if (image_index + 1) % 100 == 0 or image_index + 1 == n_images:
            logger.info("census %d/%d training masks", image_index + 1, n_images)
    return by_class


def _points_for_image(
    label: np.ndarray,
    jobs: list[tuple[int, Component, int]],
    exclusion_radius: int,
) -> tuple[list[dict], list[str], list[dict]]:
    """Place probe pixels for every ``(class_id, component, quota)`` job in one image.

    Returns ``(points, errors, shortfalls)``. A hard placement failure is recorded
    and the rest of the image is still attempted. An exclusion disk that stops a
    component early is a shortfall: the pixels already placed are kept, and the
    unused quota stays unused.
    """
    points: list[dict] = []
    errors: list[str] = []
    shortfalls: list[dict] = []
    jobs_by_class: dict[int, list[tuple[Component, int]]] = defaultdict(list)
    for class_id, component, quota in jobs:
        jobs_by_class[class_id].append((component, quota))

    for class_id, class_jobs in jobs_by_class.items():
        labeled, _n_components = ndimage.label(label == class_id, structure=_STRUCT8)
        for component, quota in class_jobs:
            component_mask = labeled == component.component_index
            size = int(component_mask.sum())
            where = (
                f"class {_class_name(class_id)} ({class_id}) in "
                f"{component.site}/{component.stem}, component {component.component_index}"
            )
            if size != component.size:
                errors.append(
                    f"{where}: size {size} on the second pass, census recorded {component.size}"
                )
                continue
            try:
                placed = pick_points_in_component(
                    component_mask, quota=quota, exclusion_radius=exclusion_radius
                )
            except ProbeSelectionError as exc:
                errors.append(f"{where} size {size}, quota {quota}: {exc}")
                continue
            if len(placed) < quota:
                shortfalls.append(
                    {
                        "class_id": int(class_id),
                        "class_name": _class_name(class_id),
                        "site": component.site,
                        "stem": component.stem,
                        "component_index": int(component.component_index),
                        "component_size": int(size),
                        "quota": int(quota),
                        "placed": len(placed),
                    }
                )
            for point_index, (row, col) in enumerate(placed):
                if int(label[row, col]) != class_id:
                    errors.append(
                        f"{where} point {(row, col)} landed on label {int(label[row, col])}"
                    )
                    continue
                points.append(
                    {
                        "class_id": int(class_id),
                        "row": int(row),
                        "col": int(col),
                        "component_size": int(size),
                        "component_index": int(component.component_index),
                        "point_index_in_component": int(point_index),
                        "component_quota": int(quota),
                    }
                )
    return points, errors, shortfalls


def build(
    local_root: Path,
    output: Path,
    per_class: int,
    min_component_size: int,
    exclusion_radius: int,
) -> None:
    if per_class < 1:
        raise ValueError(f"--per-class must be >= 1, got {per_class}")
    if min_component_size < 1:
        raise ValueError(f"--min-component-size must be >= 1, got {min_component_size}")
    if exclusion_radius < 0:
        raise ValueError(f"--exclusion-radius must be >= 0, got {exclusion_radius}")

    label_paths = _label_paths(local_root, "train")
    if not label_paths:
        raise FileNotFoundError(f"No train labelIds under {local_root}/gtFine/train")
    logger.info("Found %d train label masks under %s", len(label_paths), local_root)

    records: list[ImageRecord] = []
    for path in label_paths:
        site, stem = _stem_from_label_path(path)
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            ImageRecord(path=str(path), site=site, stem=stem, height=int(height), width=int(width))
        )

    by_class = census_train_components(records, min_component_size=min_component_size)

    assignments: dict[int, list[tuple[Component, int]]] = {}
    for class_id in range(1, NUM_CLASSES + 1):
        components = by_class.get(class_id, [])
        try:
            assignments[class_id] = select_class_assignments(components, per_class)
        except ProbeSelectionError as exc:
            raise ProbeSelectionError(
                f"class {_class_name(class_id)} ({class_id}) has {len(components)} "
                f"components of size >= {min_component_size} in the train split: {exc}"
            ) from exc

    jobs_by_image: dict[int, list[tuple[int, Component, int]]] = defaultdict(list)
    for class_id, class_assignments in assignments.items():
        for component, quota in class_assignments:
            jobs_by_image[component.image_index].append((class_id, component, quota))

    image_points: dict[int, list[dict]] = {}
    placement_errors: list[str] = []
    placement_shortfalls: list[dict] = []
    used_image_indexes = sorted(jobs_by_image)
    for done, image_index in enumerate(used_image_indexes, start=1):
        record = records[image_index]
        if (record.height, record.width) != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
            placement_errors.append(
                f"{record.site}/{record.stem} is {record.height}x{record.width}, "
                f"expected {EXPECTED_HEIGHT}x{EXPECTED_WIDTH}"
            )
            continue
        label = _load_label(Path(record.path))
        if label.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
            placement_errors.append(
                f"{record.site}/{record.stem} label shape {label.shape} is not "
                f"{EXPECTED_HEIGHT}x{EXPECTED_WIDTH}"
            )
            continue
        points, errors, shortfalls = _points_for_image(
            label, jobs_by_image[image_index], exclusion_radius=exclusion_radius
        )
        image_points[image_index] = points
        placement_errors.extend(errors)
        placement_shortfalls.extend(shortfalls)
        if done % 25 == 0 or done == len(used_image_indexes):
            logger.info(
                "placed points in %d/%d images (%d placement errors, %d shortfalls so far)",
                done,
                len(used_image_indexes),
                len(placement_errors),
                len(placement_shortfalls),
            )
    if placement_errors:
        preview = "\n".join(f"  - {message}" for message in placement_errors)
        raise ProbeSelectionError(
            f"{len(placement_errors)} probe-point placement failure(s). "
            "The JSON was not written.\n"
            f"{preview}"
        )

    unmatched_shortfalls = {
        (item["class_id"], item["site"], item["stem"], item["component_index"]): item
        for item in placement_shortfalls
    }
    if len(unmatched_shortfalls) != len(placement_shortfalls):
        raise ProbeSelectionError("duplicate placement shortfall records")

    points_per_class: dict[int, int] = {}
    detail: dict[str, dict] = {}
    for class_id in range(1, NUM_CLASSES + 1):
        class_assignments = assignments[class_id]
        seen: set[tuple[int, int, int]] = set()
        placed_counts: list[int] = []
        for component, quota in class_assignments:
            component_points = [
                point
                for point in image_points.get(component.image_index, [])
                if point["class_id"] == class_id
                and point["component_index"] == component.component_index
            ]
            n_placed = len(component_points)
            where = (
                f"class {_class_name(class_id)} ({class_id}) component "
                f"{component.component_index} in {component.site}/{component.stem}"
            )
            if n_placed < 1 or n_placed > quota:
                raise ProbeSelectionError(f"{where} placed {n_placed} pixels for quota {quota}")
            key = (class_id, component.site, component.stem, component.component_index)
            recorded = unmatched_shortfalls.get(key)
            if n_placed < quota:
                if (
                    recorded is None
                    or recorded["placed"] != n_placed
                    or recorded["quota"] != quota
                    or recorded["component_size"] != component.size
                ):
                    raise ProbeSelectionError(
                        f"{where} placed {n_placed} of quota {quota} without a matching shortfall"
                    )
                unmatched_shortfalls.pop(key)
            elif recorded is not None:
                raise ProbeSelectionError(f"{where} met quota {quota} but a shortfall was recorded")
            for point in component_points:
                pixel = (component.image_index, int(point["row"]), int(point["col"]))
                if pixel in seen:
                    raise ProbeSelectionError(f"{where} repeated pixel {pixel[1:]}")
                seen.add(pixel)
            placed_counts.append(n_placed)
        quotas = [quota for _component, quota in class_assignments]
        if sum(quotas) != per_class:
            raise ProbeSelectionError(
                f"class {_class_name(class_id)} ({class_id}) quotas sum to {sum(quotas)}, "
                f"expected {per_class}"
            )
        n_points = len(seen)
        if n_points != sum(placed_counts) or n_points < 1:
            raise ProbeSelectionError(
                f"class {_class_name(class_id)} ({class_id}) produced {n_points} distinct pixels "
                f"from placed counts {placed_counts}"
            )
        points_per_class[class_id] = n_points
        detail[str(class_id)] = {
            "class_name": _class_name(class_id),
            "n_components": len(by_class.get(class_id, [])),
            "n_points": n_points,
            "points_shortfall": per_class - n_points,
            "multi_sampled": any(quota > 1 for quota in quotas),
            "quotas": quotas,
            "placed": placed_counts,
        }
    if unmatched_shortfalls:
        raise ProbeSelectionError(
            f"{len(unmatched_shortfalls)} shortfall record(s) did not match a component"
        )

    ordered_image_indexes = sorted(
        used_image_indexes, key=lambda image_index: (records[image_index].site, records[image_index].stem)
    )
    images_out: list[dict] = []
    total_points = 0
    for image_index in ordered_image_indexes:
        record = records[image_index]
        label = _load_label(Path(record.path))
        points = sorted(
            image_points[image_index],
            key=lambda point: (point["class_id"], point["row"], point["col"], point["component_index"]),
        )
        total_points += len(points)
        images_out.append(
            {
                "site": record.site,
                "stem": record.stem,
                "label_md5": _md5(label),
                "height": EXPECTED_HEIGHT,
                "width": EXPECTED_WIDTH,
                "points": points,
            }
        )

    shortfall_points = sum(item["quota"] - item["placed"] for item in placement_shortfalls)
    expected_total = NUM_CLASSES * per_class - shortfall_points
    if total_points != expected_total or total_points != sum(points_per_class.values()):
        raise ProbeSelectionError(
            f"total points {total_points} != {expected_total} "
            f"({NUM_CLASSES} classes * {per_class} minus {shortfall_points} shortfall pixels)"
        )
    for item in placement_shortfalls:
        logger.warning(
            "shortfall: %s (%d) %s/%s component %d size %d quota %d placed %d",
            item["class_name"],
            item["class_id"],
            item["site"],
            item["stem"],
            item["component_index"],
            item["component_size"],
            item["quota"],
            item["placed"],
        )

    command = (
        "uv run python scripts/build_coralscapes_v2_probe_points.py "
        f"--local-root {local_root} "
        f"--output {output} "
        f"--per-class {per_class} "
        f"--min-component-size {min_component_size} "
        f"--exclusion-radius {exclusion_radius}"
    )
    payload = {
        "dataset": "josauder/coralscapesV2",
        "split": "train",
        "algorithm": "full_train_component_census_v2",
        "num_classes": NUM_CLASSES,
        "points_per_class_target": per_class,
        "min_component_size": min_component_size,
        "connectivity": 8,
        "exclusion_radius_px": exclusion_radius,
        "exclusion_rule": (
            "When a component's quota is 1, the probe pixel is the distance-transform "
            "maximum and no disk is removed. When the quota is greater than 1, each "
            "pixel is the distance-transform maximum of the remaining mask, then every "
            "pixel with dy**2 + dx**2 <= radius**2 is set False before the next pick. "
            "If that disk empties the component before the quota is met, the pixels "
            "already placed are kept and the unused quota is not reassigned. The class "
            "then has fewer than points_per_class_target points; see placement_shortfalls."
        ),
        "component_selection": (
            "Components are 8-connected and smaller than min_component_size are dropped. "
            "If a class has at least points_per_class_target components, that many are "
            "chosen by round-robin across images ordered by largest component, one point "
            "each. If it has fewer, every component is used and the target is split evenly, "
            "with the remainder given to the largest components."
        ),
        "image_height": EXPECTED_HEIGHT,
        "image_width": EXPECTED_WIDTH,
        "source": f"local_mirror:{local_root}",
        "command": command,
        "id2label": {str(class_id): name for class_id, name in CORALSCAPES_V2_ID2NAME.items()},
        "points_per_class": {str(class_id): int(n) for class_id, n in points_per_class.items()},
        "points_per_class_detail": detail,
        "placement_shortfalls": placement_shortfalls,
        "num_images": len(images_out),
        "total_points": total_points,
        "images": images_out,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    n_multi = sum(1 for item in detail.values() if item["multi_sampled"])
    logger.info(
        "Wrote %d images, %d points (%d classes multi-sampled, %d shortfall components) to %s",
        len(images_out),
        total_points,
        n_multi,
        len(placement_shortfalls),
        output,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("/Users/jonathan/mit/coralscapes_v2"),
        help="Cityscapes-style CoralscapesV2 mirror root (leftImg8bit + gtFine)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/eval/coralscapes_v2_probe_points.json"),
    )
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--min-component-size", type=int, default=4)
    parser.add_argument("--exclusion-radius", type=int, default=16)
    args = parser.parse_args()
    build(
        local_root=args.local_root,
        output=args.output,
        per_class=args.per_class,
        min_component_size=args.min_component_size,
        exclusion_radius=args.exclusion_radius,
    )


if __name__ == "__main__":
    main()
