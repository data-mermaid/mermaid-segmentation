"""Build a fixed CoralscapesV2 linear-probe point set.

Finds a small set of *training* images that jointly contain all 95 CoralscapesV2
classes, and for each class selects up to 5 probe pixels taken from different
connected components (8-connectivity) when possible. Each pixel is the interior
(distance-transform maximum) of its component, so it survives the model's ~4x
feature downsample. The result is written as a JSON file that pins the probe set
for reproducible, comparable evaluations across models/methods.

The builder reads a Cityscapes-style local mirror of ``josauder/coralscapesV2``
(``leftImg8bit`` + ``gtFine/*_labelIds.png``). Each chosen image records its
``site``, ``stem`` and the md5 of its label array so the evaluation can locate
the same images either from the local mirror (by stem) or from the HuggingFace
dataset (by label md5).

Usage:
    python scripts/build_coralscapes_v2_probe_points.py \
        --local-root /Users/jonathan/mit/coralscapes_v2 \
        --output configs/eval/coralscapes_v2_probe_points.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 95  # native ids 1..95 (0 = background/ignore)
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _label_paths(root: Path, split: str) -> list[Path]:
    return sorted((root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))


def _stem_from_label_path(p: Path) -> tuple[str, str]:
    site = p.parent.name
    stem = p.name.replace("_gtFine_labelIds.png", "")
    return site, stem


def _load_label(p: Path) -> np.ndarray:
    arr = np.asarray(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def _interior_point(mask: np.ndarray) -> tuple[int, int]:
    """Return the (row, col) at the maximum of the component's distance transform."""
    dt = ndimage.distance_transform_edt(mask)
    idx = int(np.argmax(dt))
    r, c = np.unravel_index(idx, mask.shape)
    return int(r), int(c)


def first_pass(label_paths: list[Path]) -> list[dict]:
    """Per-image class presence + pixel counts (no arrays retained)."""
    records: list[dict] = []
    for i, p in enumerate(label_paths):
        arr = _load_label(p)
        counts = np.bincount(arr.reshape(-1), minlength=NUM_CLASSES + 1)
        present = {int(c) for c in np.nonzero(counts)[0] if 1 <= int(c) <= NUM_CLASSES}
        site, stem = _stem_from_label_path(p)
        records.append(
            {
                "path": str(p),
                "site": site,
                "stem": stem,
                "present": present,
                "counts": {int(c): int(counts[c]) for c in present},
                "height": int(arr.shape[0]),
                "width": int(arr.shape[1]),
            }
        )
        if (i + 1) % 200 == 0:
            logger.info("  first pass %d/%d", i + 1, len(label_paths))
    return records


def greedy_cover(records: list[dict]) -> list[int]:
    """Greedy set cover of all present classes (returns chosen record indices)."""
    all_present: set[int] = set()
    for r in records:
        all_present |= r["present"]
    missing = set(range(1, NUM_CLASSES + 1)) - all_present
    if missing:
        logger.warning(
            "%d classes never appear in train and cannot be probed: %s",
            len(missing),
            sorted(missing),
        )
    uncovered = set(all_present)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    while uncovered:
        best_i, best_gain, best_px = -1, -1, -1
        for i, r in enumerate(records):
            if i in chosen_set:
                continue
            gain = len(r["present"] & uncovered)
            if gain == 0:
                continue
            px = sum(r["counts"][c] for c in (r["present"] & uncovered))
            if gain > best_gain or (gain == best_gain and px > best_px):
                best_i, best_gain, best_px = i, gain, px
        if best_i < 0:
            break
        chosen.append(best_i)
        chosen_set.add(best_i)
        uncovered -= records[best_i]["present"]
    return chosen


def top_up(records: list[dict], chosen: list[int], per_class: int, max_images: int) -> list[int]:
    """Add images so each class is present in up to ``per_class`` chosen images."""
    chosen_set = set(chosen)
    # count chosen images per class
    imgs_per_class: dict[int, int] = defaultdict(int)
    for i in chosen:
        for c in records[i]["present"]:
            imgs_per_class[c] += 1

    for c in range(1, NUM_CLASSES + 1):
        while imgs_per_class[c] < per_class and len(chosen_set) < max_images:
            candidates = [
                (records[i]["counts"][c], i)
                for i in range(len(records))
                if i not in chosen_set and c in records[i]["present"]
            ]
            if not candidates:
                break
            candidates.sort(reverse=True)
            _, best_i = candidates[0]
            chosen.append(best_i)
            chosen_set.add(best_i)
            for cc in records[best_i]["present"]:
                imgs_per_class[cc] += 1
    return chosen


def collect_components(
    records: list[dict], chosen: list[int], min_component_size: int
) -> dict[int, list[dict]]:
    """For each class, gather connected components across chosen images."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for n, i in enumerate(chosen):
        arr = _load_label(Path(records[i]["path"]))
        for c in records[i]["present"]:
            cmask = arr == c
            lab, ncomp = ndimage.label(cmask, structure=_STRUCT8)
            if ncomp == 0:
                continue
            sizes = np.bincount(lab.reshape(-1))
            for comp_id in range(1, ncomp + 1):
                size = int(sizes[comp_id])
                if size < min_component_size:
                    continue
                comp = lab == comp_id
                r, col = _interior_point(comp)
                by_class[c].append(
                    {
                        "image_idx": i,
                        "size": size,
                        "row": r,
                        "col": col,
                    }
                )
        if (n + 1) % 10 == 0:
            logger.info("  component pass %d/%d", n + 1, len(chosen))
    return by_class


def select_points(
    by_class: dict[int, list[dict]], per_class: int
) -> dict[int, list[dict]]:
    """Pick up to ``per_class`` components per class, prioritising image diversity + size."""
    selected: dict[int, list[dict]] = {}
    for c, comps in by_class.items():
        groups: dict[int, list[dict]] = defaultdict(list)
        for comp in comps:
            groups[comp["image_idx"]].append(comp)
        for g in groups.values():
            g.sort(key=lambda d: d["size"], reverse=True)
        # order images by their largest component (desc) for stable round-robin
        image_order = sorted(groups.keys(), key=lambda im: groups[im][0]["size"], reverse=True)
        picked: list[dict] = []
        while len(picked) < per_class:
            progressed = False
            for im in image_order:
                if groups[im]:
                    picked.append(groups[im].pop(0))
                    progressed = True
                    if len(picked) >= per_class:
                        break
            if not progressed:
                break
        selected[c] = picked
    return selected


def build(
    local_root: Path,
    output: Path,
    per_class: int,
    max_images: int,
    min_component_size: int,
) -> None:
    label_paths = _label_paths(local_root, "train")
    if not label_paths:
        raise FileNotFoundError(f"No train labelIds under {local_root}/gtFine/train")
    logger.info("Found %d train label masks under %s", len(label_paths), local_root)

    records = first_pass(label_paths)
    chosen = greedy_cover(records)
    logger.info("Greedy cover selected %d images", len(chosen))
    chosen = top_up(records, chosen, per_class=per_class, max_images=max_images)
    logger.info("After top-up: %d images (cap %d)", len(chosen), max_images)

    by_class = collect_components(records, chosen, min_component_size=min_component_size)
    selected = select_points(by_class, per_class=per_class)

    # Only keep images that actually contribute a probe point.
    used_image_idxs = sorted({p["image_idx"] for pts in selected.values() for p in pts})
    idx_to_local: dict[int, int] = {}
    images_out: list[dict] = []
    md5_cache: dict[int, str] = {}
    for local_i, gi in enumerate(used_image_idxs):
        arr = _load_label(Path(records[gi]["path"]))
        md5 = _md5(arr)
        md5_cache[gi] = md5
        idx_to_local[gi] = local_i
        images_out.append(
            {
                "site": records[gi]["site"],
                "stem": records[gi]["stem"],
                "label_md5": md5,
                "height": records[gi]["height"],
                "width": records[gi]["width"],
                "points": [],
            }
        )

    points_per_class: dict[int, int] = {}
    for c in range(1, NUM_CLASSES + 1):
        pts = selected.get(c, [])
        points_per_class[c] = len(pts)
        for p in pts:
            images_out[idx_to_local[p["image_idx"]]]["points"].append(
                {
                    "class_id": int(c),
                    "row": int(p["row"]),
                    "col": int(p["col"]),
                    "component_size": int(p["size"]),
                }
            )

    covered = sum(1 for c in range(1, NUM_CLASSES + 1) if points_per_class.get(c, 0) > 0)
    full5 = sum(1 for c in range(1, NUM_CLASSES + 1) if points_per_class.get(c, 0) >= per_class)
    total_points = sum(points_per_class.values())
    logger.info(
        "Classes covered: %d/%d  (with >=%d pts: %d)  images used: %d  total points: %d",
        covered,
        NUM_CLASSES,
        per_class,
        full5,
        len(images_out),
        total_points,
    )
    under = {c: n for c, n in points_per_class.items() if 0 < n < per_class}
    if under:
        logger.info("Classes with <%d probe points: %s", per_class, under)
    missing = [c for c in range(1, NUM_CLASSES + 1) if points_per_class.get(c, 0) == 0]
    if missing:
        logger.warning("Classes with NO probe points: %s", missing)

    payload = {
        "dataset": "josauder/coralscapesV2",
        "split": "train",
        "num_classes": NUM_CLASSES,
        "points_per_class_target": per_class,
        "min_component_size": min_component_size,
        "source": f"local_mirror:{local_root}",
        "id2label": {str(k): v for k, v in CORALSCAPES_V2_ID2NAME.items()},
        "points_per_class": {str(k): int(v) for k, v in points_per_class.items()},
        "num_images": len(images_out),
        "total_points": total_points,
        "images": images_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote probe set to %s", output)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--local-root",
        type=Path,
        default=Path("/Users/jonathan/mit/coralscapes_v2"),
        help="Cityscapes-style CoralscapesV2 mirror root (leftImg8bit + gtFine)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("configs/eval/coralscapes_v2_probe_points.json"),
    )
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--min-component-size", type=int, default=4)
    args = ap.parse_args()
    build(
        local_root=args.local_root,
        output=args.output,
        per_class=args.per_class,
        max_images=args.max_images,
        min_component_size=args.min_component_size,
    )


if __name__ == "__main__":
    main()
