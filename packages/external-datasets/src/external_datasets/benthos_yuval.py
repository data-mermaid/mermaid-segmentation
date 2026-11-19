"""Benthos (Yuval et al., 2021) ingestion: tile two manually downloaded mosaics, publish to S3.

The Dryad download is OAuth-protected, so ``--data-dir`` must already hold the
:data:`REQUIRED_FILES`. The coral-reef mosaics ``RS24`` and ``CR_DoubleWreck`` are tiled into
2048x2048 PNG image and label pairs; the Mediterranean ``MD_spartan`` is excluded.
"""

from __future__ import annotations

import argparse
import ast
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .labels import build_classes_and_colors
from .manifest import Source, build_manifest
from .publish import S3Target, preflight, publish_dataset, upload_files

logger = logging.getLogger(__name__)

DATASET_NAME = "benthos_yuval"
DRYAD_DATASET_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.8cz8w9gm3"
CITATION = (
    "Yuval, M., Alonso, I., Murillo, A. C., Treibitz, T. (2021). "
    "Repeatable semantic reef-mapping through photogrammetry and "
    "label-augmentation. Dryad. doi:10.5061/dryad.8cz8w9gm3"
)
TILE = 2048
EXCLUDED_SITES = ["MD_spartan"]

# Site -> raw image file, raw mask file and the label-dictionary variant its mask ids use.
MOSAICS: dict[str, dict[str, str]] = {
    "RS24": {"image": "RS24.png", "mask": "ManualRS24.png", "label_variant": "default"},
    "CR_DoubleWreck": {
        "image": "CR_DoubleWreck.png",
        "mask": "ManualDW.png",
        "label_variant": "alt",
    },
}
LABEL_DICTIONARY_FILES = {
    "default": "dictionary_labels.txt",
    "alt": "dictionary_labelsDW.txt",
}
REQUIRED_FILES: tuple[str, ...] = (
    "RS24.png",
    "ManualRS24.png",
    "CR_DoubleWreck.png",
    "ManualDW.png",
    "dictionary_labels.txt",
    "dictionary_labelsDW.txt",
)

SOURCE = Source(
    name=DATASET_NAME,
    citation=CITATION,
    raw_data_url=DRYAD_DATASET_URL,
    license="CC0 1.0 (Dryad)",
)

PARQUET_COLUMNS = {
    "site": "Mosaic site ('RS24' or 'CR_DoubleWreck').",
    "image_id": "Tile ID '<y0>_<x0>': pixel offset of the tile's top-left corner in the mosaic.",
    "source_label_name": "Benthos label name (matches keys in classes.json, incl. 'unlabeled').",
    "tile_height": "Tile height in pixels (int32).",
    "tile_width": "Tile width in pixels (int32).",
}
_ANNOTATION_COLUMNS = ["site", "image_id", "source_label_name", "tile_height", "tile_width"]
_SORT_KEYS = ["site", "image_id", "source_label_name"]

_Row = tuple[str, str, str, int, int]


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Register the Benthos Yuval flags."""
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory with the 6 manually downloaded Dryad files (see REQUIRED_FILES). "
            f"Download from: {DRYAD_DATASET_URL}"
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=workdir,
        help="Where the tiles are written locally before upload.",
    )


@contextmanager
def _decompression_limit_lifted() -> Iterator[None]:
    """Lift Pillow's process-wide decompression-bomb limit, restoring it on exit."""
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # the mosaics exceed the default limit
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def _load_image_rgb(path: Path) -> np.ndarray:
    with _decompression_limit_lifted():
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask_u8(path: Path) -> np.ndarray:
    with _decompression_limit_lifted():
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _load_label_dictionary(path: Path) -> dict[str, int]:
    parsed = ast.literal_eval(path.read_text())
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Label dictionary at {path} did not parse to a dict")
    return {str(k): int(v) for k, v in parsed.items()}


def load_label_dictionaries(data_root: Path) -> dict[str, dict[str, int]]:
    """Check the required files exist and return ``{variant: {class_name: raw_id}}``."""
    missing = [name for name in REQUIRED_FILES if not (data_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required raw files under {data_root}: {missing}. "
            "Manually download these from the Dryad dataset and pass their "
            "directory via --data-dir."
        )
    return {
        variant: _load_label_dictionary(data_root / fname)
        for variant, fname in LABEL_DICTIONARY_FILES.items()
    }


def _build_raw_to_global_lut(
    variant_dict: dict[str, int], classes_global: dict[str, int]
) -> np.ndarray:
    """Return a uint8 LUT from a site's raw mask ids to classes.json ids; unknown ids map to 0."""
    lut = np.zeros(256, dtype=np.uint8)
    for name, raw_id in variant_dict.items():
        lut[int(raw_id)] = np.uint8(classes_global.get(name, 0))
    return lut


def _tile_site(
    site: str,
    image: np.ndarray,
    mask: np.ndarray,
    tiles_root: Path,
    id_to_name: dict[int, str],
) -> tuple[list[tuple[Path, str]], list[_Row]]:
    """Write ``site``'s tile PNGs; return ``(files, annotation rows)``.

    ``files`` holds ``(local_path, key suffix)`` pairs; image and mask are cropped to their
    common size first.
    """
    h = min(image.shape[0], mask.shape[0])
    w = min(image.shape[1], mask.shape[1])
    image, mask = image[:h, :w], mask[:h, :w]

    image_dir = tiles_root / "images" / site
    label_dir = tiles_root / "labels" / site
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[Path, str]] = []
    rows: list[_Row] = []
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            crop_im = np.ascontiguousarray(image[y0:y1, x0:x1])
            crop_mk = np.ascontiguousarray(mask[y0:y1, x0:x1])
            image_id = f"{y0}_{x0}"
            im_path = image_dir / f"{image_id}.png"
            mk_path = label_dir / f"{image_id}.png"
            Image.fromarray(crop_im).save(im_path)
            Image.fromarray(crop_mk, mode="L").save(mk_path)
            files.append((im_path, f"images/{site}/{image_id}.png"))
            files.append((mk_path, f"labels/{site}/{image_id}.png"))
            for gid in np.unique(crop_mk).tolist():
                name = id_to_name.get(int(gid))
                if name is not None:
                    rows.append((site, image_id, name, int(y1 - y0), int(x1 - x0)))
    return files, rows


def _annotations_frame(rows: list[_Row]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_ANNOTATION_COLUMNS)
    return df.astype(
        {
            "site": str,
            "image_id": str,
            "source_label_name": str,
            "tile_height": "int32",
            "tile_width": "int32",
        }
    )


def tile_mosaics(
    data_root: Path,
    tiles_root: Path,
    classes_global: dict[str, int],
    label_dictionaries: dict[str, dict[str, int]],
) -> tuple[list[tuple[Path, str]], pd.DataFrame]:
    """Tile each mosaic, decoding one at a time; return ``(files, annotations)``."""
    id_to_name = {v: k for k, v in classes_global.items()}
    files: list[tuple[Path, str]] = []
    rows: list[_Row] = []
    for site, info in MOSAICS.items():
        logger.info("Tiling site %s ...", site)
        image = _load_image_rgb(data_root / info["image"])
        mask_raw = _load_mask_u8(data_root / info["mask"])
        lut = _build_raw_to_global_lut(label_dictionaries[info["label_variant"]], classes_global)
        site_files, site_rows = _tile_site(site, image, lut[mask_raw], tiles_root, id_to_name)
        files.extend(site_files)
        rows.extend(site_rows)
        logger.info("Site %s: produced %d tiles.", site, len(site_files) // 2)
    return files, _annotations_frame(rows)


def _split_uploads(
    files: list[tuple[Path, str]], prefix: str
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Split ``(path, key suffix)`` pairs into ``(images, labels)`` ``(path, full key)`` lists."""
    images = [(p, f"{prefix}/{s}") for p, s in files if s.startswith("images/")]
    labels = [(p, f"{prefix}/{s}") for p, s in files if s.startswith("labels/")]
    return images, labels


def run(args: argparse.Namespace, client: Any) -> int:
    """Tile the Benthos Yuval mosaics and publish them to S3; return the exit code."""
    target = S3Target(args.bucket, args.prefix.strip("/"), args.dry_run)
    preflight(client, target, DATASET_NAME, overwrite=args.overwrite)
    data_root = args.data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-dir does not exist: {data_root}")

    label_dictionaries = load_label_dictionaries(data_root)
    names: set[str] = set().union(*(d.keys() for d in label_dictionaries.values()))
    classes, colors = build_classes_and_colors(names)

    tiles_root = args.workdir.expanduser().resolve() / "tiles"
    files, annotations = tile_mosaics(data_root, tiles_root, classes, label_dictionaries)
    num_tiles = {
        site: sum(1 for _, suffix in files if suffix.startswith(f"images/{site}/"))
        for site in MOSAICS
    }

    image_uploads, label_uploads = _split_uploads(files, target.prefix)
    images_uploaded = upload_files(
        client, target, image_uploads, workers=args.workers, skip_existing=args.skip_existing_s3
    )
    # Label masks are derived from the class mapping, so --overwrite must refresh them.
    labels_uploaded = upload_files(
        client,
        target,
        label_uploads,
        workers=args.workers,
        skip_existing=args.skip_existing_s3 and not args.overwrite,
    )
    manifest = build_manifest(
        SOURCE,
        target,
        image_key_template=f"{target.prefix}/images/{{site}}/{{image_id}}.png",
        parquet_columns=PARQUET_COLUMNS,
        ingestion={
            "excluded_sites": EXCLUDED_SITES,
            "tile_size": TILE,
            "label_key_template": f"{target.prefix}/labels/{{site}}/{{image_id}}.png",
            "num_tiles": num_tiles,
            "images_uploaded": images_uploaded,
            "labels_uploaded": labels_uploaded,
        },
    )
    publish_dataset(
        client,
        target,
        dataset_name=DATASET_NAME,
        annotations=annotations,
        sort_by=_SORT_KEYS,
        classes=classes,
        colors=colors,
        manifest=manifest,
        overwrite=args.overwrite,
    )
    return 0
