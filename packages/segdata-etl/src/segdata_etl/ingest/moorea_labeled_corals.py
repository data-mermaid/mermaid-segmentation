"""Moorea Labeled Corals ingestion: validate a manually downloaded tree and publish to S3.

The EDI ``knb-lter-mcr.5006`` package requires authentication, so nothing is downloaded.
Raw layout (``--data-dir``)::

    <root>/<year>/<image>.<jpg|jpeg|png|tif|tiff>
    <root>/<year>/<image>.<ext>.txt      # one "row;col;label" line per point
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.images import find_undecodable
from ..common.labels import build_classes_and_colors
from ..common.manifest import IngestSource, build_manifest
from ..common.release import S3Target, preflight, put_raw_release, upload_files

logger = logging.getLogger(__name__)

DATASET_NAME = "moorea_labeled_corals"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# Points labelled OFF (any case) fell outside the annotatable region of the photo.
_OFF_LABEL_TOKEN = "off"

_COLUMNS = ["year", "image_id", "image_ext", "row", "col", "source_label_name", "raw_label_name"]
_SORT_KEYS = ["year", "image_id", "row", "col"]
PARQUET_COLUMNS = {
    "year": "Survey year (string), the name of the raw year directory.",
    "image_id": "Image filename stem, without the extension.",
    "image_ext": "Image filename extension including the dot (e.g. '.jpg').",
    "row": "Annotation pixel row (y, integer).",
    "col": "Annotation pixel column (x, integer).",
    "source_label_name": "Case-merged label name (matches keys in classes.json).",
    "raw_label_name": "Label exactly as spelled in the raw annotation file.",
}

SOURCE = IngestSource(
    citation="Beijbom et al., 'Automated Annotation of Coral Reef Survey Images', IEEE CVPR, 2012.",
    raw_data_url="https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006",
    license="CC BY 4.0",
)

_Pair = tuple[str, Path, Path]


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Register the Moorea Labeled Corals flags; ``workdir`` is unused (manual download)."""
    del workdir
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Unzipped EDI knb-lter-mcr.5006 package, laid out as <year>/<image> + <image>.txt.",
    )


def collect_pairs(raw_root: Path) -> list[_Pair]:
    """Return ``(year, image_path, annotation_path)`` for every image under ``raw_root``."""
    pairs: list[_Pair] = []
    for year_dir in sorted(d for d in raw_root.iterdir() if d.is_dir() and d.name.isdigit()):
        for image_path in sorted(year_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMG_EXTS:
                continue
            ann_path = image_path.with_name(image_path.name + ".txt")
            if not ann_path.exists():
                raise FileNotFoundError(f"Missing annotation file for image: {image_path}")
            pairs.append((year_dir.name, image_path, ann_path))
    if not pairs:
        raise FileNotFoundError(f"No <year>/<image> pairs under {raw_root}")
    return pairs


def parse_annotations(txt_path: Path) -> list[tuple[int, int, str]]:
    """Parse a ``row;col;label`` file, skipping ``#`` comments and malformed rows."""
    rows: list[tuple[int, int, str]] = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in line.strip().split(";")]
        if len(parts) != 3 or line.strip().startswith("#"):
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            logger.warning("Skipping malformed row in %s: %r", txt_path, line)
    return rows


def canonicalize(labels: Iterable[str]) -> dict[str, str]:
    """Case-merge label spellings; prefer mixed case over all-caps, then lexicographic order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        groups[label.lower()].append(label)
    out: dict[str, str] = {}
    for variants in groups.values():
        mixed = [v for v in variants if v != v.upper()]
        canonical = sorted(mixed)[0] if mixed else sorted(variants)[0]
        for v in variants:
            out[v] = canonical
    return out


def _find_damaged(pairs: list[_Pair], *, workers: int) -> tuple[list[_Pair], list[str]]:
    """Split ``pairs`` into decodable pairs and ``<year>/<filename>`` names of the rest."""
    bad = {path for path, _ in find_undecodable([p[1] for p in pairs], workers=workers)}
    valid = [p for p in pairs if p[1] not in bad]
    damaged = [f"{year}/{path.name}" for year, path, _ in pairs if path in bad]
    return valid, damaged


def _build_annotations(pairs: list[_Pair]) -> tuple[pd.DataFrame, dict[str, str], int]:
    """Return the annotations frame, the raw-to-canonical label map, and the OFF-point count."""
    raw_rows: list[tuple[str, str, str, int, int, str]] = []
    n_off = 0
    for year, image_path, ann_path in pairs:
        for row, col, label in parse_annotations(ann_path):
            if label.lower() == _OFF_LABEL_TOKEN:
                n_off += 1
                continue
            raw_rows.append((year, image_path.stem, image_path.suffix, row, col, label))
    df = pd.DataFrame(
        raw_rows, columns=["year", "image_id", "image_ext", "row", "col", "raw_label_name"]
    )
    df = df.astype({"row": "int64", "col": "int64"})
    raw_to_canonical = canonicalize(set(df["raw_label_name"]))
    df["source_label_name"] = df["raw_label_name"].map(raw_to_canonical)
    return df[_COLUMNS], raw_to_canonical, n_off


def run(args: argparse.Namespace, client: Any) -> int:
    """Ingest Moorea Labeled Corals into S3; return the process exit code."""
    target = S3Target(args.bucket, args.prefix.strip("/"), args.dry_run)
    preflight(client, target, DATASET_NAME, overwrite=args.overwrite)
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")

    pairs = collect_pairs(data_dir)
    valid, damaged = _find_damaged(pairs, workers=args.workers)
    for name in damaged:
        logger.warning("Dropping damaged image %s and its annotations", name)
    annotations, raw_to_canonical, n_off = _build_annotations(valid)
    merges = {raw: canon for raw, canon in raw_to_canonical.items() if raw != canon}
    classes, colors = build_classes_and_colors(annotations["source_label_name"])

    uploads = [(path, f"{target.prefix}/images/{year}/{path.name}") for year, path, _ in valid]
    uploaded = upload_files(
        client, target, uploads, workers=args.workers, skip_existing=args.skip_existing_s3
    )
    manifest = build_manifest(
        DATASET_NAME,
        SOURCE,
        target,
        image_key_template=f"{target.prefix}/images/{{year}}/{{image_id}}{{image_ext}}",
        parquet_columns=PARQUET_COLUMNS,
        dropped={"damaged_images": len(damaged), "off_annotations": n_off},
        images_uploaded=uploaded,
        details={"damaged_images": damaged, "case_merged_labels": merges},
    )
    put_raw_release(
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
