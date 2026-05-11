"""Verify a manually downloaded Moorea Labeled Corals dataset and upload to S3.

The EDI ``knb-lter-mcr.5006`` package requires authentication, so this script
does not download anything. Point ``--data-dir`` at the unzipped package
(see ``README.md``); the script then validates each image, parses the
``<image>.txt`` ``row;col;label`` annotations (case-merging label spellings
like ``ACROPORA`` -> ``Acropora`` and dropping ``OFF`` out-of-region points),
drops annotations of damaged images, and writes everything to S3 as a Parquet
plus ``classes.json`` / ``colors.json`` / ``manifest.json``.

Run::

    python -m mermaidseg.datasets.moorea_labeled_corals.verify_raw_data_and_add_to_s3 \
        --data-dir /path/to/moorea_labeled_corals_downloaded
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

EDI_URL = "https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006"
CITATION = "Beijbom et al., 'Automated Annotation of Coral Reef Survey Images', IEEE CVPR, 2012."
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PARQUET_NAME = "moorea_labeled_corals_annotations.parquet"

# "OFF" / "Off" / "off" marks a point that fell outside the annotatable region
# of the quadrat photo and is dropped entirely from the dataset.
OFF_LABEL_TOKEN = "off"


def collect_pairs(raw_root: Path) -> list[tuple[str, Path, Path]]:
    """Return ``(year, image_path, annotation_path)`` for every image under ``raw_root``."""
    pairs: list[tuple[str, Path, Path]] = []
    for year_dir in sorted(d for d in raw_root.iterdir() if d.is_dir() and d.name.isdigit()):
        for image_path in sorted(year_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMG_EXTS:
                continue
            ann_path = image_path.with_name(image_path.name + ".txt")
            if not ann_path.exists():
                raise FileNotFoundError(f"Missing annotation file for image: {image_path}")
            pairs.append((year_dir.name, image_path, ann_path))
    if not pairs:
        raise FileNotFoundError(f"No <year>/<image> pairs under {raw_root} (see README).")
    return pairs


def parse_annotations(txt_path: Path) -> list[tuple[int, int, str]]:
    """Parse a ``row;col;label`` annotation file."""
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


def is_valid_image(path: Path) -> bool:
    """True iff ``path`` exists and PIL can fully decode it."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False
    return True


def canonicalize(labels: Iterable[str]) -> dict[str, str]:
    """Case-merge label spellings; prefer mixed-case over all-caps, then lexicographic."""
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


def build_palette(label_names: list[str], seed: int = 1337) -> dict[str, list[int]]:
    """Deterministic RGB palette keyed by label name; ``unlabeled`` is black."""
    rng = np.random.default_rng(seed)
    return {"unlabeled": [0, 0, 0]} | {
        n: [int(x) for x in rng.integers(0, 256, size=3)] for n in label_names
    }


def upload_one(args: tuple[str, Path, str]) -> None:
    bucket, local, key = args
    boto3.client("s3").upload_file(str(local), bucket, key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a manually downloaded Moorea Labeled Corals dataset and upload to S3.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--prefix", default="external_validation_datasets/moorea_labeled_corals")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger.info(
        "Raw data must come from: %s (see README.md for the expected directory layout).",
        EDI_URL,
    )

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")
    prefix = args.prefix.strip("/")

    pairs = collect_pairs(data_dir)
    logger.info("Found %d image+annotation pairs.", len(pairs))

    logger.info("Validating images (PIL decode)...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        valid_flags = list(
            tqdm(
                pool.map(is_valid_image, [p for _, p, _ in pairs]),
                total=len(pairs),
                desc="validate",
            )
        )
    valid_pairs: list[tuple[str, Path, Path]] = []
    damaged_images: list[str] = []
    for (year, img, ann), ok in zip(pairs, valid_flags, strict=False):
        if ok:
            valid_pairs.append((year, img, ann))
        else:
            damaged_images.append(f"{year}/{img.name}")
    if damaged_images:
        logger.warning(
            "Skipping %d damaged/missing image(s); their annotations will be dropped.",
            len(damaged_images),
        )

    raw_rows: list[tuple[str, str, str, int, int, str]] = []
    raw_label_set: set[str] = set()
    n_off_dropped = 0
    for year, image_path, ann_path in valid_pairs:
        for r, c, label in parse_annotations(ann_path):
            if label.strip().lower() == OFF_LABEL_TOKEN:
                n_off_dropped += 1
                continue
            raw_rows.append((year, image_path.stem, image_path.suffix, r, c, label))
            raw_label_set.add(label)
    if n_off_dropped:
        logger.info("Dropped %d 'OFF' annotation(s) (out-of-region points).", n_off_dropped)

    raw_to_canonical = canonicalize(raw_label_set)
    df = pd.DataFrame(
        raw_rows, columns=["year", "image_id", "image_ext", "row", "col", "raw_label_name"]
    )
    df["source_label_name"] = df["raw_label_name"].map(raw_to_canonical)
    df = df[["year", "image_id", "image_ext", "row", "col", "source_label_name", "raw_label_name"]]
    label_names = sorted(df["source_label_name"].unique())
    classes = {"unlabeled": 0} | {n: i + 1 for i, n in enumerate(label_names)}
    colors = build_palette(label_names)
    merges = {k: v for k, v in raw_to_canonical.items() if k != v}
    if merges:
        logger.info("Case-merged %d label spelling(s): %s", len(merges), merges)

    s3 = boto3.client("s3")
    parquet_uri = f"s3://{args.bucket}/{prefix}/{PARQUET_NAME}"
    logger.info("Writing %d annotation rows -> %s", len(df), parquet_uri)
    df.to_parquet(parquet_uri, engine="pyarrow", index=False)

    upload_tasks = [(args.bucket, p, f"{prefix}/images/{y}/{p.name}") for y, p, _ in valid_pairs]
    logger.info("Uploading %d images with %d workers...", len(upload_tasks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm(pool.map(upload_one, upload_tasks), total=len(upload_tasks), desc="upload"))

    manifest = {
        "dataset_name": "moorea_labeled_corals",
        "source": {"paper": CITATION, "raw_data_url": EDI_URL, "license": "CC BY 4.0"},
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "s3": {
            "bucket": args.bucket,
            "prefix": prefix,
            "annotations_parquet": f"{prefix}/{PARQUET_NAME}",
            "image_key_template": f"{prefix}/images/{{year}}/{{image_id}}{{image_ext}}",
        },
        "damaged_images": damaged_images,
        "case_merged_labels": merges,
        "off_annotations_dropped": n_off_dropped,
    }
    for name, payload in [
        ("classes.json", classes),
        ("colors.json", colors),
        ("manifest.json", manifest),
    ]:
        s3.put_object(
            Bucket=args.bucket,
            Key=f"{prefix}/{name}",
            Body=json.dumps(payload, indent=2).encode(),
            ContentType="application/json",
        )

    logger.info("S3 prefix: s3://%s/%s/", args.bucket, prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
