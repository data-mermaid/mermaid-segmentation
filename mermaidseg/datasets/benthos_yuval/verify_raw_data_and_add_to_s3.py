"""Verify a manually downloaded Benthos Yuval dataset, tile it, and upload to S3.

The Dryad dataset ``doi:10.5061/dryad.8cz8w9gm3`` (Yuval et al., 2021) is
behind an OAuth-protected download, so this script does not auto-download.
Point ``--data-dir`` at a flat folder containing the 6 :data:`REQUIRED_FILES`;
the script tiles the two coral-reef mosaics (``RS24`` and ``CR_DoubleWreck``;
the Mediterranean ``MD_spartan`` is excluded) into 2048x2048 PNG image+label
pairs and uploads everything to S3. See ``README.md`` for full context.

Run::

    python -m mermaidseg.datasets.benthos_yuval.verify_raw_data_and_add_to_s3 \
        --data-dir /path/to/benthos_yuval_downloaded
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


DRYAD_DATASET_DOI = "10.5061/dryad.8cz8w9gm3"
DRYAD_DATASET_URL = f"https://datadryad.org/dataset/doi:{DRYAD_DATASET_DOI}"
CITATION = (
    "Yuval, M., Alonso, I., Murillo, A. C., Treibitz, T. (2021). "
    "Repeatable semantic reef-mapping through photogrammetry and "
    "label-augmentation. Dryad. doi:10.5061/dryad.8cz8w9gm3"
)

PARQUET_NAME = "benthos_yuval_annotations.parquet"
TILE = 2048

# Site -> (raw image filename, raw mask filename, label-dictionary variant).
# RS24 uses ``dictionary_labels.txt`` ("default"); CR_DoubleWreck uses
# ``dictionary_labelsDW.txt`` ("alt"). The Mediterranean ``MD_spartan`` is
# excluded because it is not a coral-reef ecosystem.
MOSAICS: dict[str, dict[str, str]] = {
    "RS24": {
        "image": "RS24.png",
        "mask": "ManualRS24.png",
        "label_variant": "default",
    },
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


def _load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask_u8(path: Path) -> np.ndarray:
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
    """Check required files exist and return ``{variant: {class_name: raw_id}}``."""
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


def build_classes_and_colors(
    label_dictionaries: dict[str, dict[str, int]],
    seed: int = 1337,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Build ``classes.json`` (alphabetical, ``unlabeled=0``) and a deterministic palette."""
    all_class_names = sorted(set().union(*[d.keys() for d in label_dictionaries.values()]))
    classes_global: dict[str, int] = {"unlabeled": 0}
    classes_global.update({name: i + 1 for i, name in enumerate(all_class_names)})

    rng = np.random.default_rng(seed)
    colors_global: dict[str, list[int]] = {"unlabeled": [0, 0, 0]}
    for name in all_class_names:
        colors_global[name] = [int(x) for x in rng.integers(0, 256, size=3)]
    return classes_global, colors_global


def _build_raw_to_global_lut(
    variant_dict: dict[str, int], classes_global: dict[str, int]
) -> np.ndarray:
    """uint8 LUT mapping the per-site raw mask IDs into the global classes.json IDs."""
    lut = np.zeros(256, dtype=np.uint8)
    for name, raw_id in variant_dict.items():
        lut[int(raw_id)] = np.uint8(classes_global.get(name, 0))
    return lut


def tile_mosaics(
    data_root: Path,
    workdir: Path,
    classes_global: dict[str, int],
    label_dictionaries: dict[str, dict[str, int]],
) -> tuple[list[tuple[str, str, Path, Path]], pd.DataFrame]:
    """Tile each mosaic into 2048x2048 PNG image/label pairs on local disk.

    Returns ``(tiles, df_annotations)`` where ``tiles`` is one
    ``(site, image_id, image_path, label_path)`` per saved tile (used for
    upload) and ``df_annotations`` is one row per (tile, present-class).
    """
    tiles_root = workdir / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)
    global_id_to_name: dict[int, str] = {v: k for k, v in classes_global.items()}

    tiles: list[tuple[str, str, Path, Path]] = []
    ann_rows: list[tuple[str, str, str, int, int]] = []
    for site, info in MOSAICS.items():
        logger.info("Tiling site %s ...", site)
        image = _load_image_rgb(data_root / info["image"])
        mask_raw = _load_mask_u8(data_root / info["mask"])
        lut = _build_raw_to_global_lut(label_dictionaries[info["label_variant"]], classes_global)
        mask = lut[mask_raw]

        h = min(image.shape[0], mask.shape[0])
        w = min(image.shape[1], mask.shape[1])
        image = image[:h, :w]
        mask = mask[:h, :w]

        image_dir = tiles_root / "images" / site
        label_dir = tiles_root / "labels" / site
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        site_count = 0
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                y1 = min(y0 + TILE, h)
                x1 = min(x0 + TILE, w)
                crop_im = np.ascontiguousarray(image[y0:y1, x0:x1])
                crop_mk = np.ascontiguousarray(mask[y0:y1, x0:x1])

                image_id = f"{y0}_{x0}"
                im_path = image_dir / f"{image_id}.png"
                mk_path = label_dir / f"{image_id}.png"
                Image.fromarray(crop_im).save(im_path)
                Image.fromarray(crop_mk, mode="L").save(mk_path)

                tile_h, tile_w = int(y1 - y0), int(x1 - x0)
                tiles.append((site, image_id, im_path, mk_path))
                site_count += 1
                for gid in np.unique(crop_mk).tolist():
                    name = global_id_to_name.get(int(gid))
                    if name is not None:
                        ann_rows.append((site, image_id, name, tile_h, tile_w))
        logger.info("Site %s: produced %d tiles.", site, site_count)

    df = pd.DataFrame(
        ann_rows,
        columns=["site", "image_id", "source_label_name", "tile_height", "tile_width"],
    )
    df = df.astype(
        {
            "site": str,
            "image_id": str,
            "source_label_name": str,
            "tile_height": "int32",
            "tile_width": "int32",
        }
    )
    return tiles, df


def list_existing_keys(s3, bucket: str, prefix: str) -> set[str]:
    """List every key under ``s3://bucket/prefix/``."""
    keys: set[str] = set()
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"
    ):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys


def upload_tiles(
    tiles: list[tuple[str, str, Path, Path]],
    bucket: str,
    images_prefix: str,
    labels_prefix: str,
    workers: int,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    """Upload tile PNGs to S3 in parallel. Returns the number of objects uploaded."""
    s3 = boto3.client("s3")

    tasks: list[tuple[Path, str]] = []
    for site, image_id, im_path, mk_path in tiles:
        tasks.append((im_path, f"{images_prefix}/{site}/{image_id}.png"))
        tasks.append((mk_path, f"{labels_prefix}/{site}/{image_id}.png"))

    if skip_existing and not dry_run:
        existing = list_existing_keys(s3, bucket, images_prefix) | list_existing_keys(
            s3, bucket, labels_prefix
        )
        logger.info("Found %d existing tile keys; skipping those.", len(existing))
        tasks = [(p, k) for p, k in tasks if k not in existing]

    if dry_run:
        logger.info("[dry-run] Would upload %d tile objects.", len(tasks))
        return len(tasks)
    if not tasks:
        logger.info("No new tile objects to upload.")
        return 0

    logger.info("Uploading %d tile objects with %d workers ...", len(tasks), workers)

    def _upload(local: Path, key: str) -> None:
        boto3.client("s3").upload_file(str(local), bucket, key)

    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=len(tasks), desc="upload tile", unit="obj") as pbar,
    ):
        futures = [pool.submit(_upload, local, key) for local, key in tasks]
        for fut in as_completed(futures):
            fut.result()
            pbar.update(1)
    return len(tasks)


def upload_json(s3, bucket: str, key: str, payload: dict, dry_run: bool) -> str:
    uri = f"s3://{bucket}/{key}"
    body = json.dumps(payload, indent=2).encode("utf-8")
    if dry_run:
        logger.info("[dry-run] Would upload %s (%d bytes)", uri, len(body))
        return uri
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Uploaded %s", uri)
    return uri


def build_manifest(
    bucket: str,
    prefix: str,
    classes: dict[str, int],
    num_tiles_per_site: dict[str, int],
    num_uploaded_this_run: int,
) -> dict:
    return {
        "dataset_name": "benthos_yuval",
        "source": {
            "citation": CITATION,
            "raw_data_url": DRYAD_DATASET_URL,
            "license": "CC0 1.0 (Dryad)",
            "excluded_sites": ["MD_spartan"],
        },
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "s3": {
            "bucket": bucket,
            "prefix": prefix,
            "annotations_parquet": f"{prefix}/{PARQUET_NAME}",
            "image_key_template": f"{prefix}/images/{{site}}/{{image_id}}.png",
            "label_key_template": f"{prefix}/labels/{{site}}/{{image_id}}.png",
        },
        "tile_size": TILE,
        "num_classes_incl_unlabeled": len(classes),
        "stats": {site: {"num_tiles": num_tiles_per_site.get(site, 0)} for site in MOSAICS},
        "num_uploaded_this_run": num_uploaded_this_run,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify, tile, and upload a locally-downloaded Benthos Yuval dataset to S3.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the 6 manually-downloaded Dryad files (see REQUIRED_FILES). "
            "Download from: https://datadryad.org/dataset/doi:10.5061/dryad.8cz8w9gm3"
        ),
    )
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--prefix", default="external_validation_datasets/benthos_yuval")
    parser.add_argument("--workdir", type=Path, default=Path("./benthos_yuval_workdir"))
    parser.add_argument(
        "--no-skip-existing-s3",
        dest="skip_existing_s3",
        action="store_false",
        help="By default, tile keys already in S3 are skipped; pass to re-upload.",
    )
    parser.add_argument("--upload-workers", type=int, default=16)
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute everything but skip S3 writes."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_root = args.data_dir.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_root}")
    prefix = args.prefix.strip("/")
    images_prefix = f"{prefix}/images"
    labels_prefix = f"{prefix}/labels"
    annotations_key = f"{prefix}/{PARQUET_NAME}"

    label_dictionaries = load_label_dictionaries(data_root)
    classes_global, colors_global = build_classes_and_colors(label_dictionaries)
    logger.info("Built classes.json with %d entries (incl. unlabeled).", len(classes_global))

    tiles, df_annotations = tile_mosaics(
        data_root=data_root,
        workdir=args.workdir.expanduser().resolve(),
        classes_global=classes_global,
        label_dictionaries=label_dictionaries,
    )
    num_tiles_per_site: dict[str, int] = dict.fromkeys(MOSAICS, 0)
    for site, *_ in tiles:
        num_tiles_per_site[site] += 1
    logger.info(
        "Produced %d total tiles; built annotations DataFrame with %d rows.",
        len(tiles),
        len(df_annotations),
    )

    parquet_uri = f"s3://{args.bucket}/{annotations_key}"
    if args.dry_run:
        logger.info(
            "[dry-run] Would write parquet (%d rows) to %s", len(df_annotations), parquet_uri
        )
    else:
        logger.info("Writing %d annotation rows -> %s", len(df_annotations), parquet_uri)
        df_annotations.to_parquet(parquet_uri, engine="pyarrow", index=False)

    num_uploaded = upload_tiles(
        tiles=tiles,
        bucket=args.bucket,
        images_prefix=images_prefix,
        labels_prefix=labels_prefix,
        workers=args.upload_workers,
        skip_existing=args.skip_existing_s3,
        dry_run=args.dry_run,
    )

    s3 = boto3.client("s3")
    classes_uri = upload_json(
        s3, args.bucket, f"{prefix}/classes.json", classes_global, args.dry_run
    )
    colors_uri = upload_json(s3, args.bucket, f"{prefix}/colors.json", colors_global, args.dry_run)
    manifest = build_manifest(
        bucket=args.bucket,
        prefix=prefix,
        classes=classes_global,
        num_tiles_per_site=num_tiles_per_site,
        num_uploaded_this_run=num_uploaded,
    )
    manifest_uri = upload_json(s3, args.bucket, f"{prefix}/manifest.json", manifest, args.dry_run)

    print(f"Parquet:  {parquet_uri}")
    print(f"Manifest: {manifest_uri}")
    print(f"Classes:  {classes_uri}")
    print(f"Colors:   {colors_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
