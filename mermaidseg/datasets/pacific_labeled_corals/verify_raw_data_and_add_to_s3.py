"""Verify a manually downloaded Pacific Labeled Corals dataset and upload to S3.

The Dryad package ``doi:10.5061/dryad.m5pr3`` (Beijbom et al., 2015) is behind
a download portal that does not expose a stable API, so this script does not
auto-download. Point ``--data-dir`` at the unzipped package; the script then
walks ``<site>/<subset>/{imagemap.txt, annotations.txt, imgs/}`` for the three
Pacific sites (``heron_reef``, ``line_islands``, ``nanwan_bay``; the Moorea
split is processed separately under
``mermaidseg/datasets/moorea_labeled_corals``), runs a strict integrity check
on every image (full PIL decode), joins the per-site ``labelmap.txt`` to map
integer label IDs back to names, and writes everything to S3 as a single
Parquet plus ``classes.json`` / ``colors.json`` / ``manifest.json``.

Each annotation row carries every annotator column present in the raw files:
``archived`` (always populated), and ``host`` / ``visitor1`` ... ``visitor5``
(only populated for the **evaluation** subset; ``pd.NA`` for **reference**).
The PyTorch dataset class picks one of these columns at training time.

Run::

    python -m mermaidseg.datasets.pacific_labeled_corals.verify_raw_data_and_add_to_s3 \
        --data-dir /path/to/pacific_labeled_corals_downloaded
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

DRYAD_DATASET_DOI = "10.5061/dryad.m5pr3"
DRYAD_DATASET_URL = f"https://datadryad.org/dataset/doi:{DRYAD_DATASET_DOI}"
CITATION = (
    "Beijbom, O., Edmunds, P. J., Roelfsema, C., Smith, J., Kline, D. I., "
    "Neal, B. P., Dunlap, M. J., Moriarty, V., Fan, T.-Y., Tan, C.-J., "
    "Chan, S., Treibitz, T., Gamst, A., Mitchell, B. G., Kriegman, D. "
    "(2015). Towards Automated Annotation of Benthic Survey Images: "
    "Variability of Human Experts and Operational Modes of Automation. "
    "PLOS ONE, 10(7): e0130312."
)

PARQUET_NAME = "pacific_labeled_corals_annotations.parquet"

# Pacific sites: the Moorea split of the original "Pacific Labeled Corals"
# release is processed separately under ``mermaidseg/datasets/moorea_labeled_corals``.
SITES: tuple[str, ...] = ("heron_reef", "line_islands", "nanwan_bay")
SUBSETS: tuple[str, ...] = ("reference", "evaluation")

# Reference subsets only carry ``archived`` annotations.
# Evaluation subsets additionally carry ``host`` and ``visitor1..visitor5``.
ALL_ANNOTATOR_COLS: tuple[str, ...] = (
    "archived",
    "host",
    "visitor1",
    "visitor2",
    "visitor3",
    "visitor4",
    "visitor5",
)
OTHER_ANNOTATOR_COLS: tuple[str, ...] = ALL_ANNOTATOR_COLS[1:]


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        return [dict(row) for row in reader]


def read_labelmap(site_dir: Path) -> list[dict[str, object]]:
    """Read ``labelmap.txt`` (``labelid,name``) for one site."""
    rows: list[dict[str, object]] = []
    for raw in _read_csv_dicts(site_dir / "labelmap.txt"):
        rows.append({"labelid": int(raw["labelid"]), "name": raw["name"].strip()})
    return rows


def read_imagemap(path: Path) -> dict[int, str]:
    """Read ``imagemap.txt`` (``imageid,imagefile``) into ``{imageid: filename}``."""
    mapping: dict[int, str] = {}
    for raw in _read_csv_dicts(path):
        imageid = int(raw["imageid"])
        # The raw file is sometimes ``imagefile`` and sometimes ``image``; tolerate both.
        image_name = (raw.get("imagefile") or raw.get("image") or "").strip()
        if not image_name:
            raise KeyError(f"No filename column in {path}")
        mapping[imageid] = image_name
    return mapping


def read_annotations(path: Path) -> list[dict[str, str]]:
    """Read ``annotations.txt`` as a list of raw string rows."""
    return _read_csv_dicts(path)


def collect_subset_inputs(
    data_root: Path,
) -> dict[tuple[str, str], dict[str, object]]:
    """Walk ``<site>/<subset>/`` and return per-subset metadata.

    Returns a dict keyed by ``(site, subset)`` with values:
      - ``subset_dir``: ``Path`` to ``<data_root>/<site>/<subset>``
      - ``imagemap``: ``{imageid: filename}``
      - ``ann_rows``: raw annotation rows (list of dicts)
    """
    out: dict[tuple[str, str], dict[str, object]] = {}
    for site in SITES:
        site_dir = data_root / site
        if not site_dir.is_dir():
            raise FileNotFoundError(f"Missing site directory: {site_dir}")
        for subset in SUBSETS:
            subset_dir = site_dir / subset
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"Missing subset directory: {subset_dir}")
            imagemap_path = subset_dir / "imagemap.txt"
            ann_path = subset_dir / "annotations.txt"
            imgs_dir = subset_dir / "imgs"
            for required in (imagemap_path, ann_path, imgs_dir):
                if not required.exists():
                    raise FileNotFoundError(
                        f"Missing expected file/dir: {required}"
                    )
            out[(site, subset)] = {
                "subset_dir": subset_dir,
                "imagemap": read_imagemap(imagemap_path),
                "ann_rows": read_annotations(ann_path),
            }
    return out


def _decode_image(path: Path) -> tuple[Path, str | None]:
    """Try to fully decode ``path``; return ``(path, error)`` (``None`` if OK)."""
    try:
        with Image.open(path) as img:
            img.load()
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        return path, f"{type(e).__name__}: {e}"
    return path, None


def assert_all_images_decode(
    subset_inputs: dict[tuple[str, str], dict[str, object]],
    workers: int,
) -> None:
    """Strict integrity check: every imagemap entry must fully decode in PIL.

    Raises ``RuntimeError`` listing the first few offenders if any image is
    missing or fails to decode. The current Dryad release contains no
    damaged or missing images, so we treat any failure as a hard error
    rather than silently dropping rows.
    """
    tasks: list[Path] = []
    for (_site, _subset), info in subset_inputs.items():
        subset_dir: Path = info["subset_dir"]  # type: ignore[assignment]
        imagemap: dict[int, str] = info["imagemap"]  # type: ignore[assignment]
        for filename in imagemap.values():
            tasks.append(subset_dir / "imgs" / filename)
    if not tasks:
        return

    failures: list[tuple[Path, str]] = []
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=len(tasks), desc="validate img", unit="img") as pbar,
    ):
        futures = [pool.submit(_decode_image, p) for p in tasks]
        for fut in as_completed(futures):
            path, err = fut.result()
            if err is not None:
                failures.append((path, err))
            pbar.update(1)

    if failures:
        sample = "\n  ".join(f"{p}: {e}" for p, e in failures[:5])
        more = f"\n  ... ({len(failures) - 5} more)" if len(failures) > 5 else ""
        raise RuntimeError(
            f"Integrity check failed: {len(failures)} image(s) could not be "
            f"decoded by PIL.\n  {sample}{more}"
        )


def build_annotations_df(
    subset_inputs: dict[tuple[str, str], dict[str, object]],
    labelid_to_name: dict[int, str],
) -> pd.DataFrame:
    """Construct the wide annotations DataFrame.

    One row per ``(site, subset, raw_imageid, row, col)``; each annotator
    column is mapped through ``labelid_to_name`` and falls back to ``pd.NA``
    when the column is absent (reference subsets) or holds an unknown label
    ID (matching the notebook's "labelid not in id_to_name -> skip" behavior
    for ``archived``, but applied uniformly across all annotator columns).
    """
    records: list[dict[str, object]] = []
    n_dropped_unknown_archived = 0

    for (site, subset), info in subset_inputs.items():
        imagemap: dict[int, str] = info["imagemap"]  # type: ignore[assignment]
        ann_rows: list[dict[str, str]] = info["ann_rows"]  # type: ignore[assignment]

        for raw in ann_rows:
            imageid = int(raw["imageid"])
            if imageid not in imagemap:
                continue

            archived_name = labelid_to_name.get(int(raw["archived"]))
            if archived_name is None:
                # Matches the notebook's behavior for `archived`.
                n_dropped_unknown_archived += 1
                continue

            image_path = Path(imagemap[imageid])
            record: dict[str, object] = {
                "site": site,
                "subset": subset,
                "raw_imageid": imageid,
                "image_id": image_path.stem,
                "image_ext": image_path.suffix,
                "row": int(raw["row"]),
                "col": int(raw["col"]),
                "archived": archived_name,
            }
            for col in OTHER_ANNOTATOR_COLS:
                raw_val = raw.get(col)
                if not raw_val:
                    record[col] = pd.NA
                    continue
                record[col] = labelid_to_name.get(int(raw_val), pd.NA)

            records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(
            "No usable annotation rows were assembled; check --data-dir layout."
        )

    df = df.astype(
        {
            "site": "string",
            "subset": "string",
            "raw_imageid": "int64",
            "image_id": "string",
            "image_ext": "string",
            "row": "int64",
            "col": "int64",
            **{col: "string" for col in ALL_ANNOTATOR_COLS},
        }
    )

    if n_dropped_unknown_archived:
        logger.info(
            "Dropped %d annotation row(s) whose `archived` labelid was not in labelmap.",
            n_dropped_unknown_archived,
        )
    return df


def build_classes_and_colors(
    label_names: list[str], seed: int = 1337
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """``classes.json`` (1..N alphabetical, ``unlabeled=0``) + deterministic palette."""
    sorted_names = sorted(label_names)
    classes: dict[str, int] = {"unlabeled": 0}
    classes.update({name: i + 1 for i, name in enumerate(sorted_names)})

    rng = np.random.default_rng(seed)
    colors: dict[str, list[int]] = {"unlabeled": [0, 0, 0]}
    for name in sorted_names:
        colors[name] = [int(x) for x in rng.integers(0, 256, size=3)]
    return classes, colors


def list_existing_keys(s3, bucket: str, prefix: str) -> set[str]:
    """List every key already present under ``s3://bucket/prefix/``."""
    keys: set[str] = set()
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"
    ):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys


def upload_images(
    s3,
    df_annotations: pd.DataFrame,
    subset_inputs: dict[tuple[str, str], dict[str, object]],
    bucket: str,
    images_prefix: str,
    workers: int,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    """Upload one image per unique ``(site, subset, image_id)`` to S3.

    Returns the number of objects actually uploaded.
    """
    image_index = (
        df_annotations[["site", "subset", "image_id", "image_ext", "raw_imageid"]]
        .drop_duplicates(subset=["site", "subset", "image_id"])
        .reset_index(drop=True)
    )

    tasks: list[tuple[Path, str]] = []
    for _, row in image_index.iterrows():
        site = str(row["site"])
        subset = str(row["subset"])
        image_id = str(row["image_id"])
        image_ext = str(row["image_ext"])
        raw_imageid = int(row["raw_imageid"])
        info = subset_inputs[(site, subset)]
        subset_dir: Path = info["subset_dir"]  # type: ignore[assignment]
        imagemap: dict[int, str] = info["imagemap"]  # type: ignore[assignment]
        local = subset_dir / "imgs" / imagemap[raw_imageid]
        if not local.exists():
            raise FileNotFoundError(f"Local image vanished before upload: {local}")
        key = f"{images_prefix}/{site}/{subset}/{image_id}{image_ext}"
        tasks.append((local, key))

    if skip_existing and not dry_run:
        existing = list_existing_keys(s3, bucket, images_prefix)
        logger.info("Found %d existing image keys; skipping those.", len(existing))
        tasks = [(p, k) for p, k in tasks if k not in existing]

    if dry_run:
        logger.info("[dry-run] Would upload %d image objects.", len(tasks))
        return len(tasks)
    if not tasks:
        logger.info("No new images to upload.")
        return 0

    logger.info(
        "Uploading %d image objects with %d workers ...", len(tasks), workers
    )

    def _upload(local: Path, key: str) -> None:
        boto3.client("s3").upload_file(str(local), bucket, key)

    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=len(tasks), desc="upload img", unit="obj") as pbar,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a manually downloaded Pacific Labeled Corals dataset and "
            "upload to S3."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the per-site/per-subset raw files. Expected "
            "layout: <data_dir>/<site>/labelmap.txt and "
            "<data_dir>/<site>/<subset>/{imagemap.txt, annotations.txt, imgs/}. "
            "Download from: " + DRYAD_DATASET_URL
        ),
    )
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument(
        "--prefix", default="external_validation_datasets/pacific_labeled_corals"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--no-skip-existing-s3",
        dest="skip_existing_s3",
        action="store_false",
        help="By default, image keys already in S3 are skipped; pass to re-upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but skip all S3 writes.",
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
    logger.info(
        "Raw data must be manually downloaded from %s (see README.md).",
        DRYAD_DATASET_URL,
    )

    data_root = args.data_dir.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_root}")
    prefix = args.prefix.strip("/")
    images_prefix = f"{prefix}/images"
    annotations_key = f"{prefix}/{PARQUET_NAME}"

    site_labelmaps = {site: read_labelmap(data_root / site) for site in SITES}
    reference_labelmap = site_labelmaps[SITES[0]]
    for site, rows in site_labelmaps.items():
        if rows != reference_labelmap:
            raise ValueError(
                f"Per-site labelmap.txt mismatch at {site}; expected identical "
                "labelmaps across all sites."
            )
    labelid_to_name: dict[int, str] = {
        int(r["labelid"]): str(r["name"]) for r in reference_labelmap
    }
    label_names = sorted(labelid_to_name.values())
    classes, colors = build_classes_and_colors(label_names)
    logger.info(
        "Built classes.json with %d entries (incl. unlabeled).", len(classes)
    )

    subset_inputs = collect_subset_inputs(data_root)
    n_imagemap_entries = sum(len(info["imagemap"]) for info in subset_inputs.values())
    n_ann_rows = sum(len(info["ann_rows"]) for info in subset_inputs.values())
    logger.info(
        "Collected %d imagemap entries and %d raw annotation rows across %d (site, subset) pairs.",
        n_imagemap_entries,
        n_ann_rows,
        len(subset_inputs),
    )

    assert_all_images_decode(subset_inputs, args.workers)

    df_annotations = build_annotations_df(
        subset_inputs=subset_inputs,
        labelid_to_name=labelid_to_name,
    )

    n_unique_images = (
        df_annotations.drop_duplicates(["site", "subset", "image_id"]).shape[0]
    )
    logger.info(
        "Built annotations DataFrame: %d rows across %d unique images.",
        len(df_annotations),
        n_unique_images,
    )

    s3 = boto3.client("s3")
    parquet_uri = f"s3://{args.bucket}/{annotations_key}"
    if args.dry_run:
        logger.info(
            "[dry-run] Would write parquet (%d rows) to %s",
            len(df_annotations),
            parquet_uri,
        )
    else:
        logger.info("Writing %d annotation rows -> %s", len(df_annotations), parquet_uri)
        df_annotations.to_parquet(parquet_uri, engine="pyarrow", index=False)

    num_uploaded = upload_images(
        s3=s3,
        df_annotations=df_annotations,
        subset_inputs=subset_inputs,
        bucket=args.bucket,
        images_prefix=images_prefix,
        workers=args.workers,
        skip_existing=args.skip_existing_s3,
        dry_run=args.dry_run,
    )

    classes_uri = upload_json(s3, args.bucket, f"{prefix}/classes.json", classes, args.dry_run)
    colors_uri = upload_json(s3, args.bucket, f"{prefix}/colors.json", colors, args.dry_run)

    manifest = {
        "dataset_name": "pacific_labeled_corals",
        "source": {
            "citation": CITATION,
            "raw_data_url": DRYAD_DATASET_URL,
            "license": "CC0 1.0 (Dryad)",
            "excluded_sites": ["moorea"],
        },
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "s3": {
            "bucket": args.bucket,
            "prefix": prefix,
            "annotations_parquet": annotations_key,
            "annotations_uri": parquet_uri,
            "images_prefix": images_prefix,
            "image_key_template": f"{images_prefix}/{{site}}/{{subset}}/{{image_id}}{{image_ext}}",
            "classes_json": f"{prefix}/classes.json",
            "colors_json": f"{prefix}/colors.json",
        },
        "schema": {
            "parquet_columns": {
                "site": "Pacific site code (heron_reef, line_islands, nanwan_bay).",
                "subset": "Annotation subset (reference or evaluation).",
                "raw_imageid": "Original integer imageid from imagemap.txt.",
                "image_id": "Filename stem of the source jpg (string).",
                "image_ext": "Filename suffix incl. dot, e.g. '.JPG'.",
                "row": "Annotation pixel row (integer).",
                "col": "Annotation pixel column (integer).",
                "archived": (
                    "Pacific Labeled Corals label name from the original "
                    "ecological survey (the only annotator present in "
                    "reference subsets)."
                ),
                "host": (
                    "Re-annotation by the same coral expert (evaluation "
                    "subsets only; pd.NA elsewhere)."
                ),
                **{
                    f"visitor{i}": (
                        f"Annotation by visiting coral expert #{i} "
                        "(evaluation subsets only; pd.NA elsewhere)."
                    )
                    for i in range(1, 6)
                },
            },
        },
        "ingestion": {
            "num_images_uploaded_this_run": num_uploaded,
        },
    }
    manifest_uri = upload_json(
        s3, args.bucket, f"{prefix}/manifest.json", manifest, args.dry_run
    )

    print(f"Parquet:  {parquet_uri}")
    print(f"Manifest: {manifest_uri}")
    print(f"Classes:  {classes_uri}")
    print(f"Colors:   {colors_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
