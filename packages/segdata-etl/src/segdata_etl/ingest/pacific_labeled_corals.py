"""Pacific Labeled Corals ingestion: verify a manually downloaded package, publish to S3.

The Dryad package ``doi:10.5061/dryad.m5pr3`` has no stable download API, so ``--data-dir``
points at the unzipped package::

    <root>/<site>/labelmap.txt
    <root>/<site>/<subset>/{imagemap.txt, annotations.txt, imgs/}

Only the three Pacific sites are ingested; the Moorea split is a separate dataset.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.images import find_undecodable
from ..common.labels import build_classes_and_colors
from ..common.manifest import Source, build_manifest
from ..common.publish import S3Target, preflight, publish_dataset, upload_files

logger = logging.getLogger(__name__)

DATASET_NAME = "pacific_labeled_corals"
DRYAD_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.m5pr3"
CITATION = (
    "Beijbom, O., Edmunds, P. J., Roelfsema, C., Smith, J., Kline, D. I., "
    "Neal, B. P., Dunlap, M. J., Moriarty, V., Fan, T.-Y., Tan, C.-J., "
    "Chan, S., Treibitz, T., Gamst, A., Mitchell, B. G., Kriegman, D. "
    "(2015). Towards Automated Annotation of Benthic Survey Images: "
    "Variability of Human Experts and Operational Modes of Automation. "
    "PLOS ONE, 10(7): e0130312."
)

SITES: tuple[str, ...] = ("heron_reef", "line_islands", "nanwan_bay")
SUBSETS: tuple[str, ...] = ("reference", "evaluation")
# Reference subsets carry only ``archived``; evaluation subsets add host and visitor1..5.
OTHER_ANNOTATOR_COLS: tuple[str, ...] = ("host", *(f"visitor{i}" for i in range(1, 6)))
ALL_ANNOTATOR_COLS: tuple[str, ...] = ("archived", *OTHER_ANNOTATOR_COLS)
_SORT_KEYS = ["site", "subset", "image_id", "row", "col"]

PARQUET_COLUMNS = {
    "site": "Pacific site code (heron_reef, line_islands, nanwan_bay).",
    "subset": "Annotation subset (reference or evaluation).",
    "raw_imageid": "Original integer imageid from imagemap.txt.",
    "image_id": "Filename stem of the source jpg (string).",
    "image_ext": "Filename suffix incl. dot, e.g. '.JPG'.",
    "row": "Annotation pixel row (integer).",
    "col": "Annotation pixel column (integer).",
    "archived": (
        "Pacific Labeled Corals label name from the original ecological survey "
        "(the only annotator present in reference subsets)."
    ),
    "host": "Re-annotation by the same coral expert (evaluation subsets only; NA elsewhere).",
    **{
        f"visitor{i}": f"Annotation by visiting coral expert #{i} (evaluation subsets only; NA elsewhere)."
        for i in range(1, 6)
    },
}

SOURCE = Source(
    name=DATASET_NAME,
    citation=CITATION,
    raw_data_url=DRYAD_URL,
    license="CC0 1.0 (Dryad)",
)


@dataclass(frozen=True)
class _Subset:
    site: str
    subset: str
    subset_dir: Path
    imagemap: dict[int, str]
    ann_rows: list[dict[str, str]]


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Register the Pacific Labeled Corals flags; the raw data is downloaded manually."""
    del workdir  # no automatic download
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=f"Unzipped Dryad package (manual download from {DRYAD_URL}).",
    )


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, skipinitialspace=True)]


def _read_labelmap(site_dir: Path) -> dict[int, str]:
    rows = _read_csv_dicts(site_dir / "labelmap.txt")
    return {int(r["labelid"]): r["name"].strip() for r in rows}


def _read_imagemap(path: Path) -> dict[int, str]:
    """Read ``{imageid: filename}``; the filename column is ``imagefile`` or ``image``."""
    mapping: dict[int, str] = {}
    for raw in _read_csv_dicts(path):
        name = (raw.get("imagefile") or raw.get("image") or "").strip()
        if not name:
            raise KeyError(f"No filename column in {path}")
        mapping[int(raw["imageid"])] = name
    return mapping


def _read_shared_labelmap(data_root: Path) -> dict[int, str]:
    """Return the labelmap shared by every site; differing sites raise ``ValueError``."""
    maps = {site: _read_labelmap(data_root / site) for site in SITES}
    shared = maps[SITES[0]]
    for site, labelmap in maps.items():
        if labelmap != shared:
            raise ValueError(
                f"Per-site labelmap.txt mismatch at {site}; expected identical labelmaps "
                "across all sites."
            )
    return shared


def _collect_subsets(data_root: Path) -> list[_Subset]:
    subsets: list[_Subset] = []
    for site in SITES:
        for subset in SUBSETS:
            subset_dir = data_root / site / subset
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"Missing subset directory: {subset_dir}")
            for required in ("imagemap.txt", "annotations.txt", "imgs"):
                if not (subset_dir / required).exists():
                    raise FileNotFoundError(f"Missing expected file/dir: {subset_dir / required}")
            subsets.append(
                _Subset(
                    site,
                    subset,
                    subset_dir,
                    _read_imagemap(subset_dir / "imagemap.txt"),
                    _read_csv_dicts(subset_dir / "annotations.txt"),
                )
            )
    return subsets


def _assert_images_decode(subsets: list[_Subset], *, workers: int) -> None:
    """Raise ``RuntimeError`` listing the first offenders if any mapped image is unreadable."""
    paths = [s.subset_dir / "imgs" / name for s in subsets for name in s.imagemap.values()]
    failures = find_undecodable(paths, workers=workers)
    if failures:
        sample = "\n  ".join(f"{p}: {e}" for p, e in failures[:5])
        more = f"\n  ... ({len(failures) - 5} more)" if len(failures) > 5 else ""
        raise RuntimeError(
            f"Integrity check failed: {len(failures)} image(s) could not be decoded.\n"
            f"  {sample}{more}"
        )


_DROP_UNMAPPED_IMAGEID = "unmapped imageid"
_DROP_UNKNOWN_ARCHIVED = "unknown archived"


def _build_record(
    subset: _Subset, raw: dict[str, str], labelid_to_name: dict[int, str]
) -> tuple[dict[str, object] | None, str | None]:
    """Return ``(row, None)``, or ``(None, reason)`` when its image or archived label is unknown."""
    imageid = int(raw["imageid"])
    if imageid not in subset.imagemap:
        return None, _DROP_UNMAPPED_IMAGEID
    archived = labelid_to_name.get(int(raw["archived"]))
    if archived is None:
        return None, _DROP_UNKNOWN_ARCHIVED
    image_path = Path(subset.imagemap[imageid])
    record: dict[str, object] = {
        "site": subset.site,
        "subset": subset.subset,
        "raw_imageid": imageid,
        "image_id": image_path.stem,
        "image_ext": image_path.suffix,
        "row": int(raw["row"]),
        "col": int(raw["col"]),
        "archived": archived,
    }
    for col in OTHER_ANNOTATOR_COLS:
        raw_val = raw.get(col)
        record[col] = labelid_to_name.get(int(raw_val), pd.NA) if raw_val else pd.NA
    return record, None


def _build_annotations_df(
    subsets: list[_Subset], labelid_to_name: dict[int, str]
) -> tuple[pd.DataFrame, Counter[str]]:
    """Return the wide annotations frame (one row per site, subset, image, row, col) and drop counts.

    The counts are keyed by drop reason; each non-zero reason is logged at WARNING.
    """
    records: list[dict[str, object]] = []
    dropped: Counter[str] = Counter()
    for subset in subsets:
        for raw in subset.ann_rows:
            rec, reason = _build_record(subset, raw, labelid_to_name)
            if rec is not None:
                records.append(rec)
            elif reason is not None:
                dropped[reason] += 1
    for reason, n in dropped.items():
        logger.warning("Dropped %d annotation row(s): %s", n, reason)
    if not records:
        raise RuntimeError("No usable annotation rows were assembled; check --data-dir layout.")
    df = pd.DataFrame.from_records(records).astype(
        {
            "site": "string",
            "subset": "string",
            "raw_imageid": "int64",
            "image_id": "string",
            "image_ext": "string",
            "row": "int64",
            "col": "int64",
            **dict.fromkeys(ALL_ANNOTATOR_COLS, "string"),
        }
    )
    return df, dropped


def _image_uploads(df: pd.DataFrame, subsets: list[_Subset], prefix: str) -> list[tuple[Path, str]]:
    """Return ``(local path, key)`` for each distinct annotated image."""
    by_key = {(s.site, s.subset): s for s in subsets}
    unique = df.drop_duplicates(["site", "subset", "image_id"])
    uploads = []
    for site, subset, imageid, image_id, ext in unique[
        ["site", "subset", "raw_imageid", "image_id", "image_ext"]
    ].itertuples(index=False):
        s = by_key[(site, subset)]
        local = s.subset_dir / "imgs" / s.imagemap[imageid]
        uploads.append((local, f"{prefix}/images/{site}/{subset}/{image_id}{ext}"))
    return uploads


def run(args: argparse.Namespace, client: Any) -> int:
    """Ingest Pacific Labeled Corals into S3; return the process exit code."""
    target = S3Target(args.bucket, args.prefix.strip("/"), args.dry_run)
    preflight(client, target, DATASET_NAME, overwrite=args.overwrite)
    data_root = args.data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-dir does not exist: {data_root}")

    labelid_to_name = _read_shared_labelmap(data_root)
    subsets = _collect_subsets(data_root)
    _assert_images_decode(subsets, workers=args.workers)
    annotations, dropped = _build_annotations_df(subsets, labelid_to_name)
    classes, colors = build_classes_and_colors(labelid_to_name.values())

    uploaded = upload_files(
        client,
        target,
        _image_uploads(annotations, subsets, target.prefix),
        workers=args.workers,
        skip_existing=args.skip_existing_s3,
    )
    manifest = build_manifest(
        SOURCE,
        target,
        image_key_template=f"{target.prefix}/images/{{site}}/{{subset}}/{{image_id}}{{image_ext}}",
        parquet_columns=PARQUET_COLUMNS,
        ingestion={
            "excluded_sites": ["moorea"],
            "images_uploaded": uploaded,
            "annotations_dropped_unknown_archived": dropped[_DROP_UNKNOWN_ARCHIVED],
            "annotations_dropped_unmapped_imageid": dropped[_DROP_UNMAPPED_IMAGEID],
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
