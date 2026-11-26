"""Completeness verification of one scraped source, read back from S3.

A source gets its completion marker only when :func:`verify_source` reports no failures.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from coralnet_scraper import s3_io

_STATUS_SUFFIX = re.compile(r" - (Confirmed|Unconfirmed|Unclassified)$")
_IMAGE_PAGE_ID = re.compile(r"/image/(\d+)/?")


@dataclass
class Verification:
    """Result of :func:`verify_source`.

    Attributes:
        failures: Codes that block the completion marker.
        warnings: Codes recorded in the marker that do not block it.
        stats: Counts written into the marker.
    """

    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the source passed, i.e. no failure was found."""
        return not self.failures


def _image_object_sizes(objects: dict[str, int], images_prefix: str) -> dict[int, int]:
    """Return the size of each ``images/<id>.jpg`` object, keyed by image id."""
    sizes: dict[int, int] = {}
    for key, size in objects.items():
        stem = PurePosixPath(key).stem
        if key.startswith(images_prefix) and stem.isdigit():
            sizes[int(stem)] = size
    return sizes


def strip_status(name: object) -> str:
    """Return a thumbnail title without its `` - <status>`` suffix."""
    return _STATUS_SUFFIX.sub("", str(name))


def _listed_ids(image_list: pd.DataFrame) -> list[int]:
    """Return the image ids parsed from the ``Image Page`` links, skipping unparseable ones."""
    return [
        int(m.group(1))
        for page in image_list["Image Page"].astype(str)
        if (m := _IMAGE_PAGE_ID.search(page))
    ]


def _check_image_list(image_list: pd.DataFrame, expected_total: int) -> tuple[list[int], list[str]]:
    """Return the listed image ids and the image-list failures."""
    ids = _listed_ids(image_list)
    failures = []
    if len(ids) != len(image_list) or len(set(ids)) != len(ids):
        failures.append("image_list_bad_ids")
    if len(image_list) != expected_total:
        failures.append(f"image_list_count:{len(image_list)}!={expected_total}")
    return ids, failures


def _bad_columns(annotations: pd.DataFrame) -> bool:
    """Return whether the annotations lack ``Name`` or carry HTML or unnamed columns."""
    junk = any("<" in str(c) or str(c).startswith("Unnamed") for c in annotations.columns)
    return "Name" not in annotations.columns or junk


def _check_annotation_names(annotations: pd.DataFrame, list_names: set[str]) -> list[str]:
    """Return the failures of annotation names: null, or absent from the image list."""
    failures = []
    if annotations["Name"].isna().any():
        failures.append("annotations_null_name")
    foreign = set(annotations["Name"].dropna().astype(str)) - list_names
    if foreign:
        failures.append(f"annotations_name_not_in_list:{len(foreign)}")
    return failures


def _list_warnings(image_list: pd.DataFrame, annotated: set[str]) -> list[str]:
    """Return the warnings of the image list: unannotated Confirmed images, duplicate names."""
    titles = [str(n) for n in image_list["Name"].dropna()]
    warnings = []
    confirmed = {strip_status(t) for t in titles if t.endswith(" - Confirmed")}
    if confirmed - annotated:
        warnings.append(f"confirmed_without_annotations:{len(confirmed - annotated)}")
    duplicated = [name for name, n in Counter(map(strip_status, titles)).items() if n > 1]
    if duplicated:
        warnings.append(f"duplicate_names:{len(duplicated)}")
    return warnings


def verify_source(
    s3: Any, bucket: str, prefix: str, source_id: int, *, expected_total: int
) -> Verification:
    """Check a source's objects on S3 against each other and against CoralNet's image count.

    Args:
        s3: S3 client.
        bucket: Bucket holding the source.
        prefix: Base key prefix of the scrape.
        source_id: CoralNet source id.
        expected_total: The image count CoralNet's overview reports for the source.

    Returns:
        The failures, warnings and counts. A missing ``annotations.csv`` or ``image_list.csv``,
        or unusable annotation columns, end the checks early with no counts.
    """
    v = Verification()
    base = s3_io.source_prefix(prefix, source_id)
    objects = s3_io.object_sizes(s3, bucket, base)
    ann_key = s3_io.object_key_annotation_csv(prefix, source_id)
    list_key = s3_io.object_key_image_list_csv(prefix, source_id)
    for key, code in ((ann_key, "annotations_missing"), (list_key, "image_list_missing")):
        if key not in objects:
            v.failures.append(code)
    if v.failures:
        return v

    image_list = s3_io.read_csv_s3(s3, bucket, list_key, encoding="utf-8")
    annotations = s3_io.read_csv_s3(s3, bucket, ann_key, encoding="utf-8")
    assert image_list is not None and annotations is not None  # both listed just above
    ids, v.failures = _check_image_list(image_list, expected_total)
    if _bad_columns(annotations):
        v.failures.append("annotations_bad_columns")
        return v
    list_names = {strip_status(n) for n in image_list["Name"].dropna()}
    v.failures += _check_annotation_names(annotations, list_names)
    annotated = set(annotations["Name"].dropna().astype(str))
    v.warnings += _list_warnings(image_list, annotated)

    image_sizes = _image_object_sizes(objects, f"{base}images/")
    missing = [i for i in ids if image_sizes.get(i, 0) <= 0]
    if missing:
        v.failures.append(f"image_objects_missing:{len(missing)}")
    orphans = set(image_sizes) - set(ids)
    if orphans:
        v.warnings.append(f"orphan_image_objects:{len(orphans)}")

    v.stats = {
        "total_images": len(image_list),
        "annotation_rows": len(annotations),
        "annotated_images": len(annotated),
        "image_objects": len(image_sizes),
    }
    return v


def orphan_image_keys(s3: Any, bucket: str, prefix: str, source_id: int) -> list[str]:
    """Return the keys of the source's image objects whose id its ``image_list.csv`` lacks.

    Raises:
        FileNotFoundError: If the source has no ``image_list.csv``.
    """
    list_key = s3_io.object_key_image_list_csv(prefix, source_id)
    image_list = s3_io.read_csv_s3(s3, bucket, list_key, encoding="utf-8")
    if image_list is None:
        raise FileNotFoundError(f"s3://{bucket}/{list_key}")
    listed = set(_listed_ids(image_list))
    images_prefix = f"{s3_io.source_prefix(prefix, source_id)}images/"
    return [
        key
        for key in s3_io.object_sizes(s3, bucket, images_prefix)
        if (stem := PurePosixPath(key).stem).isdigit() and int(stem) not in listed
    ]
