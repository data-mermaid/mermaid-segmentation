"""Classify remediation action from audited S3 row + live website probe (no I/O)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mermaidseg.datasets.coralnet.scraper.models import RemediationAction, SourceProbe


def _row_bool(row: Mapping[str, Any], key: str) -> bool:
    v = row.get(key)
    if v is None:
        return False
    return bool(v)


def classify_remediation_action(
    *,
    probe: SourceProbe,
    has_images_folder: bool,
    has_annotations_csv: bool,
    has_image_list_csv: bool,
    n_images_s3: int,
    n_images_csv: int,
    n_annotations: int,
    annotations_csv_read_failed: bool,
    annotations_empty: bool,
    image_count_match: bool,
    image_list_covers_annotations: bool = True,
) -> RemediationAction:
    if not probe.accessible:
        return RemediationAction.SKIP_NOT_ACCESSIBLE

    if probe.n_confirmed_website is not None and probe.n_confirmed_website == 0:
        return RemediationAction.SKIP_NO_CONFIRMED_ANNOTATIONS

    tw = probe.total_images_website
    if tw == 0:
        return RemediationAction.SKIP_EMPTY_WEBSITE

    csv_broken_or_empty = (
        annotations_csv_read_failed
        or annotations_empty
        or (has_annotations_csv and n_annotations == 0)
    )
    csvs_ok_core = (
        has_annotations_csv
        and has_image_list_csv
        and not annotations_csv_read_failed
        and not annotations_empty
        and n_annotations > 0
    )

    if tw < n_images_s3:
        return RemediationAction.MANUAL_REVIEW

    gap = tw - n_images_s3
    missing_struct = not has_images_folder or not has_annotations_csv or not has_image_list_csv

    if missing_struct:
        return RemediationAction.FULL_REDOWNLOAD

    if csv_broken_or_empty and tw > 0:
        return RemediationAction.REDOWNLOAD_CSV

    if csvs_ok_core and not image_list_covers_annotations:
        return RemediationAction.REDOWNLOAD_CSV

    if csvs_ok_core and not image_count_match and gap > 0:
        return RemediationAction.REDOWNLOAD_IMAGES

    return RemediationAction.MANUAL_REVIEW


def classify_from_audit_series(
    probe: SourceProbe, audit_row: Mapping[str, Any]
) -> RemediationAction:
    """Convenience: map parquet/audit columns into :func:`classify_remediation_action`."""
    return classify_remediation_action(
        probe=probe,
        has_images_folder=_row_bool(audit_row, "has_images_folder"),
        has_annotations_csv=_row_bool(audit_row, "has_annotations_csv"),
        has_image_list_csv=_row_bool(audit_row, "has_image_list_csv"),
        n_images_s3=int(audit_row.get("n_images_s3") or 0),
        n_images_csv=int(audit_row.get("n_images_csv") or 0),
        n_annotations=int(audit_row.get("n_annotations") or 0),
        annotations_csv_read_failed=_row_bool(audit_row, "annotations_csv_read_failed"),
        annotations_empty=_row_bool(audit_row, "annotations_empty"),
        image_count_match=_row_bool(audit_row, "image_count_match"),
        image_list_covers_annotations=_row_bool(audit_row, "image_list_covers_annotations"),
    )
