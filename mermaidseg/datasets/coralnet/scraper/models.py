"""Dataclasses and enums shared by CoralNet scraping and remediation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RemediationAction(StrEnum):
    SKIP_NOT_ACCESSIBLE = "skip_not_accessible"
    SKIP_NO_CONFIRMED_ANNOTATIONS = "skip_no_confirmed_annotations"
    SKIP_EMPTY_WEBSITE = "skip_empty_website"
    FULL_REDOWNLOAD = "full_redownload"
    REDOWNLOAD_CSV = "redownload_csv"
    REDOWNLOAD_IMAGES = "redownload_images"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class SourceProbe:
    source_id: int
    source_url: str
    accessible: bool
    error: str | None
    total_images_website: int
    n_confirmed_website: int | None = None
    n_unconfirmed_website: int | None = None
    n_unclassified_website: int | None = None


@dataclass
class DownloadResult:
    ok: bool
    skipped_no_confirmed: bool = False
    skipped_empty: bool = False
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok
