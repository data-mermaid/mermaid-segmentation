"""Dataclass returned by CoralNet scraping."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DownloadResult:
    """Outcome of downloading one CoralNet source.

    Attributes:
        ok: Whether the source was handled without a failure.
        skipped_no_confirmed: The source has no confirmed images on the website.
        skipped_empty: The source reports zero images.
        errors: Short failure codes or messages collected during the download.
    """

    ok: bool
    skipped_no_confirmed: bool = False
    skipped_empty: bool = False
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return ``ok``, so a result is truthy exactly when the download succeeded."""
        return self.ok
