"""Dataclass returned by CoralNet scraping."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageFailures:
    """Image failures of one source, by kind.

    Attributes:
        missing_url: Images whose page gave no ``Image URL``.
        upload_failed: Images whose upload reported failure or whose worker raised.
    """

    missing_url: int = 0
    upload_failed: int = 0


@dataclass
class DownloadResult:
    """Outcome of downloading one CoralNet source.

    Attributes:
        ok: Whether the source was handled without a failure.
        skipped_no_confirmed: The source has no confirmed images on the website.
        errors: Short failure codes or messages collected during the download.
        image_failures: Counts of failed images, by kind.
    """

    ok: bool
    skipped_no_confirmed: bool = False
    errors: list[str] = field(default_factory=list)
    image_failures: ImageFailures = field(default_factory=ImageFailures)

    def __bool__(self) -> bool:
        """Return ``ok``, so a result is truthy exactly when the download succeeded."""
        return self.ok
