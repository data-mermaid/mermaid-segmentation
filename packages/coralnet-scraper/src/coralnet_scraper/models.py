"""Dataclass returned by CoralNet scraping."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DownloadResult:
    ok: bool
    skipped_no_confirmed: bool = False
    skipped_empty: bool = False
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok
