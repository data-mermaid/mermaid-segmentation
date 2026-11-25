"""Image decodability checks: ``find_undecodable`` reports the files that fail to decode."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# libjpeg decodes at 1/8 scale at best; draft() picks the largest reduction fitting the box.
_DRAFT_SCALE = 8


def _draft_jpeg(img: Image.Image, box: tuple[int, int]) -> None:
    """Ask the JPEG decoder to decode at the largest reduced scale that fits ``box``; no-op otherwise."""
    if img.format != "JPEG":
        return
    img.draft(img.mode, box)


def _decode_error(path: Path) -> str | None:
    """Return an error description if ``path`` cannot be decoded, else None."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            _draft_jpeg(img, (max(1, width // _DRAFT_SCALE), max(1, height // _DRAFT_SCALE)))
            img.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def find_undecodable(paths: list[Path], *, workers: int) -> list[tuple[Path, str]]:
    """Decode each path and return ``(path, error)`` for those that fail.

    JPEG validity is checked at reduced scale (up to 1/8 per edge) because proof of
    decodability does not need full pixels, and full-size mosaics decoded by parallel
    workers would multiply peak memory. Other formats are fully decoded.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = list(pool.map(_decode_error, paths))
    return [(path, err) for path, err in zip(paths, errors, strict=True) if err is not None]
