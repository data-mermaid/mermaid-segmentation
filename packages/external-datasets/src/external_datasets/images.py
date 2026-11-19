"""Image decodability checks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

# libjpeg decodes at 1/8 scale at best; draft() picks the largest reduction fitting the box.
_DRAFT_SCALE = 8


def _decode_error(path: Path) -> str | None:
    """Return an error description if ``path`` cannot be decoded, else None."""
    try:
        with Image.open(path) as img:
            if img.format == "JPEG":
                width, height = img.size
                img.draft(
                    img.mode,
                    (max(1, width // _DRAFT_SCALE), max(1, height // _DRAFT_SCALE)),
                )
            img.load()
    except (OSError, ValueError, SyntaxError) as exc:
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
