"""Image decodability checks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

# Source mosaics exceed PIL's decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None


def _decode_error(path: Path) -> str | None:
    """Return an error description if ``path`` cannot be fully decoded, else None."""
    try:
        with Image.open(path) as img:
            img.load()
    except (OSError, ValueError, SyntaxError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def find_undecodable(paths: list[Path], *, workers: int) -> list[tuple[Path, str]]:
    """Fully decode each path and return ``(path, error)`` for those that fail."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = list(pool.map(_decode_error, paths))
    return [(path, err) for path, err in zip(paths, errors, strict=True) if err is not None]
