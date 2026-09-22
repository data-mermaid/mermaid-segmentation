"""Small helpers for incremental JSON/CSV/summary output."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    tmp.replace(path)


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if columns is None:
        columns = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any, digits: int = 4) -> str:
    """Format a float/None for concise logging."""
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # nan
        return "n/a"
    return f"{f:.{digits}f}"
