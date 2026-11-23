"""Load the curated native-label to MERMAID benthic-attribute maps shipped as package data."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from importlib import resources

_PACKAGE = "segdata_etl.label_maps"
_REPO_DIR = "packages/segdata-etl/src/segdata_etl/label_maps/curated"
DATASETS = (
    "catlin_seaview",
    "moorea_labeled_corals",
    "pacific_labeled_corals",
    "benthos_yuval",
)


@dataclass(frozen=True)
class CuratedMapInfo:
    """Provenance of a curated map: repo-relative CSV path, CSV byte sha256 and row count."""

    path: str
    sha256: str
    n_entries: int


def load_curated(dataset: str) -> tuple[dict[str, str], CuratedMapInfo]:
    """Return the ``native_name -> mermaid_benthic_attribute`` map of ``dataset`` and its info."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown curated dataset {dataset!r}; expected one of {DATASETS}")
    raw = resources.files(_PACKAGE).joinpath(f"curated/{dataset}.csv").read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    mapping = {row["native_name"]: row["mermaid_benthic_attribute"] for row in reader}
    info = CuratedMapInfo(
        path=f"{_REPO_DIR}/{dataset}.csv",
        sha256=hashlib.sha256(raw).hexdigest(),
        n_entries=len(mapping),
    )
    return mapping, info
