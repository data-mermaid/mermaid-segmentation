"""Paths, config helpers and the label comparison shared by the deduplication steps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml

from mermaidseg.datasets.coralnet.coralnet_dataset import (
    _CORALNET_ID2NAME_KEY,
    _DEFAULT_MANIFEST_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE_ROOT = os.environ.get("MERMAIDSEG_LOCAL_CACHE_DIR", "/archive/vision/beery/red_sea_coral/cache_MERMAID")
DEFAULT_BUCKET = "dev-datamermaid-sm-sources"
DEFAULT_MANIFEST = _DEFAULT_MANIFEST_PATH
ID2NAME_KEY = _CORALNET_ID2NAME_KEY

# Intermediate outputs (hashes, pair lists, review files, ...): large, not versioned (data/ is gitignored).
DEFAULT_OUTPUTS = REPO_ROOT / "data/coralnet_dedup"
# Versioned results and curated inputs: the files training reads, the low-quality source list, ...
CONFIG_DIR = REPO_ROOT / "configs/coralnet"
DEFAULT_SPLIT_CONFIG = REPO_ROOT / "configs/data_config_all.yaml"
DEFAULT_CONCEPTS_CSV = REPO_ROOT / "configs/class_to_concepts.csv"
DEFAULT_EXCLUDE_SOURCES = CONFIG_DIR / "low_quality_sources.txt"

TAX_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus")
NON_CONCEPT_COLS = ("source_label_class_name", "source_dataset_source", "target_label_class_name",
                    "target_dataset_source", "species")


def read_source_list(path: Path | None) -> set[int]:
    """Source ids from a text file, one per line (``#`` starts a comment)."""
    if path is None:
        return set()
    lines = (ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines())
    return {int(ln) for ln in lines if ln}


def coralnet_splits(config_path: Path) -> dict[str, set[int]]:
    """Return ``{split: set(source_id)}`` from a data config's ``coralnet`` whitelists."""

    def _find(d):
        if isinstance(d, dict):
            if isinstance(d.get("coralnet"), dict):
                return d["coralnet"]
            for v in d.values():
                if (r := _find(v)) is not None:
                    return r
        return None

    cn = _find(yaml.safe_load(Path(config_path).read_text())) or {}
    return {
        k: set(cn[k]["whitelist_sources"])
        for k in ("train", "val", "test")
        if isinstance(cn.get(k), dict) and cn[k].get("whitelist_sources")
    }


def add_split_labels(summary: pd.DataFrame, splits: dict[str, set[int]]) -> pd.DataFrame:
    """Label each source with its split (``unused`` if in none) and flag cross-split pairs."""
    split_of = {s: k for k, v in splits.items() for s in v}
    summary = summary.drop(columns=["split_a", "split_b", "cross_split"], errors="ignore")
    summary.insert(2, "split_a", summary["source_a"].map(split_of).fillna("unused"))
    summary.insert(3, "split_b", summary["source_b"].map(split_of).fillna("unused"))
    used = (summary["split_a"] != "unused") & (summary["split_b"] != "unused")
    summary.insert(4, "cross_split", used & (summary["split_a"] != summary["split_b"]))
    return summary


def load_id2name(bucket_root: Path) -> dict[int, str]:
    """CoralNet label id -> lower-cased label name, from the local S3 mirror."""
    raw = json.loads((bucket_root / ID2NAME_KEY).read_text())
    return {int(k): str(v).strip().lower() for k, v in raw.items()}


def load_concepts(csv_path: Path) -> pd.DataFrame:
    """CoralNet label name -> concept fields (strings; ``not_given`` = unspecified)."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df[df["source_dataset_source"] == "coralnet"]
    df["name"] = df["source_label_class_name"].str.strip().str.lower()
    cols = [c for c in df.columns if c not in NON_CONCEPT_COLS and c != "name"]
    return df.drop_duplicates("name").set_index("name")[cols].apply(lambda s: s.str.strip().str.lower())


def compare_labels(fa: pd.Series | None, fb: pd.Series | None) -> str:
    """Relation between two labels' concept rows.

    - ``contradictory``: some field is specified on both sides with different values;
    - ``equivalent``: the same fields are specified, with the same values (synonyms);
    - ``a_finer`` / ``b_finer``: no conflict and one side specifies a strict superset of the
      other's fields (e.g. "acropora" vs "hard coral");
    - ``compatible``: no conflict but neither is a superset (e.g. taxon vs bleaching condition);
    - ``unmapped``: a label is missing from the concept CSV.
    """
    if fa is None or fb is None:
        return "unmapped"
    spec_a, spec_b = fa != "not_given", fb != "not_given"
    both = spec_a & spec_b
    if (fa[both] != fb[both]).any():
        return "contradictory"
    if (spec_a == spec_b).all():
        return "equivalent"
    if (spec_a | ~spec_b).all():  # a specifies everything b does, and more
        return "a_finer"
    if (spec_b | ~spec_a).all():
        return "b_finer"
    return "compatible"


def specificity(f: pd.Series | None) -> float:
    """Number of specified taxonomic ranks (``none`` for non-living counts as specified)."""
    if f is None:
        return float("nan")
    return float(sum(f[r] != "not_given" for r in TAX_RANKS if r in f.index))


def repo_relative(path: Path) -> str:
    """``path`` relative to the repo root if it lies inside it (as data configs reference it), else absolute."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
