"""Joint label space, taxonomy indices, and concept LUTs from `class_to_concepts.csv`.

Supports two ways to define the set of class IDs:

- ``csv_canonical``: unique terminal class names in the CSV (rows with no
  ``should_map_to_label``), sorted lexicographically.
- ``mermaid_api``: all benthic attribute ``name`` values from the MERMAID API;
  resolved CSV terminals are matched to API names (case-insensitive).

Class id ``0`` is reserved for ignore / unknown at training time; valid classes
are ``1 .. N``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

LabelspaceMode = Literal["mermaid_api", "csv_canonical"]

TAXONOMY_COLUMNS: tuple[str, ...] = (
    "kingdom",
    "phylum",
    "order",
    "family",
    "genus",
    "species",
    "class",
)


def _norm_key(s: str) -> str:
    """Lowercase + strip for case-insensitive comparison of labels, sources, and taxonomy tokens."""
    return str(s).strip().lower()


def _lower_optional_label_field(x: Any) -> Any:
    """Lowercase mapping fields; preserve empty / NA."""
    if x is None:
        return x
    try:
        if pd.isna(x):
            return x
    except (TypeError, ValueError):
        pass
    if isinstance(x, float) and np.isnan(x):
        return x
    s = str(x).strip()
    if s == "":
        return ""
    if s.lower() == "nan":
        return np.nan
    return s.lower()


def _lower_taxonomy_cell(x: Any) -> Any:
    """Lowercase taxonomy column values; preserve empty / NA."""
    if x is None:
        return x
    try:
        if pd.isna(x):
            return x
    except (TypeError, ValueError):
        pass
    if isinstance(x, float) and np.isnan(x):
        return x
    s = str(x).strip()
    if s == "":
        return s
    return s.lower()


def _normalize_source(val: Any) -> str | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, float) and np.isnan(val):
        return None
    s = str(val).strip().lower()
    return s if s and s != "nan" else None


def _token_sort_key(name: str) -> str:
    """Map names that differ only by word order (e.g. ``bleached porites`` vs ``porites bleached``)."""
    parts = [p for p in _norm_key(str(name)).replace("/", " ").split() if p]
    return " ".join(sorted(parts))


def _is_empty_map(val: Any) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in {"nan", "none"}


def load_class_to_concepts_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV with lowercase column names and lowercase text for stable comparison."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    df.columns = pd.Index([str(c).strip().lower() for c in df.columns])
    for c in ("name", "source", "should_map_to_label", "should_map_to_label_source"):
        if c in df.columns:
            df[c] = df[c].map(_lower_optional_label_field)
    for col in TAXONOMY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_lower_taxonomy_cell)
    return df


def infer_binary_concept_columns(df: pd.DataFrame) -> list[str]:
    """Columns between ``species`` and final ``class`` (taxonomic class)."""
    cols = list(df.columns)
    if "species" not in cols or "class" not in cols:
        raise ValueError("CSV must contain 'species' and 'class' columns.")
    i0 = cols.index("species") + 1
    i1 = cols.index("class")
    return cols[i0:i1]


def fetch_mermaid_benthic_attribute_names(
    url: str = "https://api.datamermaid.org/v1/benthicattributes/",
    timeout: float = 60.0,
) -> list[str]:
    """Return sorted unique benthic attribute names from the MERMAID API."""
    names: list[str] = []
    next_url: str | None = url
    while next_url:
        response = requests.get(next_url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        for rec in data.get("results", []):
            if "name" in rec and rec["name"]:
                names.append(str(rec["name"]).strip().lower())
        next_url = data.get("next") or None
    return sorted(set(names), key=str.lower)


def parse_taxonomy_cell_to_id(raw: Any, value2id: dict[str, int]) -> int:
    """Map a CSV cell to a taxonomy id; ``not_given`` / empty → 0 (ignore)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0
    s = str(raw).strip()
    if s == "":
        return 0
    sk = s.lower().replace(" ", "_")
    if sk in ("not_given", "notgiven") or s.lower() == "not given":
        return 0
    lookup = _norm_key(s)
    if lookup in value2id:
        return int(value2id[lookup])
    if s in value2id:
        return int(value2id[s])
    if sk in value2id:
        return int(value2id[sk])
    return 0


def parse_binary_cell_to_trinary(raw: Any) -> int:
    """Trinary concept channel: 0 = ignore (not_given), 1 = False, 2 = True."""
    if raw is None:
        return 0
    if raw is pd.NA:
        return 0
    if isinstance(raw, bool):
        return 2 if raw else 1
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if np.isnan(raw):
            return 0
        if raw == 0:
            return 1
        if raw == 1:
            return 2
    s = str(raw).strip().lower().replace("_", " ")
    if s in {"", "nan", "not given", "not_given", "notgiven"}:
        return 0
    if s in {"false", "f", "0", "no"}:
        return 1
    if s in {"true", "t", "1", "yes"}:
        return 2
    logger.debug("parse_binary_cell_to_trinary: unrecognised value %r → ignore", raw)
    return 0


@dataclass
class TaxonomyLevelSpec:
    """Per taxonomic column: value strings and integer ids (0 = ignore)."""

    column: str
    id2value: dict[int, str]
    value2id: dict[str, int]
    dtype: Literal["uint8", "uint16"]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "ignore_id": 0,
            "id2value": {str(k): v for k, v in sorted(self.id2value.items())},
            "value2id": self.value2id,
            "dtype": self.dtype,
        }


@dataclass
class BinaryConceptSpec:
    """Ordered binary (trinary) concept columns."""

    columns: list[str]
    ignore_value: int = 0
    false_value: int = 1
    true_value: int = 2

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "ignore_id": self.ignore_value,
            "false_id": self.false_value,
            "true_id": self.true_value,
        }


def build_taxonomy_levels(df: pd.DataFrame) -> dict[str, TaxonomyLevelSpec]:
    """One id table per taxonomy column; ``not_given`` / empty → id 0."""
    out: dict[str, TaxonomyLevelSpec] = {}
    for col in TAXONOMY_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing taxonomy column {col!r} in CSV.")
        values: set[str] = set()
        for raw in df[col].tolist():
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            s = str(raw).strip()
            if s == "":
                continue
            sk = s.lower().replace(" ", "_")
            if sk in {"not_given", "notgiven"}:
                continue
            if s.strip().lower() == "not given":
                continue
            values.add(s.lower())
        sorted_vals = sorted(values, key=str.lower)
        id2value: dict[int, str] = {0: "not_given"}
        value2id: dict[str, int] = {"not_given": 0, "": 0}
        for i, v in enumerate(sorted_vals, start=1):
            id2value[i] = v
            value2id[v] = i
            value2id[_norm_key(v)] = i
            value2id[v.replace(" ", "_")] = i
        n = len(sorted_vals) + 1
        dtype: Literal["uint8", "uint16"] = "uint8" if n < 256 else "uint16"
        out[col] = TaxonomyLevelSpec(column=col, id2value=id2value, value2id=value2id, dtype=dtype)
    return out


def build_binary_concepts(df: pd.DataFrame) -> BinaryConceptSpec:
    cols = infer_binary_concept_columns(df)
    return BinaryConceptSpec(columns=cols)


def _find_rows_for_name_and_source(
    df: pd.DataFrame,
    name: str,
    source: str | None,
) -> pd.DataFrame:
    """Rows matching ``name``; if ``source`` is set, filter to that source."""
    nkey = _norm_key(name)
    name_match = df[df["name"].map(lambda x: _norm_key(str(x)) == nkey)]
    src = _normalize_source(source)
    if src is None:
        return name_match
    skey = _norm_key(src)
    return name_match[name_match["source"].map(lambda x: _norm_key(str(x)) == skey)]


def _find_row_for_label(df: pd.DataFrame, name: str, source: str | None) -> pd.Series | None:
    """Resolve a CSV row by (``source``, ``name``), with name-only and token-order fallbacks."""
    name = str(name).strip().lower()
    src = _normalize_source(source)
    if src is not None:
        rows = _find_rows_for_name_and_source(df, name, src)
        if not rows.empty:
            return rows.iloc[0]
    rows = _find_rows_for_name_and_source(df, name, None)
    if not rows.empty:
        if src is not None:
            skey = _norm_key(src)
            pref = rows[rows["source"].map(lambda x: _norm_key(str(x)) == skey)]
            if not pref.empty:
                return pref.iloc[0]
        return rows.iloc[0]
    ts = _token_sort_key(name)
    candidates: list[pd.Series] = []
    for _, row in df.iterrows():
        if _token_sort_key(row["name"]) != ts:
            continue
        if src is None or _norm_key(str(row["source"])) == _norm_key(src):
            candidates.append(row)
    if not candidates:
        return None
    if src is not None:
        for row in candidates:
            if _norm_key(str(row["source"])) == _norm_key(src):
                return row
    return candidates[0]


def resolve_mapping_chain(
    df: pd.DataFrame,
    source: str,
    name: str,
    max_hops: int = 64,
) -> tuple[str, str]:
    """Follow ``should_map_to_label`` until terminal; return (terminal_name, terminal_source), all lowercase."""
    name_key = str(name).strip().lower()
    row = _find_row_for_label(df, name_key, _normalize_source(source))
    if row is None:
        logger.debug("resolve_mapping_chain: no start row for source=%r name=%r", source, name)
        return name_key, (_normalize_source(source) or "")

    hops = 0
    visited: set[tuple[str, str]] = set()
    while hops < max_hops and not _is_empty_map(row.get("should_map_to_label")):
        key = (_norm_key(str(row["source"])), _norm_key(str(row["name"])))
        if key in visited:
            raise ValueError(f"Mapping cycle detected at source={row['source']!r} name={row['name']!r}")
        visited.add(key)

        tgt_label = str(row["should_map_to_label"]).strip().lower()
        tgt_src = _normalize_source(row.get("should_map_to_label_source"))
        if tgt_src is None:
            tgt_src = _normalize_source(row.get("source"))
        nxt = _find_row_for_label(df, tgt_label, tgt_src)
        if nxt is None:
            break
        if _norm_key(str(nxt["name"])) == _norm_key(str(row["name"])) and _norm_key(
            str(nxt["source"])
        ) == _norm_key(str(row["source"])):
            break
        row = nxt
        hops += 1

    if hops >= max_hops:
        raise ValueError(f"Mapping chain exceeded {max_hops} hops from {source=!r} {name=!r}")
    return _norm_key(str(row["name"])), _norm_key(str(row["source"]))


def _terminal_rows_mask(df: pd.DataFrame) -> pd.Series:
    sl = df["should_map_to_label"]
    return sl.isna() | (sl.astype("string").str.strip() == "")


@dataclass
class Labelspace:
    """Joint class ids + helpers to map dataset-specific names to ids."""

    mode: LabelspaceMode
    id2name: dict[int, str]
    name2id: dict[str, int]
    api_fold_to_canonical: dict[str, str] = field(default_factory=dict)
    df: pd.DataFrame | None = None

    def canonical_name_for_api(self, terminal_name: str) -> str | None:
        """Return API-canonical spelling for a terminal CSV name (mermaid_api mode)."""
        if self.mode != "mermaid_api":
            return terminal_name
        fold = _norm_key(terminal_name)
        return self.api_fold_to_canonical.get(fold)

    def source_name_to_class_id(self, source: str, name: str) -> int:
        """Map a dataset-specific label to joint class id (0 = unknown / not in space)."""
        if self.df is None:
            raise ValueError("Labelspace.df is required for chain resolution.")
        src = _normalize_source(source) or ""
        nm = str(name).strip().lower()
        tname, _tsrc = resolve_mapping_chain(self.df, src, nm)
        if self.mode == "csv_canonical":
            return int(self.name2id.get(_norm_key(tname), 0))
        # mermaid_api: class ids are keyed by case-folded API name
        return int(self.name2id.get(_norm_key(tname), 0))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_classes": len(self.id2name),
            "id2name": {str(k): v for k, v in sorted(self.id2name.items())},
            "name2id": self.name2id,
        }


def build_labelspace(
    mode: LabelspaceMode,
    csv_path: str | Path,
) -> Labelspace:
    """Build :class:`Labelspace` from CSV + optional MERMAID API."""
    df = load_class_to_concepts_csv(csv_path)
    if mode == "csv_canonical":
        term = df[_terminal_rows_mask(df)]
        names = sorted({str(x).strip().lower() for x in term["name"].tolist() if str(x).strip()})
        id2name = {i: names[i - 1] for i in range(1, len(names) + 1)}
        name2id = {_norm_key(n): i for i, n in id2name.items()}
        return Labelspace(mode=mode, id2name=id2name, name2id=name2id, df=df)

    if mode == "mermaid_api":
        api_names = fetch_mermaid_benthic_attribute_names()
        id2name = {i: n for i, n in enumerate(api_names, start=1)}
        nid: dict[str, int] = {}
        fold_map: dict[str, str] = {}
        for i, n in id2name.items():
            f = _norm_key(n)
            nid[f] = i
            fold_map.setdefault(f, n)
        return Labelspace(
            mode=mode,
            id2name=id2name,
            name2id=nid,
            api_fold_to_canonical=fold_map,
            df=df,
        )

    raise ValueError(f"Unknown mode {mode!r}")


def _row_for_canonical_class(df: pd.DataFrame, canonical_name: str) -> pd.Series:
    """Pick a terminal CSV row for a canonical class name (first match)."""
    term = df[_terminal_rows_mask(df)]
    nkey = _norm_key(canonical_name)
    for _, row in term.iterrows():
        if _norm_key(str(row["name"])) == nkey:
            return row
    raise KeyError(f"No terminal CSV row for canonical class {canonical_name!r}")


def _terminal_row_by_norm_name(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Map case-folded terminal ``name`` → one representative terminal row."""
    term = df[_terminal_rows_mask(df)]
    out: dict[str, pd.Series] = {}
    for _, row in term.iterrows():
        k = _norm_key(str(row["name"]))
        out.setdefault(k, row)
    return out


def _csv_rows_named(df: pd.DataFrame, nkey: str) -> pd.DataFrame:
    """All rows whose ``name`` matches ``nkey`` (normalized)."""
    return df[df["name"].map(lambda x: _norm_key(str(x)) == nkey)]


def _prefer_mermaid_source(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    pref = rows[rows["source"].map(lambda x: _norm_key(str(x)) == "mermaid")]
    if not pref.empty:
        return pref.iloc[0]
    return rows.iloc[0]


def _representative_csv_row_for_joint_class(
    df: pd.DataFrame,
    labelspace: Labelspace,
    joint_class_id: int,
    terminal_index: dict[str, pd.Series] | None = None,
) -> pd.Series | None:
    """Find a CSV row whose concepts describe this joint class id."""
    raw_name = labelspace.id2name[joint_class_id]
    if labelspace.mode == "csv_canonical":
        try:
            return _row_for_canonical_class(df, raw_name)
        except KeyError:
            return None

    nkey = _norm_key(raw_name)
    idx = terminal_index if terminal_index is not None else _terminal_row_by_norm_name(df)
    hit = idx.get(nkey)
    if hit is not None:
        return hit

    # Non-terminal rows (e.g. mermaid "acropora" -> coralscapes "acropora alive") are not in ``idx``.
    direct = _csv_rows_named(df, nkey)
    row_direct = _prefer_mermaid_source(direct)
    if row_direct is not None:
        return row_direct

    # Row whose mapping chain terminates at this API name (terminal spelling).
    for _, row in df.iterrows():
        try:
            tname, _s = resolve_mapping_chain(df, str(row["source"]), str(row["name"]))
        except ValueError:
            continue
        if _norm_key(tname) == nkey:
            return idx.get(_norm_key(str(tname)))
    return None


def build_class_lut(
    labelspace: Labelspace,
    taxonomy: dict[str, TaxonomyLevelSpec],
    binary_spec: BinaryConceptSpec,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """One row per class id (1..N) with taxonomy ids and trinary concept columns."""
    df = labelspace.df if labelspace.df is not None else load_class_to_concepts_csv(csv_path or "")
    rows: list[dict[str, Any]] = []
    term_idx = _terminal_row_by_norm_name(df) if labelspace.mode == "mermaid_api" else None

    for cid in sorted(labelspace.id2name.keys()):
        raw_name = labelspace.id2name[cid]
        row = _representative_csv_row_for_joint_class(df, labelspace, cid, term_idx)

        if row is None:
            logger.warning(
                "build_class_lut: no class_to_concepts.csv row for joint label class_id=%s name=%r "
                "(mermaid_api: add a row covering this MERMAID name, or concepts stay all-ignore)",
                cid,
                raw_name,
            )
            rec = {"class_id": cid, "canonical_name": raw_name}
            for tcol in TAXONOMY_COLUMNS:
                rec[f"tax_{tcol}"] = 0
            for bcol in binary_spec.columns:
                rec[f"bin_{bcol}"] = 0
            rows.append(rec)
            continue

        rec = {"class_id": cid, "canonical_name": _norm_key(str(row["name"]))}
        for tcol in TAXONOMY_COLUMNS:
            spec = taxonomy[tcol]
            rec[f"tax_{tcol}"] = parse_taxonomy_cell_to_id(row.get(tcol), spec.value2id)
        for bcol in binary_spec.columns:
            rec[f"bin_{bcol}"] = parse_binary_cell_to_trinary(row.get(bcol))
        rows.append(rec)

    return pd.DataFrame(rows).set_index("class_id")


def taxonomy_lut_arrays(class_lut: pd.DataFrame) -> np.ndarray:
    """Shape ``(n_classes + 1, 7)`` uint8/uint16 — row 0 unused (ignore)."""
    n = int(class_lut.index.max()) + 1
    out = np.zeros((n, len(TAXONOMY_COLUMNS)), dtype=np.int32)
    for cid in class_lut.index:
        vals = [int(class_lut.loc[cid, f"tax_{c}"]) for c in TAXONOMY_COLUMNS]
        out[int(cid), :] = vals
    return out


def binary_concept_lut_array(class_lut: pd.DataFrame, binary_columns: list[str]) -> np.ndarray:
    """Shape ``(n_classes + 1, n_concepts)`` uint8; row 0 = zeros."""
    n = int(class_lut.index.max()) + 1
    k = len(binary_columns)
    out = np.zeros((n, k), dtype=np.uint8)
    for cid in class_lut.index:
        vals = [int(class_lut.loc[cid, f"bin_{c}"]) for c in binary_columns]
        out[int(cid), :] = vals
    return out


def write_labelspace_artifacts(
    out_dir: str | Path,
    labelspace: Labelspace,
    taxonomy: dict[str, TaxonomyLevelSpec],
    binary_spec: BinaryConceptSpec,
    class_lut: pd.DataFrame,
) -> None:
    """Write ``labelspace.json``, ``taxonomy_levels.json``, ``concepts_binary.json``, ``class_lut.parquet``."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    (root / "labelspace.json").write_text(json.dumps(labelspace.to_json_dict(), indent=2), encoding="utf-8")
    tax_doc = {lev: spec.to_json_dict() for lev, spec in taxonomy.items()}
    (root / "taxonomy_levels.json").write_text(
        json.dumps({"level_order": list(TAXONOMY_COLUMNS), "levels": tax_doc}, indent=2),
        encoding="utf-8",
    )
    (root / "concepts_binary.json").write_text(json.dumps(binary_spec.to_json_dict(), indent=2), encoding="utf-8")
    class_lut.to_parquet(root / "class_lut.parquet")
