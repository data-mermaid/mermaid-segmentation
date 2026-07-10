"""Class-frequency weights from annotation parquet counts (no image I/O).

Ticket 2a (1A): count labels in CoralNet/MERMAID annotation tables, map to target ids,
and write a versioned JSON artifact that training loads into the loss.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1


def inverse_sqrt_weights(
    counts: np.ndarray,
    *,
    const: float = 2_000_000.0,
    ignore_index: int = 0,
) -> np.ndarray:
    """Inverse-√frequency weights; ignore slot is zeroed and excluded from mean-norm."""
    counts = np.asarray(counts, dtype=np.float64)
    weights = 1.0 / np.sqrt(counts + const)
    if 0 <= ignore_index < len(weights):
        weights[ignore_index] = 0.0
    active = np.ones(len(weights), dtype=bool)
    if 0 <= ignore_index < len(weights):
        active[ignore_index] = False
    if active.any() and weights[active].mean() > 0:
        weights[active] /= weights[active].mean()
    return weights.astype(np.float64)


def counts_from_label_series(
    labels: pd.Series,
    name_to_target: dict[str, int],
    num_classes: int,
) -> np.ndarray:
    """Map source label names to target ids and accumulate annotation counts."""
    counts = np.zeros(num_classes, dtype=np.int64)
    mapped = labels.map(name_to_target).fillna(0).astype(int)
    for tid, count in mapped.value_counts().items():
        tid_i = int(tid)
        if 0 <= tid_i < num_classes:
            counts[tid_i] += int(count)
    return counts


def build_target_name_map(class_subset: list[str]) -> dict[str, int]:
    """Assign target ids 1..N alphabetically (case-sensitive names as in training
    config)."""
    return {name: idx for idx, name in enumerate(sorted(class_subset), start=1)}


def load_source_to_target_name_map(
    mapping_path: Path | str | None,
) -> dict[str, str]:
    """Load coralnet→mermaid (or identity) name map; keys/values lowercased for
    lookup."""
    if mapping_path is None:
        return {}
    path = Path(mapping_path)
    raw = json.loads(path.read_text())
    return {str(k).lower(): str(v) for k, v in raw.items()}


def resolve_source_label_to_target_id(
    source_label: str,
    *,
    source_to_mermaid: dict[str, str],
    target_name_to_id: dict[str, int],
) -> int:
    """Map a source label string to a target id (0 = ignore)."""
    key = str(source_label).lower()
    mermaid_name = source_to_mermaid.get(key, key)
    # Prefer exact case match against class_subset, then case-insensitive.
    if mermaid_name in target_name_to_id:
        return target_name_to_id[mermaid_name]
    lower_map = {n.lower(): i for n, i in target_name_to_id.items()}
    return lower_map.get(mermaid_name.lower(), 0)


def count_parquet_labels(
    parquet_paths: list[Path | str],
    *,
    label_column: str,
    source_to_mermaid: dict[str, str],
    target_name_to_id: dict[str, int],
    num_classes: int,
) -> np.ndarray:
    """Accumulate target-id counts from one or more annotation parquet files."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in parquet_paths:
        path = Path(path)
        df = pd.read_parquet(path, columns=[label_column])
        series = df[label_column].astype(str)
        name_to_target = {
            name: resolve_source_label_to_target_id(
                name,
                source_to_mermaid=source_to_mermaid,
                target_name_to_id=target_name_to_id,
            )
            for name in series.unique()
        }
        counts += counts_from_label_series(series, name_to_target, num_classes)
        logger.info("Counted labels from %s (%d rows)", path, len(df))
    return counts


def build_weight_artifact(
    *,
    counts: np.ndarray,
    target_id2label: dict[int, str],
    class_subset: list[str],
    const: float = 2_000_000.0,
    ignore_index: int = 0,
    mapping_path: str | None = None,
    git_sha: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the versioned class-weight artifact dict."""
    weights = inverse_sqrt_weights(counts, const=const, ignore_index=ignore_index)
    labels = ["ignore"] + [target_id2label[i] for i in sorted(target_id2label)]
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "recipe": {
            "type": "inverse_sqrt_freq",
            "const": const,
            "ignore_index": ignore_index,
            "note": (
                "Counts from annotation parquet/manifests (point labels), not "
                "padded pixel masks (padding: 3)."
            ),
        },
        "class_subset": list(class_subset),
        "target_id2label": {str(k): v for k, v in sorted(target_id2label.items())},
        "counts": [int(c) for c in counts.tolist()],
        "weights": [float(w) for w in weights.tolist()],
        "labels": labels[: len(counts)],
        "mapping_path": mapping_path,
        "git_sha": git_sha,
        "sources": sources or [],
    }
    payload["mapping_hash"] = hashlib.sha256(
        json.dumps(
            {
                "class_subset": payload["class_subset"],
                "mapping_path": mapping_path,
                "target_id2label": payload["target_id2label"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    return payload


def save_weight_artifact(artifact: dict[str, Any], path: Path | str) -> Path:
    """Atomically write the weight artifact JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    logger.info("Wrote class-weight artifact to %s", path)
    return path


def load_class_weight_tensor(path: Path | str) -> torch.Tensor:
    """Load ``weights`` from a class-weight artifact JSON as a float tensor."""
    path = Path(path)
    artifact = json.loads(path.read_text())
    if "weights" not in artifact:
        raise ValueError(f"Class-weight artifact missing 'weights': {path}")
    return torch.tensor(artifact["weights"], dtype=torch.float32)
