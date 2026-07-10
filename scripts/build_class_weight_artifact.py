#!/usr/bin/env python3
"""Build a versioned class-weight artifact from annotation parquet/manifests.

Counts labels without loading images (Ticket 2a / 1A). Example::

    uv run python scripts/build_class_weight_artifact.py \\
      --class-subset-from configs/training_config_dinov3_base.yaml \\
      --mapping configs/coralnet_to_mermaid_mapping_temporary.json \\
      --coralnet-parquet /path/to/coralnet_training_manifest.parquet \\
      --coralnet-label-column source_label_name \\
      --mermaid-parquet /path/to/mermaid_confirmed_annotations.parquet \\
      --mermaid-label-column benthic_attribute_name \\
      --output artifacts/class_weights_dinov3_base.json
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from mermaidseg.model.class_weights import (
    build_target_name_map,
    build_weight_artifact,
    count_parquet_labels,
    load_source_to_target_name_map,
    save_weight_artifact,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _load_class_subset(path: Path) -> list[str]:
    cfg = yaml.safe_load(path.read_text())
    subset = cfg.get("training", cfg).get("class_subset")
    if not subset:
        raise ValueError(f"No training.class_subset in {path}")
    return list(subset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-subset-from",
        type=Path,
        required=True,
        help="Training YAML containing training.class_subset",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional coralnet→mermaid JSON map (name or id keys)",
    )
    parser.add_argument(
        "--coralnet-parquet",
        type=Path,
        action="append",
        default=[],
        help="CoralNet annotation/manifest parquet (repeatable)",
    )
    parser.add_argument(
        "--coralnet-label-column",
        default="source_label_name",
        help="Label column in CoralNet parquet (default: source_label_name)",
    )
    parser.add_argument(
        "--mermaid-parquet",
        type=Path,
        action="append",
        default=[],
        help="MERMAID annotation parquet (repeatable)",
    )
    parser.add_argument(
        "--mermaid-label-column",
        default="benthic_attribute_name",
        help="Label column in MERMAID parquet",
    )
    parser.add_argument(
        "--const",
        type=float,
        default=2_000_000.0,
        help="Inverse-√freq smoother (default: 2000000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not args.coralnet_parquet and not args.mermaid_parquet:
        parser.error("Provide at least one --coralnet-parquet or --mermaid-parquet")

    class_subset = _load_class_subset(args.class_subset_from)
    target_name_to_id = build_target_name_map(class_subset)
    target_id2label = {i: n for n, i in target_name_to_id.items()}
    num_classes = len(class_subset) + 1  # + ignore slot
    source_to_mermaid = load_source_to_target_name_map(args.mapping)

    counts = np.zeros(num_classes, dtype=np.int64)
    sources: list[dict] = []

    if args.coralnet_parquet:
        part = count_parquet_labels(
            args.coralnet_parquet,
            label_column=args.coralnet_label_column,
            source_to_mermaid=source_to_mermaid,
            target_name_to_id=target_name_to_id,
            num_classes=num_classes,
        )
        counts = counts + part
        sources.append(
            {
                "dataset": "coralnet",
                "paths": [str(p) for p in args.coralnet_parquet],
                "label_column": args.coralnet_label_column,
            }
        )

    if args.mermaid_parquet:
        # MERMAID labels are already benthic names; identity map via empty overlay.
        part = count_parquet_labels(
            args.mermaid_parquet,
            label_column=args.mermaid_label_column,
            source_to_mermaid={},
            target_name_to_id=target_name_to_id,
            num_classes=num_classes,
        )
        counts = counts + part
        sources.append(
            {
                "dataset": "mermaid",
                "paths": [str(p) for p in args.mermaid_parquet],
                "label_column": args.mermaid_label_column,
            }
        )

    artifact = build_weight_artifact(
        counts=counts,
        target_id2label=target_id2label,
        class_subset=class_subset,
        const=args.const,
        mapping_path=str(args.mapping) if args.mapping else None,
        git_sha=_git_sha(),
        sources=sources,
    )
    save_weight_artifact(artifact, args.output)
    logger.info(
        "Active classes with counts>0: %d / %d",
        int((counts[1:] > 0).sum()),
        len(class_subset),
    )


if __name__ == "__main__":
    main()
