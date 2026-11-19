r"""Build a fixed CoralscapesV2 linear-probe point set.

Usage:
    uv run python scripts/build_coralscapes_v2_probe_points.py \
        --local-root /path/to/coralscapes_v2 \
        --output configs/eval/coralscapes_v2_probe_points.json \
        --per-class 10 \
        --min-component-size 4 \
        --exclusion-radius 16

    uv run python scripts/build_coralscapes_v2_probe_points.py \
        --local-root /path/to/coralscapes_v2 \
        --output configs/eval/coralscapes_v2_39_probe_k50.json \
        --per-class 50 \
        --label-space legacy39
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mermaidseg.evaluation.coralscapes_probe_points import build_probe_points


def main() -> None:
    """Parse the command line and run :func:`build_probe_points`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        required=True,
        help="Cityscapes-style CoralscapesV2 mirror root (leftImg8bit + gtFine)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/eval/coralscapes_v2_probe_points.json"),
    )
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--min-component-size", type=int, default=4)
    parser.add_argument("--exclusion-radius", type=int, default=16)
    parser.add_argument(
        "--label-space",
        choices=("v2", "legacy39"),
        default="v2",
        help="v2 keeps native ids 1..95. legacy39 remaps masks onto the original 39 classes "
        "before selecting components. label_md5 stays the hash of the original mask.",
    )
    args = parser.parse_args()
    build_probe_points(
        local_root=args.local_root,
        output=args.output,
        per_class=args.per_class,
        min_component_size=args.min_component_size,
        exclusion_radius=args.exclusion_radius,
        label_space=args.label_space,
    )


if __name__ == "__main__":
    main()
