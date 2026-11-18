"""Export a trained checkpoint into a self-describing bundle (``model.json`` + weights)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mermaidseg.export import export_bundle


def main(argv: list[str] | None = None) -> None:
    """Export ``--checkpoint`` into ``--output-dir`` and print the bundle path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(export_bundle(args.checkpoint, args.output_dir))


if __name__ == "__main__":
    main()
