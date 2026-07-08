"""Rebuild ``configs/coralnet_to_mermaid_mapping_temporary.json`` from the CoralNet manifest.

Keys the mapping by the *exact* set of CoralNet label names present in the training manifest the
``CoralNetDataset`` actually loads (lowercased ``coralnet_id`` -> name via ``coralnet_id2name.json``,
mirroring ``CoralNetDataset.load_annotations``). Existing label -> MERMAID mappings are preserved;
labels that appear in the manifest but have no mapping yet are written with a ``null`` value so they
are visible and fillable.

Usage (resolves the same default manifest as the dataset; honours
``MERMAID_CORALNET_MANIFEST_PATH`` / ``MERMAID_CORALNET_MANIFEST_VERSION`` and
``MERMAIDSEG_LOCAL_CACHE_DIR``)::

    uv run python scripts/build_coralnet_to_mermaid_mapping.py
    uv run python scripts/build_coralnet_to_mermaid_mapping.py --keep-orphans --minify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mermaidseg.datasets.coralnet.coralnet_dataset import (
    _CORALNET_ID2NAME_KEY,
    _resolve_default_manifest_path,
)
from mermaidseg.datasets.local_cache import LocalS3Cache

DEFAULT_BUCKET = "dev-datamermaid-sm-sources"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "configs" / "coralnet_to_mermaid_mapping_temporary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="S3 key (relative to the bucket) of the training manifest. "
        "Defaults to the same resolution the dataset uses.",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket holding the manifest.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Path to the JSON file to (re)write."
    )
    parser.add_argument(
        "--keep-orphans",
        action="store_true",
        help="Also retain existing keys that are NOT in the current manifest "
        "(default drops them so the key space matches the manifest exactly).",
    )
    parser.add_argument(
        "--minify",
        action="store_true",
        help="Write compact JSON (default is sorted + indented for easy manual editing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_path or _resolve_default_manifest_path()

    cache = LocalS3Cache.configure_from_env()
    print(f"Reading manifest s3://{args.bucket}/{manifest_path}", flush=True)
    df = cache.read_parquet(args.bucket, manifest_path)
    id2name = cache.read_json(args.bucket, _CORALNET_ID2NAME_KEY)

    # Mirror CoralNetDataset.load_annotations: coralnet_id -> name -> lowercase.
    manifest_labels = sorted(
        {
            str(id2name[str(cid)]).lower()
            for cid in df["coralnet_id"].dropna().unique()
            if id2name.get(str(cid)) is not None
        }
    )

    existing: dict[str, str | None] = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text())

    rebuilt: dict[str, str | None] = {label: existing.get(label) for label in manifest_labels}

    orphans = sorted(set(existing) - set(manifest_labels))
    if args.keep_orphans:
        for label in orphans:
            rebuilt[label] = existing[label]

    rebuilt = dict(sorted(rebuilt.items()))

    new_labels = [label for label in manifest_labels if label not in existing]
    unmapped = [label for label, target in rebuilt.items() if target is None]

    if args.minify:
        args.output.write_text(json.dumps(rebuilt, ensure_ascii=False))
    else:
        args.output.write_text(json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rebuilt)} entries to {args.output}", flush=True)
    print(f"  manifest labels:        {len(manifest_labels)}", flush=True)
    print(f"  newly added (null):     {len(new_labels)}", flush=True)
    print(f"  total unmapped (null):  {len(unmapped)}", flush=True)
    print(
        f"  orphan keys not in manifest: {len(orphans)} "
        f"({'kept' if args.keep_orphans else 'dropped'})",
        flush=True,
    )
    if new_labels:
        print("\nNew manifest labels needing a MERMAID mapping:", flush=True)
        for label in new_labels:
            print(f"  {label!r}", flush=True)


if __name__ == "__main__":
    main()
