#!/usr/bin/env python3
"""Write the CoralNet source x binary concept pairs to mask in training.

A pair is masked when the source has False points for the concept and no True points
(``only_false``). Points come from the ``points.parquet`` of the ``coralnet`` version that
``--data-config`` pins (default: the final-run mixture ``configs/data_config.yaml``), read from the
synced ``--cache-dir``, over the groups of its ``--split`` block and selected by that block's
``annotators`` as training selects them; an image's source is its group ``s<source id>``. Concept values come from the frozen concept schema training uses (``--concepts``
restricted to the datasets of ``--schema-config``). No images are read.

Outputs ``--out`` (one row per masked pair: source_id, concept, n_true, n_false, n_not_given) and a
``.meta.yaml`` sidecar recording the inputs, their sha1s and the counted sources (``source_ids``).
Training refuses a CoralNet source that is not in ``source_ids``, so rebuild the mask whenever the
CoralNet train sources of the run's data config change.

    python scripts/build_concept_source_mask.py --cache-dir /path/to/cache \
        --out configs/concept_source_masks/coralnet_only_false.csv
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
from collections.abc import Collection, Sequence
from pathlib import Path

import pandas as pd
import yaml

from mermaidseg.data.config import DataSection, Split, entry_split_groups
from mermaidseg.data.factory import open_dataset_version
from mermaidseg.data.point_dataset import select_points
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema, concept_label_key
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    MASK_COLUMNS,
    _source_sort_key,
    file_sha1,
    meta_path,
    only_false_pairs,
    source_concept_counts,
)
from mermaidseg.dataset_reconciliation.concepts import DEFAULT_CLASS_TO_CONCEPTS_CSV
from mermaidseg.io import load_section
from mermaidseg.run_setup import DEFAULT_SCHEMA_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "coralnet"


def _rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def group_source_ids(groups: pd.Series) -> pd.Series:
    """Map CoralNet groups ``s<id>`` to integer source ids.

    Raises:
        ValueError: A group is not ``s`` followed by digits.
    """
    ids = groups.astype(str).str.removeprefix("s")
    bad = sorted(set(groups[~ids.str.fullmatch(r"\d+")].astype(str)))
    if bad:
        raise ValueError(f"CoralNet groups are not s<source id>: {bad[:5]}")
    return ids.astype(int)


def concept_keys(dataset: str, labels: pd.DataFrame) -> pd.Series:
    """Map each ``label_id`` of ``labels`` to its concept label key (see ``concept_label_key``).

    Args:
        dataset: Dataset name.
        labels: ``labels.parquet`` rows (``label_id``, ``name``, ``mermaid_benthic_attribute``).
    """
    keys = [
        concept_label_key(dataset, name, target)
        for name, target in zip(labels["name"], labels["mermaid_benthic_attribute"], strict=True)
    ]
    return pd.Series(keys, index=labels["label_id"].to_numpy())


def point_annotations(
    dataset: str,
    points: pd.DataFrame,
    images: pd.DataFrame,
    labels: pd.DataFrame,
    groups: Collection[str] | None,
    annotators: Sequence[str] | None,
) -> pd.DataFrame:
    """Return one row per point of the images in ``groups``: ``source_id``, ``source_label_name``.

    Points are selected by annotator as training selects them (``select_points``), and
    ``source_label_name`` is each point's concept label key (:func:`concept_keys`).

    Args:
        dataset: Dataset name.
        points: ``points.parquet`` rows (``image_id``, ``row``, ``col``, ``label_id``,
            ``annotator``).
        images: ``images.parquet`` rows (``image_id``, ``group``, ``height``, ``width``).
        labels: ``labels.parquet`` rows (``label_id``, ``name``, ``mermaid_benthic_attribute``).
        groups: Groups to keep; ``None`` keeps all.
        annotators: The split block's annotators in priority order; ``None`` requires one.

    Raises:
        ValueError: A group is not in ``images``, a group is not ``s<source id>``, the annotator
            selection fails, or a point's label id is missing from ``labels``.
    """
    if groups is not None:
        unknown = sorted(set(groups) - set(images["group"]))
        if unknown:
            raise ValueError(f"groups not in the dataset version: {unknown}")
        images = images[images["group"].isin(list(groups))]
    kept = select_points(points, images, annotators)
    keys = kept["label_id"].map(concept_keys(dataset, labels))
    unnamed = sorted(set(kept.loc[keys.isna(), "label_id"]))
    if unnamed:
        raise ValueError(f"{dataset}: label ids missing from labels.parquet: {unnamed}")
    group_of = images.set_index("image_id")["group"]
    return pd.DataFrame(
        {
            "source_id": group_source_ids(kept["image_id"].map(group_of)).to_numpy(),
            "source_label_name": keys.to_numpy(),
        }
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and build the concept source mask file."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs/data_config.yaml")
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory scripts/sync_data.py synced --data-config into",
    )
    p.add_argument("--concepts", type=Path, default=Path(DEFAULT_CLASS_TO_CONCEPTS_CSV))
    p.add_argument("--schema-config", type=Path, default=Path(DEFAULT_SCHEMA_CONFIG))
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "configs/concept_source_masks/coralnet_only_false.csv",
    )
    p.add_argument(
        "--counts-out",
        type=Path,
        default=None,
        help="optional CSV with the counts of every source x binary concept pair (masked or not)",
    )
    args = p.parse_args(argv)
    split: Split = args.split

    section = DataSection.model_validate(load_section(args.data_config, "data"))
    version = open_dataset_version(section, DATASET, args.cache_dir)
    if version.points is None:
        raise SystemExit(f"{DATASET} {version.manifest.version} has no points.parquet")
    entry = section.datasets[DATASET]
    groups = entry_split_groups(entry, split, REPO_ROOT)
    annotations = point_annotations(
        DATASET,
        version.points.to_pandas(),
        version.images.to_pandas(),
        version.labels.to_pandas(),
        groups,
        getattr(entry, split).annotators,
    )

    schema_sources = set(load_section(args.schema_config, "data")["datasets"])
    schema = ConceptSchema.from_csv(args.concepts, sources=schema_sources)

    counts = source_concept_counts(annotations, schema, dataset_source=DATASET)
    masked = only_false_pairs(counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    masked[MASK_COLUMNS].to_csv(args.out, index=False)
    if args.counts_out is not None:
        counts.to_csv(args.counts_out, index=False)

    n_sources = int(counts["source_id"].nunique())
    n_concepts = int(counts["concept"].nunique())
    meta = {
        "rule": "only_false",
        "rule_description": "source has n_false > 0 and n_true == 0 for the binary concept",
        "dataset": DATASET,
        "dataset_version": version.manifest.version,
        "split": split,
        "data_config": _rel(args.data_config),
        "data_config_sha1": file_sha1(args.data_config),
        "concepts_csv": _rel(args.concepts),
        "concepts_csv_sha1": file_sha1(args.concepts),
        "schema_config": _rel(args.schema_config),
        "n_sources": n_sources,
        "n_binary_concepts": n_concepts,
        "n_pairs": n_sources * n_concepts,
        "n_masked": len(masked),
        "n_points": len(annotations),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source_ids": sorted({str(s) for s in counts["source_id"]}, key=_source_sort_key),
    }
    meta_path(args.out).write_text(yaml.safe_dump(meta, sort_keys=False))

    print(
        f"{len(masked)} of {n_sources} sources x {n_concepts} binary concepts "
        f"= {n_sources * n_concepts} pairs are only_false"
    )
    print(f"wrote {_rel(args.out)} and {_rel(meta_path(args.out))}")


if __name__ == "__main__":
    main()
