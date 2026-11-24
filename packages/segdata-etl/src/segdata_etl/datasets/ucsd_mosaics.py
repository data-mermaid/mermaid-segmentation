"""``segdata build ucsd_mosaics``: stage the UCSD mosaics mirror as a dense dataset.

The build reads image and label from the pinned Hugging Face revision, never writes to S3, and leaves
a version under ``--stage-dir`` plus ``uploads.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from segdata_etl.common import hf_dense
from segdata_etl.common.hf_dense import DenseSource, RowInfo

logger = logging.getLogger(__name__)

DATASET = "ucsd_mosaics"
REVISION = "6a6483841f8ee20bd7d8ee2aa760f10d1e9db704"


def _vocab(raw: bytes) -> dict[int, str]:
    return {int(c["id"]): c["name"] for c in json.loads(raw)}


def _describe(row: Mapping[str, Any], split: str) -> RowInfo:
    del split  # the site column names the group
    return RowInfo(
        group=row["site"].lower(),
        stem=hf_dense.stem_of(row["filename"]),
        source_name=row["filename"],
        attrs={"site": row["site"], "original_split": row["original_split"]},
    )


def _assign_ids(infos: Sequence[RowInfo]) -> dict[tuple[str, str], str]:
    """Use the bare stem as the id when stems are unique across sites, else ``<group>/<stem>``.

    Raises:
        ValueError: A ``(group, stem)`` pair repeats, so two rows would share an id.
    """
    pairs = Counter((i.group, i.stem) for i in infos)
    repeated = sorted(p for p, n in pairs.items() if n > 1)
    if repeated:
        raise ValueError(f"ucsd_mosaics (site, stem) pairs are not unique: {repeated[:5]}")
    shared = any(n > 1 for n in Counter(i.stem for i in infos).values())
    logger.info("image ids are %s", "group-qualified" if shared else "bare stems")
    return {(i.group, i.stem): f"{i.group}/{i.stem}" if shared else i.stem for i in infos}


SOURCE = DenseSource(
    dataset=DATASET,
    repo="josauder/UCSD-mosaics-mirror",
    revision=REVISION,
    license=f"research-only (LICENSE file absent at revision {REVISION[:7]})",
    citation=(
        "Edwards et al. 2017, Coral Reefs 36; Alonso et al. 2019, J. Field Robotics 36; "
        "Raine et al. 2024, CVPR"
    ),
    vocab_file="classes.json",
    shard_dir="data/",
    shard_pattern=re.compile(r"data/([^/]+)-\d+-of-\d+\.parquet"),
    meta_columns=("filename", "site", "original_split"),
    parse_vocab=_vocab,
    describe=_describe,
    assign_ids=_assign_ids,
)


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the ``--revision`` flag; ``workdir`` is unused because the build stages under ``--stage-dir``."""
    del workdir
    hf_dense.add_revision_argument(parser, SOURCE)


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a UCSD mosaics version; ``client`` only checks the bucket for version clashes."""
    return hf_dense.run_build(SOURCE, args, client)
