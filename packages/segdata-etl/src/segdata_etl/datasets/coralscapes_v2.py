"""``segdata build coralscapes_v2``: stage the Coralscapes V2 ``instances`` config as a dense dataset.

The build reads image and label from the pinned Hugging Face revision, never writes to S3, and leaves
a version under ``--stage-dir`` plus ``uploads.jsonl``. The ``instance_ids`` column is not staged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from segdata_etl.common import hf_dense
from segdata_etl.common.hf_dense import DenseSource, RowInfo

DATASET = "coralscapes_v2"


def _vocab(raw: bytes) -> dict[int, str]:
    return {int(k): v for k, v in json.loads(raw).items()}


def _describe(row: Mapping[str, Any], split: str) -> RowInfo:
    return RowInfo(
        group=split,
        stem=row["stem"],
        source_name=row["stem"],
        attrs={"site": row["site"], "hf_split": split},
    )


def _assign_ids(infos: Sequence[RowInfo]) -> dict[tuple[str, str], str]:
    repeated = sorted(s for s, n in Counter(i.stem for i in infos).items() if n > 1)
    if repeated:
        raise ValueError(f"coralscapes stems are not unique: {repeated[:5]}")
    return {(i.group, i.stem): i.stem for i in infos}


SOURCE = DenseSource(
    dataset=DATASET,
    repo="josauder/coralscapesV2",
    revision="8a4567b2fbec14882e09aebff63a7a1ca5e24dca",
    license="apache-2.0",
    citation=(
        "Sauder et al., CoralscapesV2: Panoptic and Fine-Grained Visual Scene Understanding "
        "in Coral Reefs, ECCV Workshop on Marine Vision, 2026"
    ),
    vocab_file="id2label.json",
    shard_dir="instances/",
    shard_pattern=re.compile(r"instances/(train|validation|test)-.*\.parquet"),
    meta_columns=("stem", "site"),
    parse_vocab=_vocab,
    describe=_describe,
    assign_ids=_assign_ids,
)


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the ``--revision`` flag; ``workdir`` is unused because the build stages under ``--stage-dir``."""
    del workdir
    hf_dense.add_revision_argument(parser, SOURCE)


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a Coralscapes V2 version; ``client`` only checks the bucket for version clashes."""
    return hf_dense.run_build(SOURCE, args, client)
