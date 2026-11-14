"""Source counts of the CoralNet data at each curation stage, on the real manifest and configs.

1. Initial selection (``configs/data_config_all.yaml``, 863 sources): exactly the manifest sources
   with at least 21 images, 500 confirmed points and 5 distinct labels. This rule was
   reverse-engineered: the list was committed without the code that made it, and no other
   thresholds near these give the same set (there are sources at 20 images, 4 labels and 490
   points that are not in it).
2. Manual exclusions (``configs/coralnet/low_quality_sources.txt``, 37 sources): 826 sources.
3. Deduplication (``configs/data_config_dedup.yaml`` + ``image_blacklist.parquet``): the 826
   sources are split 566 / 116 / 144, but 85 train sources lose every image (all their images are
   copies kept from another source), so 741 sources and 514,623 images remain.

Needs the training manifest in the local cache (``MERMAIDSEG_LOCAL_CACHE_DIR``); skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from mermaidseg.datasets.coralnet.dedup.common import (
    CONFIG_DIR,
    DEFAULT_BUCKET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_MANIFEST,
    REPO_ROOT,
    coralnet_splits,
    read_source_list,
)

MANIFEST = Path(DEFAULT_CACHE_ROOT) / DEFAULT_BUCKET / DEFAULT_MANIFEST

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not MANIFEST.exists(), reason=f"manifest not in local cache: {MANIFEST}"),
]

MIN_IMAGES, MIN_POINTS, MIN_LABELS = 21, 500, 5


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    df = pq.read_table(MANIFEST, columns=["source_id", "image_id", "coralnet_id"]).to_pandas()
    df["image_id"] = df["image_id"].astype(str)
    return df


@pytest.fixture(scope="module")
def per_source(manifest) -> pd.DataFrame:
    g = manifest.groupby("source_id")
    return pd.DataFrame({"images": g["image_id"].nunique(), "points": g.size(), "labels": g["coralnet_id"].nunique()})


def _union(splits: dict[str, set[int]]) -> set[int]:
    return set().union(*splits.values())


def test_initial_selection_is_size_rule(per_source):
    initial = _union(coralnet_splits(REPO_ROOT / "configs/data_config_all.yaml"))
    assert len(initial) == 863

    s = per_source
    rule = set(s.index[(s["images"] >= MIN_IMAGES) & (s["points"] >= MIN_POINTS) & (s["labels"] >= MIN_LABELS)])
    assert rule == initial

    # The thresholds are tight: loosening any one of them adds sources.
    assert ((s["images"] == MIN_IMAGES - 1) & (s["points"] >= MIN_POINTS) & (s["labels"] >= MIN_LABELS)).any()
    assert ((s["labels"] == MIN_LABELS - 1) & (s["images"] >= MIN_IMAGES) & (s["points"] >= MIN_POINTS)).any()
    assert ((s["points"] < MIN_POINTS) & (s["images"] >= MIN_IMAGES) & (s["labels"] >= MIN_LABELS)).any()


def test_manual_exclusions():
    initial = _union(coralnet_splits(REPO_ROOT / "configs/data_config_all.yaml"))
    excluded = read_source_list(CONFIG_DIR / "low_quality_sources.txt")
    assert len(excluded) == 37
    assert excluded <= initial
    assert len(initial - excluded) == 826


def test_dedup_splits_and_remaining_sources(manifest):
    initial = _union(coralnet_splits(REPO_ROOT / "configs/data_config_all.yaml"))
    excluded = read_source_list(CONFIG_DIR / "low_quality_sources.txt")
    splits = coralnet_splits(REPO_ROOT / "configs/data_config_dedup.yaml")

    assert {k: len(v) for k, v in splits.items()} == {"train": 566, "val": 116, "test": 144}
    assert not (splits["train"] & splits["val"] or splits["train"] & splits["test"] or splits["val"] & splits["test"])
    assert _union(splits) == initial - excluded

    blacklist = set(pd.read_parquet(CONFIG_DIR / "image_blacklist.parquet")["image_id"].astype(str))
    kept = manifest[manifest["source_id"].isin(_union(splits)) & ~manifest["image_id"].isin(blacklist)]
    kept = kept.drop_duplicates(["source_id", "image_id"])

    assert kept["source_id"].nunique() == 741
    assert len(kept) == 514_623

    remaining = {sp: set(kept.loc[kept["source_id"].isin(ids), "source_id"]) for sp, ids in splits.items()}
    assert {k: len(v) for k, v in remaining.items()} == {"train": 481, "val": 116, "test": 144}
    # Only train sources are emptied: val/test sources share no image with any other source.
    assert splits["val"] <= remaining["val"] and splits["test"] <= remaining["test"]
