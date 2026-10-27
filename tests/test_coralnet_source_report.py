"""Source selection for the CoralNet inspection report. No S3 access."""

from scripts.coralnet_source_report import select_sources

SPLITS = {1: "train", 2: "val", 3: "test", 4: "train"}


def test_select_sources_keeps_the_whole_manifest_by_default() -> None:
    assert select_sources(SPLITS, only_splits=None, only_sources=None) is None
    assert select_sources(SPLITS, only_splits=None, only_sources=[3, 1]) == [3, 1]


def test_select_sources_filters_to_whitelisted_splits() -> None:
    assert select_sources(SPLITS, only_splits=["val", "test"], only_sources=None) == [2, 3]
    assert select_sources(SPLITS, only_splits=["train"], only_sources=[1, 2]) == [1]
    assert select_sources(SPLITS, only_splits=["test"], only_sources=[1]) == []
