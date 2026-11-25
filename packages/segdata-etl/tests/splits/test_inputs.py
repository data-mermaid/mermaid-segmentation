from __future__ import annotations

import pytest
from s3_fake import FakeS3

from segdata_etl.splits.inputs import class_counts, images_per_source, sync_tables
from segdata_spec.errors import PublishRefused
from segdata_spec.taxonomy import hierarchy_from_table

from .conftest import (
    BUCKET,
    FixtureCache,
    benthic_attributes_table,
    images_table,
    label_counts_table,
    labels_table,
)

HIERARCHY = hierarchy_from_table(
    benthic_attributes_table(
        [("1", "Hard coral", None), ("2", "Acropora", "1"), ("3", "Sand", None)]
    )
)
IMAGES = images_table([("i1", "s10", "A=", 1), ("i2", "s10", "B=", 1), ("i3", "s20", "C=", 1)])


def test_class_counts_roll_up_mapped_labels_into_the_subset():
    labels = labels_table([(1, "Acropora"), (2, None), (3, "Sand")])
    counts = label_counts_table(
        [("i1", 1, 3), ("i2", 1, 2), ("i1", 2, 5), ("i3", 3, 4), ("i3", 1, 1)]
    )
    result = class_counts(counts, labels, IMAGES, HIERARCHY, {"hard coral"})
    # Label 2 is unmapped and "sand" has no ancestor in the subset: neither is counted.
    assert result.to_dict() == {"hard coral": {10: 5, 20: 1}}


def test_images_per_source_counts_images_per_source_id():
    assert images_per_source(IMAGES).to_dict() == {10: 2, 20: 1}


def test_sync_tables_refuses_a_taxonomy_cached_but_not_published(fixture_cache: FixtureCache):
    pin = fixture_cache.taxonomy(
        "20261120-1234567", benthic_attributes_table([("1", "Sand", None)])
    )
    nothing_published = FakeS3()

    with pytest.raises(PublishRefused, match="not published"):
        sync_tables(nothing_published, {}, fixture_cache.cache_dir, BUCKET, pin)

    assert nothing_published.ops("put") == []


def test_sync_tables_without_a_taxonomy_reads_no_taxonomy_manifest(fixture_cache: FixtureCache):
    client = FakeS3()

    assert sync_tables(client, {}, fixture_cache.cache_dir, BUCKET) == {}

    assert [k for _, _, k, _ in client.calls if k.startswith("taxonomy/")] == []
