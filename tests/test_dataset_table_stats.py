"""Counts for the reconciled dataset table. No cache or cluster access."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from scripts.dataset_table_stats import (
    DATASET_SPECS,
    PAPER_COLUMNS,
    STRUCT8,
    DenseCounts,
    SparseCounts,
    add_split,
    apply_dense,
    apply_sparse,
    build_table,
    count_dense_mask,
    count_sparse,
    empty_level,
    label_is_mapped,
    load_mapped_names,
    paper_column,
    strip_transforms,
)

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
_REPO = Path(__file__).resolve().parents[1]


def test_mapped_names_drop_missing_keys_other_and_conflicting_rows(tmp_path):
    path = tmp_path / "class_to_concepts.csv"
    pd.DataFrame(
        {
            "source_label_class_name": [
                "Acropora",
                "off",
                "Sand",
                "Sand",
                "MissingTarget",
                "  Padina ",
            ],
            "source_dataset_source": ["Moorea", "moorea", "MOOREA", "moorea", "moorea", "Pacific"],
            "target_label_class_name": ["acropora", "Other", "sand", "other", None, ""],
        }
    ).to_csv(path, index=False)

    mapped = load_mapped_names(path)

    # Acropora is present and not other. off is other. Sand has one other row.
    # Empty / missing targets stay mapped. Lookup is case-insensitive.
    assert mapped["moorea"] == {"acropora", "missingtarget"}
    assert "sand" not in mapped["moorea"]
    assert "off" not in mapped["moorea"]
    assert mapped["pacific"] == {"padina"}
    assert label_is_mapped("Moorea", "Acropora", mapped)
    assert not label_is_mapped("moorea", "not-in-csv", mapped)
    assert not label_is_mapped("moorea", "Sand", mapped)


def test_sparse_points_match_pixels_and_ignore_unlabeled():
    annotations = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "b", "d"],
            "source_label_name": ["Acropora", "off", "unlabeled", "Sand", "off"],
        }
    )
    images = pd.DataFrame({"image_id": ["a", "b", "c", "d"]})

    counts = count_sparse(annotations, images, {"acropora"})

    # a has a mapped point. b is ignore + unmapped, c has no points, d is only
    # unmapped. Source still counts every image row.
    assert counts.images == 4
    assert counts.mapped_images == 1
    assert counts.source_annotations == 4
    assert counts.source_pixels == counts.source_annotations
    assert counts.source_labels == {"Acropora", "off", "Sand"}
    assert counts.mapped_annotations == 1
    assert counts.mapped_pixels == 1
    assert counts.mapped_labels == {"Acropora"}


def test_diagonal_pixels_are_one_eight_connected_component():
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1

    counts = count_dense_mask(mask, {1: "coral"}, {"coral"})

    assert counts.source_components == 1
    assert counts.source_pixels == 2
    assert counts.mapped_components == 1
    assert counts.mapped_images == 1
    _labeled, n_four = ndimage.label(mask == 1, structure=_CROSS)
    assert int(n_four) == 2
    _labeled, n_eight = ndimage.label(mask == 1, structure=STRUCT8)
    assert int(n_eight) == 1


def test_dense_mask_drops_ignore_ids_and_unmapped_classes():
    mask = np.array(
        [
            [0, 1, 1],
            [2, 2, 3],
            [4, 4, 4],
        ],
        dtype=np.uint8,
    )
    id_to_name = {
        0: "should-ignore",
        1: "Acropora",
        2: "Other Junk",
        3: "unlabeled",
        4: "background",
    }

    counts = count_dense_mask(mask, id_to_name, {"acropora", "background"})

    assert counts.source_components == 3
    assert counts.source_pixels == 2 + 2 + 3
    assert counts.source_labels == {"Acropora", "Other Junk", "background"}
    assert counts.mapped_components == 2
    assert counts.mapped_pixels == 5
    assert counts.mapped_labels == {"Acropora", "background"}
    assert counts.mapped_images == 1
    assert 0 not in counts.unknown_ids


def test_dense_mask_without_mapped_labels_is_not_a_mapped_image():
    unmapped = np.array([[2, 2], [3, 0]], dtype=np.uint8)
    mapped = np.array([[1, 0], [2, 2]], dtype=np.uint8)
    id_to_name = {1: "Acropora", 2: "junk", 3: "unlabeled"}

    unmapped_counts = count_dense_mask(unmapped, id_to_name, {"acropora"})
    mapped_counts = count_dense_mask(mapped, id_to_name, {"acropora"})
    total = DenseCounts()
    total.add(unmapped_counts)
    total.add(mapped_counts)

    assert unmapped_counts.source_components == 1
    assert unmapped_counts.mapped_images == 0
    assert unmapped_counts.mapped_components == 0
    assert mapped_counts.mapped_images == 1
    assert total.mapped_images == 1
    assert total.source_components == unmapped_counts.source_components + 2


def test_unknown_positive_id_is_source_only():
    mask = np.array([[7, 7], [0, 0]], dtype=np.uint8)

    counts = count_dense_mask(mask, {}, set())

    assert counts.unknown_ids == {7}
    assert counts.source_labels == {"id_7"}
    assert counts.source_components == 1
    assert counts.source_pixels == 2
    assert counts.mapped_components == 0
    assert counts.mapped_labels == set()
    assert counts.mapped_images == 0


def test_paper_columns_place_held_out_sets_in_test():
    assert paper_column("moorea_labeled_corals", "train") == "train"
    assert paper_column("catlin_seaview", "train") == "train"
    assert paper_column("ucsd_mosaics", "train") == "train"
    assert paper_column("pacific_labeled_corals", "val") == "test"
    assert paper_column("benthos_yuval", "val") == "test"
    assert paper_column("coralscapes_v2", "train") == "train"
    assert paper_column("coralscapes_v2", "val") == "val"
    assert paper_column("coralscapes_v2", "test") == "test"
    assert paper_column("mermaid", "train") == "train"
    assert paper_column("coralnet", "train") == "train"
    assert paper_column("coralnet", "val") == "val"
    assert paper_column("coralnet", "test") == "test"
    assert set(PAPER_COLUMNS) == {spec.key for spec in DATASET_SPECS}
    with pytest.raises(KeyError):
        paper_column("mermaid", "val")


def test_table_places_splits_and_sums_totals():
    moorea_source = empty_level()
    moorea_mapped = empty_level()
    apply_sparse(
        moorea_source,
        moorea_mapped,
        paper_column("moorea_labeled_corals", "train"),
        SparseCounts(
            images=10,
            mapped_images=8,
            source_annotations=100,
            source_pixels=100,
            mapped_annotations=80,
            mapped_pixels=80,
            source_labels={"a", "b"},
            mapped_labels={"a"},
        ),
    )
    benthos_source = empty_level()
    benthos_mapped = empty_level()
    add_split(benthos_source, "test", images=4, annotations=7, pixels=50, labels={"x"})
    add_split(benthos_mapped, "test", images=4, annotations=6, pixels=40, labels={"x"})

    table = build_table(
        {
            "moorea_labeled_corals": (moorea_source, moorea_mapped),
            "benthos_yuval": (benthos_source, benthos_mapped),
        }
    )

    assert len(table) == len(DATASET_SPECS) * 2 + 2
    moorea = table[
        (table["dataset_key"] == "moorea_labeled_corals") & (table["level"] == "source")
    ].iloc[0]
    assert moorea["images_train"] == 10
    assert moorea["images_val"] == 0
    assert moorea["images_test"] == 0
    assert moorea["annotations_train"] == 100
    assert moorea["pixels_train"] == 100
    assert moorea["n_classes"] == 2
    assert moorea["annotation_type"] == "Sparse"

    benthos = table[(table["dataset_key"] == "benthos_yuval") & (table["level"] == "source")].iloc[
        0
    ]
    assert benthos["images_train"] == 0
    assert benthos["images_test"] == 4
    assert benthos["annotations_test"] == 7
    assert benthos["pixels_test"] == 50
    assert benthos["annotation_type"] == "Dense"

    pacific = table[
        (table["dataset_key"] == "pacific_labeled_corals") & (table["level"] == "source")
    ].iloc[0]
    assert pacific["n_classes"] == 0
    assert pacific["images_test"] == 0

    source_total = table[(table["dataset"] == "Total") & (table["level"] == "source")].iloc[0]
    assert source_total["n_classes"] == 3
    assert source_total["images_train"] == 10
    assert source_total["images_test"] == 4
    assert source_total["annotations_train"] == 100
    assert source_total["annotations_test"] == 7
    assert source_total["pixels_test"] == 50

    mapped_total = table[(table["dataset"] == "Total") & (table["level"] == "mapped")].iloc[0]
    assert mapped_total["n_classes"] == 2
    assert mapped_total["annotations_train"] == 80
    assert mapped_total["pixels_train"] == 80
    assert mapped_total["annotations_test"] == 6
    assert mapped_total["pixels_test"] == 40
    assert mapped_total["images_train"] == 8
    assert mapped_total["images_test"] == 4


def test_apply_counts_omit_images_without_mapped_labels():
    source = empty_level()
    mapped = empty_level()
    apply_sparse(
        source,
        mapped,
        "train",
        SparseCounts(
            images=3,
            mapped_images=1,
            source_annotations=4,
            source_pixels=4,
            mapped_annotations=1,
            mapped_pixels=1,
            source_labels={"Acropora", "off"},
            mapped_labels={"Acropora"},
        ),
    )
    apply_dense(
        source,
        mapped,
        "test",
        images=2,
        counts=DenseCounts(
            source_components=3,
            source_pixels=9,
            mapped_components=1,
            mapped_pixels=2,
            mapped_images=1,
            source_labels={"Acropora", "junk"},
            mapped_labels={"Acropora"},
        ),
    )

    assert source["images_train"] == 3
    assert mapped["images_train"] == 1
    assert source["images_test"] == 2
    assert mapped["images_test"] == 1
    assert mapped["annotations_test"] == 1
    assert mapped["pixels_test"] == 2


def test_strip_transforms_keeps_filters_and_skips():
    sentinel = object()
    data = {
        "local_cache_dir": "/cache",
        "mermaid": {
            "train": {"annotation_date_cutoff": "2026-08-12", "transform": sentinel},
            "val": None,
        },
    }

    stripped = strip_transforms(data)

    assert stripped["local_cache_dir"] == "/cache"
    assert stripped["mermaid"]["val"] is None
    assert stripped["mermaid"]["train"] == {"annotation_date_cutoff": "2026-08-12"}
    assert data["mermaid"]["train"]["transform"] is sentinel


def test_script_entrypoint_spawn_pool():
    result = subprocess.run(
        [sys.executable, "scripts/dataset_table_stats.py", "--check-pool"],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "pool ok" in result.stdout
