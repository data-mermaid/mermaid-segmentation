"""Tests for mermaidseg.dataset_reconciliation.dataset_stats."""

from __future__ import annotations

import pytest

from mermaidseg.dataset_reconciliation.dataset_stats import (
    compute_class_by_source,
    compute_class_counts,
    compute_source_stats,
    compute_train_summary,
    resolve_split_annotations,
)
from tests._dataset_stubs import make_mermaid_stub, make_registry_stub


def _basic_registry():
    return make_registry_stub(
        target_id2label={1: "Acropora", 2: "Porites"},
        source_to_target_pairs=[(1, 1), (2, 2)],  # global source 1→target 1, 2→target 2
    )


class TestResolveSplitAnnotations:
    def test_plain_dataset_adds_target_id_column(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Porites"],
                "img-2": ["Acropora"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Indonesia"},
            source_id2name={1: "Acropora", 2: "Porites"},
            global_offset=0,
        )
        registry = _basic_registry()

        df_ann, df_img, source_name_to_target = resolve_split_annotations(stub, registry)
        assert list(df_ann["target_id"]) == [1, 2, 1]
        assert df_img["image_id"].tolist() == ["img-1", "img-2"]
        assert source_name_to_target == {"Acropora": 1, "Porites": 2}

    def test_subset_filters_by_image_ids(self):
        from torch.utils.data import Subset

        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Porites"],
                "img-3": ["Acropora"],
            },
            image_to_region={"img-1": "A", "img-2": "B", "img-3": "C"},
            source_id2name={1: "Acropora", 2: "Porites"},
            global_offset=0,
        )
        registry = _basic_registry()
        subset = Subset(stub, [0, 2])  # img-1 and img-3

        df_ann, df_img, mapping = resolve_split_annotations(subset, registry)
        assert sorted(df_ann["image_id"].unique().tolist()) == ["img-1", "img-3"]
        assert sorted(df_img["image_id"].tolist()) == ["img-1", "img-3"]
        assert list(df_ann["target_id"]) == [1, 1]

    def test_combined_concatenates_children_with_offsets(self):
        from tests._dataset_stubs import ConcatStub, make_coralnet_stub

        stub_a = make_mermaid_stub(
            image_to_classes={"a-1": ["Acropora"]},
            image_to_region={"a-1": "A"},
            source_id2name={1: "Acropora"},
            global_offset=0,
        )
        stub_b = make_coralnet_stub(
            image_to_classes={"b-1": ["123"]},
            image_to_source={"b-1": 99},
            source_id2name={1: "123"},
            global_offset=10,
        )
        # Mermaid local id 1 → global 1 → target 1 (Acropora)
        # CoralNet local id 1 + offset 10 = global 11 → target 2 (Porites)
        registry = make_registry_stub(
            target_id2label={1: "Acropora", 2: "Porites"},
            source_to_target_pairs=[(1, 1), (11, 2)],
            num_global_source=11,
        )
        wrapper = ConcatStub(_datasets=[stub_a, stub_b])

        df_ann, df_img, mapping = resolve_split_annotations(wrapper, registry)
        assert sorted(df_ann["image_id"].unique().tolist()) == ["a-1", "b-1"]
        assert sorted(df_ann["target_id"].tolist()) == [1, 2]
        # mapping is the union; per-child target_id is already encoded on each row.
        assert mapping == {"Acropora": 1, "123": 2}

    def test_unknown_shape_returns_none_and_warns(self, caplog):
        import logging

        registry = _basic_registry()
        with caplog.at_level(
            logging.WARNING, logger="mermaidseg.dataset_reconciliation.dataset_stats"
        ):
            result = resolve_split_annotations(object(), registry)
        assert result is None
        assert any("unsupported split shape" in r.message.lower() for r in caplog.records)


def _resolve(splits, registry):
    return {k: resolve_split_annotations(v, registry) for k, v in splits.items()}


class TestComputeClassCounts:
    def test_schema_and_values(self):
        from torch.utils.data import Subset

        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Acropora", "Porites"],
                "img-2": ["Acropora"],
                "img-3": ["Other"],
                "img-4": ["Porites"],
            },
            image_to_region={"img-1": "A", "img-2": "A", "img-3": "B", "img-4": "B"},
            source_id2name={1: "Acropora", 2: "Porites", 3: "Other"},
            global_offset=0,
        )
        registry = make_registry_stub(
            target_id2label={1: "Acropora", 2: "Porites", 3: "Other"},
            source_to_target_pairs=[(1, 1), (2, 2), (3, 3)],
        )
        resolved = _resolve(
            {
                "train": Subset(stub, [0, 1]),  # img-1 + img-2
                "val": Subset(stub, [2]),  # img-3
                "test": Subset(stub, [3]),  # img-4
            },
            registry,
        )

        counts = compute_class_counts(resolved, registry)

        assert list(counts.columns) == [
            "target_id",
            "target_name",
            "class_kind",
            "train_annotations",
            "val_annotations",
            "test_annotations",
            "train_images",
            "val_images",
            "test_images",
            "train_fraction",
            "val_fraction",
            "test_fraction",
        ]
        assert counts["target_id"].tolist() == [0, 1, 2, 3]
        assert counts["target_name"].tolist() == ["ignore", "Acropora", "Porites", "Other"]
        assert counts["class_kind"].tolist() == [
            "background",
            "target",
            "target",
            "unclassified",
        ]
        acropora = counts.set_index("target_name").loc["Acropora"]
        assert acropora["train_annotations"] == 3
        assert acropora["val_annotations"] == 0
        assert acropora["train_images"] == 2
        assert abs(counts["train_fraction"].sum() - 1.0) < 1e-9


class TestComputeSourceStats:
    def test_mermaid_region_rows(self):
        from torch.utils.data import Subset

        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Acropora", "Porites"],
                "img-2": ["Acropora", "Porites"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Caribbean"},
            source_id2name={1: "Acropora", 2: "Porites"},
            global_offset=0,
        )
        registry = make_registry_stub(
            target_id2label={1: "Acropora", 2: "Porites"},
            source_to_target_pairs=[(1, 1), (2, 2)],
        )
        resolved = _resolve(
            {"train": Subset(stub, [0]), "val": Subset(stub, [1])},
            registry,
        )
        result = compute_source_stats(resolved)
        assert set(result["source_type"]) == {"region"}
        idx = result.set_index("source_key")
        assert idx.loc["Indonesia", "train_images"] == 1
        assert idx.loc["Indonesia", "train_annotations"] == 3
        assert idx.loc["Indonesia", "val_images"] == 0
        assert idx.loc["Caribbean", "val_images"] == 1
        assert idx.loc["Caribbean", "val_annotations"] == 2
        assert "test_images" not in result.columns

    def test_coralnet_source_keys_are_strings(self):
        from torch.utils.data import Subset

        from tests._dataset_stubs import make_coralnet_stub

        stub = make_coralnet_stub(
            image_to_classes={"img-1": ["a"], "img-2": ["b"]},
            image_to_source={"img-1": 42, "img-2": 7},
            source_id2name={1: "a", 2: "b"},
            global_offset=0,
        )
        registry = make_registry_stub(
            target_id2label={1: "a", 2: "b"},
            source_to_target_pairs=[(1, 1), (2, 2)],
        )
        resolved = _resolve({"train": Subset(stub, [0, 1])}, registry)
        result = compute_source_stats(resolved)
        assert set(result["source_type"]) == {"source"}
        assert set(result["source_key"]) == {"42", "7"}
        assert result["source_key"].dtype == object


class TestComputeTrainSummary:
    def test_summary_metrics_and_density(self):
        import pandas as pd
        from torch.utils.data import Subset

        # Train annotation counts per class: A=16, B=8, C=4, D=2, E=1, F=1 (total 32), so
        # p = 1/2, 1/4, 1/8, 1/16, 1/32, 1/32 and entropy = 1.9375 bits.
        names = ["A", "B", "C", "D", "E", "F"]
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["A"] * 10 + ["B"] * 8 + ["C"] * 4,
                "img-2": ["A"] * 6 + ["D"] * 2 + ["E", "F"],
                "img-3": ["A"],
            },
            image_to_region={"img-1": "R", "img-2": "R", "img-3": "R"},
            source_id2name={i + 1: n for i, n in enumerate(names)},
            global_offset=0,
        )
        # make_mermaid_stub derives df_images from df_annotations, so add a
        # zero-annotation image by hand.
        stub.df_images = pd.concat(
            [
                stub.df_images,
                pd.DataFrame([{"image_id": "img-4", "region_id": "R", "region_name": "R"}]),
            ],
            ignore_index=True,
        )
        registry = make_registry_stub(
            target_id2label={i + 1: n for i, n in enumerate(names)},
            source_to_target_pairs=[(i + 1, i + 1) for i in range(len(names))],
        )
        resolved = _resolve(
            {
                "train": Subset(stub, [0, 1]),
                "val": Subset(stub, [2]),
                "test": Subset(stub, [3]),  # img-4 (no annotations)
            },
            registry,
        )
        summary = compute_train_summary(resolved, registry)

        assert summary["total_images"] == 4
        assert summary["total_annotations"] == 33
        assert summary["splits"]["train"] == {"images": 2, "annotations": 32}
        assert summary["splits"]["val"] == {"images": 1, "annotations": 1}
        assert summary["splits"]["test"] == {"images": 1, "annotations": 0}
        assert summary["num_target_classes"] == 7  # 6 targets + background
        assert summary["eligible_num_classes"] == 6

        assert summary["top1_share"] == pytest.approx(16 / 32)
        assert summary["top3_share"] == pytest.approx(28 / 32)
        assert summary["top5_share"] == pytest.approx(31 / 32)
        assert summary["effective_num_classes"] == pytest.approx(2**1.9375)

        assert summary["annotations_per_image"]["test"]["min"] == 0
        assert summary["annotations_per_image"]["train"]["max"] == 22
        assert summary["annotations_per_image"]["train"]["mean"] == 16.0


class TestComputeClassBySource:
    def test_long_format_omits_zero_rows(self):
        from torch.utils.data import Subset

        stub = make_mermaid_stub(
            image_to_classes={"img-1": ["Acropora"], "img-2": ["Porites"]},
            image_to_region={"img-1": "Indonesia", "img-2": "Caribbean"},
            source_id2name={1: "Acropora", 2: "Porites"},
            global_offset=0,
        )
        registry = make_registry_stub(
            target_id2label={1: "Acropora", 2: "Porites"},
            source_to_target_pairs=[(1, 1), (2, 2)],
        )
        resolved = _resolve(
            {"train": Subset(stub, [0]), "val": Subset(stub, [1])},
            registry,
        )
        result = compute_class_by_source(resolved, registry)

        assert list(result.columns) == [
            "source_key",
            "source_type",
            "target_id",
            "target_name",
            "split",
            "annotations",
            "images",
        ]
        assert len(result) == 2
        row = result.set_index(["source_key", "target_name", "split"]).loc[
            ("Indonesia", "Acropora", "train")
        ]
        assert row["annotations"] == 1
        assert row["target_id"] == 1
