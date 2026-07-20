"""Tests for MERMAID train/val holdout."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from mermaidseg.datasets.mermaid.mermaid_dataset import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_HOLDOUT_ROLE,
    DEFAULT_HOLDOUT_SEED,
    RARE_IMAGE_THRESHOLD,
    MermaidDataset,
    build_image_holdout_features,
    compute_split_quality,
    select_composite_holdout_image_ids,
    select_stratified_holdout_image_ids,
)


def _ann_frame(
    n_images: int = 20,
    regions: list[str] | None = None,
    *,
    labels: list[str] | None = None,
    benthic_ids: list[str] | None = None,
    growth_forms: list[str] | None = None,
) -> pd.DataFrame:
    regions = regions or ["A", "B"]
    labels = labels or ["Acropora"]
    benthic_ids = benthic_ids or ["101"]
    growth_forms = growth_forms or ["Branching"]
    rows = []
    for i in range(n_images):
        rows.append(
            {
                "image_id": f"img-{i:03d}",
                "region_id": i % len(regions),
                "region_name": regions[i % len(regions)],
                "benthic_attribute_name": labels[i % len(labels)],
                "benthic_attribute_id": benthic_ids[i % len(benthic_ids)],
                "growth_form_name": growth_forms[i % len(growth_forms)],
                "row": 10,
                "col": 10,
            }
        )
    return pd.DataFrame(rows)


class TestBuildImageHoldoutFeatures:
    def test_modes_and_region(self):
        df = _ann_frame(2, regions=["Alpha"], labels=["Acropora", "Porites"])
        features = build_image_holdout_features(df)
        assert list(features.columns) == [
            "image_id",
            "region_name",
            "benthic_attribute_name",
            "benthic_attribute_id",
            "growth_form_name",
        ]
        assert set(features["region_name"]) == {"Alpha"}
        assert set(features["benthic_attribute_name"]) == {"Acropora", "Porites"}


class TestSelectStratifiedHoldoutImageIds:
    def test_deterministic(self):
        df = _ann_frame(100)
        a = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        b = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        assert a == b
        c = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=7)
        assert a != c

    def test_complement(self):
        df = _ann_frame(40)
        val_ids = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        all_ids = set(df["image_id"].astype(str))
        assert val_ids <= all_ids
        assert val_ids.isdisjoint(all_ids - val_ids)
        assert val_ids | (all_ids - val_ids) == all_ids

    def test_per_region_val_presence(self):
        df = _ann_frame(20, regions=["Alpha", "Beta"])
        val_ids = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        for region in df["region_name"].unique():
            region_ids = set(df.loc[df["region_name"] == region, "image_id"].astype(str))
            if len(region_ids) >= 2:
                assert region_ids & val_ids

    def test_rare_class_gets_val_coverage(self):
        rows = []
        for i in range(3):
            rows.append(
                {
                    "image_id": f"rare-{i}",
                    "region_name": "R1",
                    "benthic_attribute_name": "RareTaxon",
                    "benthic_attribute_id": "999",
                    "growth_form_name": "None",
                    "row": 0,
                    "col": 0,
                }
            )
            for j in range(24):
                rows.append(
                    {
                        "image_id": f"rare-{i}",
                        "region_name": "R1",
                        "benthic_attribute_name": "Common",
                        "benthic_attribute_id": "1",
                        "growth_form_name": "None",
                        "row": j // 5,
                        "col": j % 5,
                    }
                )
        for i in range(20):
            rows.append(
                {
                    "image_id": f"common-{i:02d}",
                    "region_name": "R1",
                    "benthic_attribute_name": "Common",
                    "benthic_attribute_id": "1",
                    "growth_form_name": "None",
                    "row": 0,
                    "col": 0,
                }
            )
        df = pd.DataFrame(rows)
        composite = select_composite_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        rare_aware = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        assert not any(image_id in composite for image_id in ("rare-0", "rare-1", "rare-2"))
        assert any(image_id in rare_aware for image_id in ("rare-0", "rare-1", "rare-2"))

    def test_singleton_rare_pair_stays_train(self):
        df = pd.DataFrame(
            [
                {
                    "image_id": "solo",
                    "region_name": "Only",
                    "benthic_attribute_name": "Rare",
                    "benthic_attribute_id": "1",
                    "growth_form_name": "Branching",
                }
            ]
        )
        val_ids = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        assert val_ids == set()

    def test_rare_growth_form_gets_val_coverage(self):
        rows = []
        for i in range(3):
            rows.append(
                {
                    "image_id": f"gf-{i}",
                    "region_name": "R1",
                    "benthic_attribute_name": "Acropora",
                    "benthic_attribute_id": "1",
                    "growth_form_name": "Columnar",
                    "row": i,
                    "col": i,
                }
            )
        for i in range(20):
            rows.append(
                {
                    "image_id": f"plain-{i:02d}",
                    "region_name": "R1",
                    "benthic_attribute_name": "Acropora",
                    "benthic_attribute_id": "1",
                    "growth_form_name": "None",
                    "row": 0,
                    "col": 0,
                }
            )
        df = pd.DataFrame(rows)
        val_ids = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        assert any(image_id in val_ids for image_id in ("gf-0", "gf-1", "gf-2"))

    def test_regional_quota_near_target(self):
        df = _ann_frame(100, regions=["Alpha"])
        val_ids = select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42)
        target = max(1, int(round(100 * 0.1)))
        assert abs(len(val_ids) - target) <= 2

    def test_rare_aware_beats_composite_on_coverage(self):
        rows = []
        for region in ("R1", "R2"):
            for i in range(3):
                rows.append(
                    {
                        "image_id": f"{region}-rare-{i}",
                        "region_name": region,
                        "benthic_attribute_name": f"Rare-{region}",
                        "benthic_attribute_id": f"r{i}",
                        "growth_form_name": "None",
                        "row": i,
                        "col": i,
                    }
                )
            for i in range(30):
                rows.append(
                    {
                        "image_id": f"{region}-common-{i:02d}",
                        "region_name": region,
                        "benthic_attribute_name": "Common",
                        "benthic_attribute_id": "1",
                        "growth_form_name": "None",
                        "row": 0,
                        "col": 0,
                    }
                )
        df = pd.DataFrame(rows)
        composite = compute_split_quality(
            df,
            select_composite_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42),
            holdout_fraction=0.1,
            rare_image_threshold=RARE_IMAGE_THRESHOLD,
        )
        rare_aware = compute_split_quality(
            df,
            select_stratified_holdout_image_ids(df, holdout_fraction=0.1, holdout_seed=42),
            holdout_fraction=0.1,
            rare_image_threshold=RARE_IMAGE_THRESHOLD,
        )
        assert rare_aware["rare_class_val_coverage"] >= composite["rare_class_val_coverage"]


class TestMermaidDatasetSplits:
    def _build(self, df: pd.DataFrame, **kwargs) -> MermaidDataset:
        with (
            patch.object(MermaidDataset, "load_annotations", autospec=True),
            patch(
                "mermaidseg.datasets.mermaid.mermaid_dataset.boto3.client",
                return_value=MagicMock(),
            ),
            patch.object(MermaidDataset, "read_image", return_value=MagicMock()),
        ):
            ds = object.__new__(MermaidDataset)
            ds.annotations_path = "s3://bucket/x.parquet"
            ds.source_bucket = "bucket"
            ds.s3 = MagicMock()
            ds.holdout_fraction = kwargs.get("holdout_fraction")
            ds.holdout_seed = kwargs.get("holdout_seed", 42)
            ds.holdout_role = kwargs.get("holdout_role")
            ds.split = ds.holdout_role
            filtered = ds._apply_image_holdout(df.copy())
            ds.df_annotations = filtered
            ds.df_images = ds._derive_df_images_from_annotations(filtered)
            return ds

    def test_holdout_train_val_are_complements(self):
        df = _ann_frame(40)
        train = self._build(df, holdout_fraction=0.1, holdout_seed=42, holdout_role="train")
        val = self._build(df, holdout_fraction=0.1, holdout_seed=42, holdout_role="val")
        train_ids = set(train.df_images["image_id"].astype(str))
        val_ids = set(val.df_images["image_id"].astype(str))
        assert train_ids.isdisjoint(val_ids)
        assert train_ids | val_ids == set(df["image_id"].astype(str))
        assert len(val_ids) >= 1

    def test_default_holdout_settings(self):
        import inspect

        sig = inspect.signature(MermaidDataset.__init__)
        assert sig.parameters["holdout_fraction"].default == DEFAULT_HOLDOUT_FRACTION
        assert sig.parameters["holdout_seed"].default == DEFAULT_HOLDOUT_SEED
        assert sig.parameters["holdout_role"].default == DEFAULT_HOLDOUT_ROLE

    def test_holdout_role_sets_split(self):
        df = _ann_frame(40)
        ds = self._build(df, holdout_fraction=0.1, holdout_seed=42, holdout_role="val")
        assert ds.split == "val"
