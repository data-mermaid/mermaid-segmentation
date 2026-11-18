"""CoralNet point evaluation: dedup fields and per-source image sampling."""

from __future__ import annotations

import pytest


def test_evaluate_coralnet_forwards_dedup_fields(monkeypatch):
    from mermaidseg.evaluation import coralnet_eval

    seen = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            import pandas as pd

            self.df_images = pd.DataFrame({"image_id": ["1"]})
            self.df_annotations = pd.DataFrame({"source_label_name": ["sand"]})

    monkeypatch.setattr(coralnet_eval, "CoralNetDataset", FakeDataset)
    monkeypatch.setattr(coralnet_eval, "build_class_lookup", lambda *a, **k: None)
    monkeypatch.setattr(coralnet_eval, "build_concept_lookup", lambda *a, **k: None)
    monkeypatch.setattr(coralnet_eval, "run_point_evaluation", lambda **k: {"base": k["base_ds"]})

    kwargs = {
        "predictor": None,
        "whitelist_sources": [4, 5],
        "image_blacklist": "/bl.parquet",
        "annotation_overrides": "/ov.parquet",
        "id2label": {},
        "concept_names": [],
        "taxonomy_csv": "x",
        "hierarchy": None,
        "output_dir": "out",
    }
    result = coralnet_eval.evaluate_coralnet(**kwargs)
    assert seen["whitelist_sources"] == [4, 5]
    assert seen["image_blacklist"] == "/bl.parquet"
    assert seen["annotation_overrides"] == "/ov.parquet"
    assert seen["image_keep"] is None
    assert result["whitelist_sources"] == [4, 5]

    result = coralnet_eval.evaluate_coralnet(
        **{**kwargs, "whitelist_sources": [1, 2]}, image_keep="/keep.parquet", split="train"
    )
    assert seen["whitelist_sources"] == [1, 2]
    assert seen["split"] == "train"
    assert seen["image_keep"] == "/keep.parquet"
    assert result["image_keep"] == "/keep.parquet"


def test_sample_images_per_source_is_seeded_and_capped():
    import pandas as pd

    from mermaidseg.evaluation.coralnet_eval import sample_images_per_source

    df = pd.DataFrame(
        {
            "source_id": [1] * 10 + [2] * 3,
            "image_id": [f"a{i}" for i in range(10)] + ["b0", "b1", "b2"],
        }
    )
    first = sample_images_per_source(df, 4, seed=0)
    assert first["source_id"].value_counts().to_dict() == {1: 4, 2: 3}
    assert set(first.loc[first["source_id"] == 2, "image_id"]) == {"b0", "b1", "b2"}
    assert first.equals(sample_images_per_source(df, 4, seed=0))
    draws = {tuple(sample_images_per_source(df, 4, seed=s)["image_id"]) for s in range(5)}
    assert len(draws) > 1
    with pytest.raises(ValueError, match="must be positive"):
        sample_images_per_source(df, 0)
