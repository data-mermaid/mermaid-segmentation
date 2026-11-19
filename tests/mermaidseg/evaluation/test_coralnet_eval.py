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
        "whitelist_sources": [5, 4],
        "image_blacklist": "/bl.parquet",
        "annotation_overrides": "/ov.parquet",
        "id2label": {},
        "concept_names": [],
        "taxonomy_csv": "x",
        "hierarchy": None,
        "output_dir": "out",
    }
    result = coralnet_eval.evaluate_coralnet(**kwargs)
    assert seen["whitelist_sources"] == [5, 4]  # forwarded as given, reported sorted
    assert seen["image_blacklist"] == "/bl.parquet"
    assert seen["annotation_overrides"] == "/ov.parquet"
    assert seen["image_keep"] is None
    assert result["whitelist_sources"] == [4, 5]
    assert "image_keep" not in result

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


def test_write_coralnet_sample_writes_the_seeded_per_source_sample_as_strings(
    tmp_path, monkeypatch
):
    import pandas as pd

    from mermaidseg.evaluation import coralnet_eval

    seen = {}
    df_images = pd.DataFrame(
        {
            "source_id": [1] * 6 + [2],
            "image_id": [100 + i for i in range(6)] + [200],
        }
    )

    class FakeDataset:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.df_images = df_images

    monkeypatch.setattr(coralnet_eval, "CoralNetDataset", FakeDataset)
    path = tmp_path / "nested" / "keep.parquet"

    returned = coralnet_eval.write_coralnet_sample(
        path=path,
        whitelist_sources=[2, 1],
        image_blacklist="/bl.parquet",
        annotation_overrides="/ov.parquet",
        images_per_source=3,
        seed=7,
    )

    assert returned == path
    assert seen == {
        "whitelist_sources": [2, 1],
        "image_blacklist": "/bl.parquet",
        "annotation_overrides": "/ov.parquet",
        "transform": None,
    }
    written = pd.read_parquet(path)
    assert list(written.columns) == ["dataset", "source_id", "image_id"]
    assert set(written["dataset"]) == {"coralnet"}
    assert written["source_id"].tolist() == ["1", "1", "1", "2"]
    assert written["image_id"].tolist()[3] == "200"
    assert set(written["image_id"].tolist()[:3]) <= {str(100 + i) for i in range(6)}
    expected = coralnet_eval.sample_images_per_source(df_images, 3, seed=7)
    assert written["image_id"].tolist() == expected["image_id"].astype(str).tolist()
