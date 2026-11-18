"""CoralNet eval split selection: ``--coralnet-splits`` and the dedup fields of the data config."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = {
    "coralnet": {
        "train": {"whitelist_sources": [1, 2]},
        "val": {"whitelist_sources": [3], "image_blacklist": "/bl.parquet"},
        "test": {
            "whitelist_sources": [4, 5],
            "image_blacklist": "/bl.parquet",
            "annotation_overrides": "/ov.parquet",
        },
    },
    "benthos_yuval": {"train": None, "val": {"whitelist_sites": ["RS24"]}},
}


@pytest.fixture(scope="module")
def evaluate_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_script", _REPO_ROOT / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_field(evaluate_script):
    f = evaluate_script._split_field
    assert f(DATA, "coralnet", "val", "whitelist_sources") == [3]
    assert f(DATA, "coralnet", "test", "whitelist_sources") == [4, 5]
    assert f(DATA, "coralnet", "test", "annotation_overrides") == "/ov.parquet"
    assert f(DATA, "coralnet", "val", "annotation_overrides") is None
    assert f(DATA, "benthos_yuval", "val", "whitelist_sites") == ["RS24"]
    assert f(DATA, "benthos_yuval", "train", "whitelist_sites") is None
    assert f(DATA, "missing", "val", "whitelist_sources") is None


def test_coralnet_split_without_sources_fails_early(evaluate_script):
    data = {"coralnet": {"val": {"whitelist_sources": [3]}}}
    with pytest.raises(ValueError, match=r"no coralnet\.test\.whitelist_sources"):
        evaluate_script._coralnet_sources(data, "test")


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


def test_train_split_reads_the_train_sources(evaluate_script):
    assert "train" in evaluate_script.SPLITS
    assert evaluate_script._coralnet_sources(DATA, "train") == [1, 2]


def test_train_split_allows_only_coralnet_and_loss(evaluate_script):
    check = evaluate_script.check_split_evals
    check("train", ["coralnet", "val_loss"])
    check("val", list(evaluate_script.ALL_EVALS))
    for other in ("pacific", "coralscapes", "coralscapes39"):
        with pytest.raises(ValueError, match="--split train supports only"):
            check("train", ["coralnet", other])


def test_train_loss_data_config_is_coralnet_train_without_a_transform(evaluate_script, tmp_path):
    data = {
        "local_cache_dir": "/cache",
        "default": {
            "train": {"transform": {"RandomResizedCrop": {"size": [256, 256]}}},
            "val": {},
        },
        "coralnet": {
            "train": {
                "whitelist_sources": [1, 2],
                "image_blacklist": "/bl.parquet",
                "transform": {"Resize": {"height": 8}},
            },
            "val": None,
        },
        "mermaid": {"train": {"annotation_date_cutoff": "2026-08-12"}},
    }
    keep = tmp_path / "keep.parquet"
    out = evaluate_script.coralnet_loss_data_config(data, "train", keep)
    assert out == {
        "data": {
            "local_cache_dir": "/cache",
            "coralnet": {
                "val": {
                    "whitelist_sources": [1, 2],
                    "image_blacklist": "/bl.parquet",
                    "image_keep": str(keep.resolve()),
                }
            },
        }
    }


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
