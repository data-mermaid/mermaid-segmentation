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

    coralnet_eval.evaluate_coralnet(
        predictor=None,
        whitelist_sources=[4, 5],
        image_blacklist="/bl.parquet",
        annotation_overrides="/ov.parquet",
        id2label={},
        concept_names=[],
        taxonomy_csv="x",
        hierarchy=None,
        output_dir="out",
    )
    assert seen["whitelist_sources"] == [4, 5]
    assert seen["image_blacklist"] == "/bl.parquet"
    assert seen["annotation_overrides"] == "/ov.parquet"
