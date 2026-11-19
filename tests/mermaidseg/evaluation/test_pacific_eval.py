"""evaluate_pacific: what it asks of the dataset, the lookups and the shared runner."""

from __future__ import annotations

import pandas as pd

from mermaidseg.evaluation import pacific_eval
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS


def test_evaluate_pacific_builds_the_val_dataset_lookups_and_runs_per_site(monkeypatch):
    calls: dict[str, dict] = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            calls["dataset"] = kwargs
            self.df_images = pd.DataFrame({"image_id": ["1"]})
            self.df_annotations = pd.DataFrame(
                {"source_label_name": ["Sand", "sand", "Acropora", "SAND"]}
            )

    def build_class_lookup(*args):
        calls["class_lookup"] = {"args": args}
        return "class-lookup"

    def build_concept_lookup(*args):
        calls["concept_lookup"] = {"args": args}
        return "concept-lookup"

    def run(**kwargs):
        calls["run"] = kwargs
        return {"ran": True}

    monkeypatch.setattr(pacific_eval, "PacificLabeledCoralsDataset", FakeDataset)
    monkeypatch.setattr(pacific_eval, "build_class_lookup", build_class_lookup)
    monkeypatch.setattr(pacific_eval, "build_concept_lookup", build_concept_lookup)
    monkeypatch.setattr(pacific_eval, "run_point_evaluation", run)
    hierarchy = {"acropora": None}

    result = pacific_eval.evaluate_pacific(
        predictor="predictor",  # type: ignore[arg-type]
        whitelist_subsets=["a"],
        whitelist_sites=["heron_reef"],
        id2label={0: "ignore"},
        concept_names=["c1"],
        taxonomy_csv="tax.csv",
        hierarchy=hierarchy,
        output_dir="out",
        batch_size=3,
        num_workers=0,
        max_images_per_unit=2,
        removed_concepts=["genus__x"],
    )

    assert result == {"ran": True}
    assert calls["dataset"] == {
        "whitelist_subsets": ["a"],
        "whitelist_sites": ["heron_reef"],
        "transform": None,
        "split": "val",
    }
    names = ["sand", "acropora"]
    assert calls["class_lookup"]["args"] == (
        "pacific_labeled_corals",
        names,
        {0: "ignore"},
        hierarchy,
    )
    assert calls["concept_lookup"]["args"] == ("pacific_labeled_corals", names, ["c1"], "tax.csv")
    run_kwargs = calls["run"]
    assert run_kwargs.pop("base_ds").__class__ is FakeDataset
    assert run_kwargs == {
        "eval_name": "pacific_labeled_corals",
        "unit_col": "site",
        "key_cols": ["site", "subset", "image_id"],
        "predictor": "predictor",
        "class_lookup": "class-lookup",
        "concept_lookup": "concept-lookup",
        "concept_names": ["c1"],
        "id2label": {0: "ignore"},
        "output_dir": "out",
        "binary_concepts": REPORTED_BINARY_CONCEPTS,
        "taxonomy_csv": "tax.csv",
        "batch_size": 3,
        "num_workers": 0,
        "max_images_per_unit": 2,
        "removed_concepts": ["genus__x"],
    }
