"""Pacific evaluation: the reader of the val block and what evaluate_pacific asks of the runner."""

from __future__ import annotations

from mermaidseg.evaluation import pacific_eval
from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from tests.mermaidseg.evaluation.conftest import write_eval_cache


def test_the_reader_keeps_the_first_listed_annotator_of_each_point(tmp_path):
    cache = write_eval_cache(tmp_path)

    reader = pacific_eval.pacific_reader(cache.section, cache.cache_dir, tmp_path)

    points = reader.points.sort_values("image_id")
    assert points[["image_id", "label_id", "annotator"]].to_numpy().tolist() == [
        ["p1", 1, "host"],
        ["p2", 2, "archived"],
    ]


def test_evaluate_pacific_builds_the_lookups_and_runs_per_site(tmp_path, monkeypatch):
    cache = write_eval_cache(tmp_path)
    reader = pacific_eval.pacific_reader(cache.section, cache.cache_dir, tmp_path)
    calls: dict = {}

    def run(**kwargs):
        calls.update(kwargs)
        return {"ran": True}

    monkeypatch.setattr(pacific_eval, "run_point_evaluation", run)

    result = pacific_eval.evaluate_pacific(
        predictor="predictor",  # type: ignore[arg-type]
        reader=reader,
        id2label={0: "ignore", 1: "porites", 2: "sand"},
        concept_names=[],
        taxonomy_csv="configs/class_to_concepts.csv",
        hierarchy=None,
        output_dir="out",
        batch_size=3,
        num_workers=0,
        max_images_per_unit=2,
        removed_concepts=["genus__x"],
    )

    assert result == {"ran": True}
    assert calls.pop("class_lookup").name_to_class_id == {"porites": 1, "sand": 2}
    assert sorted(calls.pop("concept_lookup").name_to_row) == ["porites", "sand"]
    assert calls.pop("reader") is reader
    assert calls == {
        "eval_name": "pacific_labeled_corals",
        "unit_col": "group",
        "predictor": "predictor",
        "concept_names": [],
        "id2label": {0: "ignore", 1: "porites", 2: "sand"},
        "output_dir": "out",
        "binary_concepts": REPORTED_BINARY_CONCEPTS,
        "taxonomy_csv": "configs/class_to_concepts.csv",
        "batch_size": 3,
        "num_workers": 0,
        "max_images_per_unit": 2,
        "removed_concepts": ["genus__x"],
    }
