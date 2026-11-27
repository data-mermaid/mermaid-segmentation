"""Pacific evaluation: the reader of the val block and what evaluate_pacific asks of the runner."""

from __future__ import annotations

from mermaidseg.evaluation import pacific_eval
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
    assert {
        k: calls[k]
        for k in ("predictor", "id2label", "concept_names", "output_dir", "taxonomy_csv")
    } == {
        "predictor": "predictor",
        "id2label": {0: "ignore", 1: "porites", 2: "sand"},
        "concept_names": [],
        "output_dir": "out",
        "taxonomy_csv": "configs/class_to_concepts.csv",
    }
    assert calls["eval_name"] == "pacific_labeled_corals"
    assert calls["unit_col"] == "group"
    assert (calls["batch_size"], calls["num_workers"], calls["max_images_per_unit"]) == (3, 0, 2)
    assert calls["removed_concepts"] == ["genus__x"]
