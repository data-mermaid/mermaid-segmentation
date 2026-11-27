"""CoralNet point evaluation: split-file groups, the reader and per-source image sampling."""

from __future__ import annotations

import pandas as pd
import pytest

from mermaidseg.evaluation import coralnet_eval
from mermaidseg.evaluation.coralnet_eval import (
    coralnet_reader,
    sample_images_per_source,
    split_groups,
    write_coralnet_sample,
)
from tests.mermaidseg.evaluation.conftest import write_eval_cache


@pytest.mark.parametrize(
    ("split", "groups"),
    [("train", ["s1", "s2"])],
)
def test_split_groups_are_the_split_files_groups_of_that_split(tmp_path, split, groups):
    cache = write_eval_cache(tmp_path)
    assert split_groups(cache.split_file, split) == groups


def test_a_split_with_no_group_raises(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("group,split\ns1,train\n")
    with pytest.raises(ValueError, match="no group in split 'test'"):
        split_groups(path, "test")


def test_sample_images_per_source_is_seeded_and_capped():
    df = pd.DataFrame(
        {
            "group": ["s1"] * 10 + ["s2"] * 3,
            "image_id": [f"a{i}" for i in range(10)] + ["b0", "b1", "b2"],
        }
    )
    first = sample_images_per_source(df, 4, seed=0)
    assert first["group"].value_counts().to_dict() == {"s1": 4, "s2": 3}
    assert set(first.loc[first["group"] == "s2", "image_id"]) == {"b0", "b1", "b2"}
    assert first.equals(sample_images_per_source(df, 4, seed=0))
    draws = {tuple(sample_images_per_source(df, 4, seed=s)["image_id"]) for s in range(5)}
    assert len(draws) > 1


def test_write_coralnet_sample_writes_and_returns_the_seeded_per_source_sample(tmp_path):
    cache = write_eval_cache(tmp_path)
    reader = coralnet_reader(cache.section, cache.cache_dir, groups=["s1", "s2"], split="train")
    path = tmp_path / "nested" / "keep.parquet"

    returned = write_coralnet_sample(path=path, reader=reader, images_per_source=2, seed=7)

    written = pd.read_parquet(path)
    assert list(written.columns) == ["dataset", "source_id", "image_id"]
    assert set(written["dataset"]) == {"coralnet"}
    assert written["source_id"].tolist() == ["1", "1", "2"]
    assert written["image_id"].tolist()[2] == "c2a"
    assert set(written["image_id"].tolist()[:2]) < {"c1a", "c1b", "c1c"}
    assert returned == written["image_id"].tolist()


def test_evaluate_coralnet_maps_labels_through_the_readers_targets(tmp_path, monkeypatch):
    cache = write_eval_cache(tmp_path)
    reader = coralnet_reader(cache.section, cache.cache_dir, groups=["s3"], split="val")
    seen: dict = {}

    def run(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(coralnet_eval, "run_point_evaluation", run)

    result = coralnet_eval.evaluate_coralnet(
        predictor=None,  # type: ignore[arg-type]
        reader=reader,
        id2label={0: "ignore", 1: "sand"},
        concept_names=[],
        taxonomy_csv="configs/class_to_concepts.csv",
        hierarchy=None,
        output_dir=tmp_path / "out",
    )

    assert seen["class_lookup"].name_to_class_id == {"acro": 0, "sand": 1}
    assert seen["unit_col"] == "group"
    assert result["groups"] == ["s3"]
