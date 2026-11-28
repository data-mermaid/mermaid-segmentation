"""Evaluation orchestration: the registry, its helpers, and ``run_evaluations``."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch
import yaml

from mermaidseg.data.config import DataSection, SplitSelection
from mermaidseg.evaluation import metrics, point_sources, run
from mermaidseg.evaluation.run import EvalOptions, run_evaluations
from mermaidseg.inference.bundle import MODEL_JSON, load_bundle
from mermaidseg.io import REPO_ROOT
from tests.mermaidseg.conftest import deployable_lineage, export_ckpt, repo_with_origin
from tests.mermaidseg.evaluation.conftest import EvalCache, write_eval_cache
from tests.mermaidseg.inference import _tiny

_LOGGER = "mermaidseg.evaluation.run"


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("deployable")
    repo, pushed = repo_with_origin(base)
    checkpoint, _ = _tiny.write_run(base / "run", checkpoint_lineage=deployable_lineage(pushed))
    return export_ckpt(checkpoint, tmp_path_factory.mktemp("out") / "bundle", repo_dir=repo)


@pytest.fixture
def cache(tmp_path: Path) -> EvalCache:
    return write_eval_cache(tmp_path / "data")


@pytest.fixture
def data_config(tmp_path: Path, cache: EvalCache) -> Path:
    return _write_config(tmp_path, cache.data)


def _write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "data_config.yaml"
    path.write_text(yaml.safe_dump({"data": data}))
    return path


def _options(
    bundle: Path, tmp_path: Path, cache: EvalCache, data_config: Path, **overrides: Any
) -> EvalOptions:
    """The parser's defaults for the tiny bundle and cache, with ``overrides`` applied."""
    options = EvalOptions(
        bundle=bundle,
        cache_dir=cache.cache_dir,
        data_config=data_config,
        output_dir=tmp_path / "out",
        evals=list(run.ALL_EVALS),
        split="val",
        images_per_source=None,
        sample_seed=0,
        batch_size=8,
        num_workers=0,
        device="cpu",
        amp=False,
        probe_points=REPO_ROOT / "configs" / "eval" / "coralscapes_v2_probe_points.json",
        coralscapes39_probe=REPO_ROOT / "configs" / "eval" / "coralscapes_v2_39_probe_k50.json",
        svm_c=1.0,
        val_loss_max_images=1000,
        max_images_per_unit=None,
        max_test_images=None,
    )
    return replace(options, **overrides)


# --- helpers ------------------------------------------------------------------------------------


def test_concept_names_are_the_bundles_in_channel_order(bundle_dir: Path):
    bundle = load_bundle(bundle_dir, "cpu")
    assert run.concept_names_of(bundle) == _tiny.CONCEPT_NAMES


def test_concept_names_follow_channel_id_order_not_insertion_order():
    bundle = SimpleNamespace(concept_id2name={10: "last", 2: "first", 3: "middle"})
    assert run.concept_names_of(bundle) == ["first", "middle", "last"]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("split", "groups"), [("train", ["s1", "s2"]), ("val", ["s3"]), ("test", ["s4"])]
)
def test_coralnet_groups_are_the_split_files_groups_of_the_split(cache, split, groups):
    assert run.coralnet_groups(cache.section, split) == groups


def test_coralnet_groups_need_a_split_file(cache):
    data = {**cache.data, "datasets": {**cache.data["datasets"]}}
    data["datasets"]["coralnet"] = {
        k: v
        for k, v in data["datasets"]["coralnet"].items()
        if k not in ("split_file", "train", "test")
    } | {"val": {"groups": ["s3"]}}
    section = DataSection.model_validate(data)
    with pytest.raises(ValueError, match="no split_file"):
        run.coralnet_groups(section, "val")


def test_no_training_overlap_when_the_coralnet_entry_has_no_train_block(cache):
    data = {**cache.data, "datasets": {**cache.data["datasets"]}}
    data["datasets"]["coralnet"] = {
        k: v for k, v in data["datasets"]["coralnet"].items() if k != "train"
    }
    section = DataSection.model_validate(data)
    assert run.trained_on_scored_groups(section, ["s1", "s2"]) == []


def _with_coralscapes_train(cache: EvalCache, train: dict[str, Any] | None) -> dict[str, Any]:
    """The cache's data config with the coralscapes_v2 train block set, or removed for None."""
    entry = {k: v for k, v in cache.data["datasets"]["coralscapes_v2"].items() if k != "train"}
    if train is not None:
        entry["train"] = train
    return {**cache.data, "datasets": {**cache.data["datasets"], "coralscapes_v2": entry}}


def test_coralscapes_trained_on_scored_when_the_train_block_selects_every_group(cache):
    section = DataSection.model_validate(_with_coralscapes_train(cache, {"groups": "all"}))
    assert run.coralscapes_trained_on_scored(section, "val") is True


def test_coralscapes_not_trained_on_scored_when_the_entry_has_no_train_block(cache):
    section = DataSection.model_validate(_with_coralscapes_train(cache, None))
    assert run.coralscapes_trained_on_scored(section, "val") is False


def test_train_split_allows_only_coralnet_and_loss():
    check = run.check_split_evals
    check("train", ["coralnet", "val_loss"])
    check("val", list(run.ALL_EVALS))
    for other in ("pacific", "coralscapes", "coralscapes39"):
        with pytest.raises(ValueError, match="--split train supports only"):
            check("train", ["coralnet", other])


def test_failed_sections_lists_only_the_sections_with_an_error():
    summary = {
        "bundle": "b",
        "coralnet_val": {"error": "boom"},
        "pacific": {"pooled": {}},
        "val_loss": {"error": "bang"},
    }

    assert run.failed_sections(summary) == ["coralnet_val", "val_loss"]


# --- run_evaluations: wiring of the eval stages, with the evaluators replaced by recorders ------

_POOLED_RESULT = {"pooled": {"class_accuracy": 0.5, "taxonomic": {}}}


class _Calls:
    """Keyword arguments each stubbed evaluator received, by evaluator name."""

    def __init__(self) -> None:
        self.by_name: dict[str, list[dict[str, Any]]] = {}

    def record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.by_name.setdefault(name, []).append(kwargs)

    def one(self, name: str) -> dict[str, Any]:
        assert len(self.by_name.get(name, [])) == 1, name
        return self.by_name[name][0]

    def never(self, name: str) -> bool:
        return name not in self.by_name


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """Replace every package evaluator; the readers are built over the tiny cache."""
    recorded = _Calls()

    def _point_dataset(name: str, reader: Any, **kwargs: Any) -> Any:
        key = "pacific" if name == point_sources.PACIFIC_SOURCE else "coralnet"
        recorded.record(key, {"name": name, "reader": reader, **kwargs})
        return _POOLED_RESULT

    def _probe(label_space: Any, probe_jsons: Any, version: Any, **kwargs: Any) -> Any:
        recorded.record(
            label_space.eval_name, {"probe_jsons": probe_jsons, "version": version, **kwargs}
        )
        return {}

    def _val_loss(**kwargs: Any) -> Any:
        recorded.record("val_loss", kwargs)
        return {"overall": {"loss": 1.0}}

    write_sample = point_sources.write_coralnet_sample

    def _write_sample(**kwargs: Any) -> Any:
        recorded.record("sample", kwargs)
        return write_sample(**kwargs)

    monkeypatch.setattr(run, "evaluate_point_dataset", _point_dataset)
    monkeypatch.setattr(run, "evaluate_coralscapes_probe", _probe)
    monkeypatch.setattr(run, "evaluate_val_loss", _val_loss)
    monkeypatch.setattr(run, "write_coralnet_sample", _write_sample)
    monkeypatch.setattr(run, "load_benthic_hierarchy", lambda *_: {"hier": 1})
    monkeypatch.setattr(run, "run_removed_concepts", lambda *_: ["removed__a"])
    # The tiny bundle has no reported binary concept channels; the eval stages are under test.
    monkeypatch.setattr(metrics, "REPORTED_BINARY_CONCEPTS", ())
    return recorded


def _run(
    bundle_dir: Path, tmp_path: Path, cache: EvalCache, data_config: Path, **overrides: Any
) -> dict[str, Any]:
    """Run ``run_evaluations`` on the tiny bundle and return the parsed ``summary.json``."""
    run_evaluations(_options(bundle_dir, tmp_path, cache, data_config, **overrides))
    return json.loads((tmp_path / "out" / "summary.json").read_text())


def _groups(reader: Any) -> list[str]:
    return sorted(reader.images["group"].unique())


@pytest.mark.parametrize(("amp", "expected"), [(False, None), (True, torch.bfloat16)])
def test_amp_dtype_is_resolved_from_the_bundle_config(
    calls, bundle_dir, tmp_path, cache, data_config, amp, expected
):
    _run(bundle_dir, tmp_path, cache, data_config, evals=["coralscapes39"], amp=amp)

    assert calls.one("coralscapes39")["predictor"].amp_dtype is expected


def test_the_bundles_concept_mapping_path_is_resolved_against_the_repo_root(
    calls, tmp_path, cache, data_config
):
    mapping = "configs/not_the_default_mapping.csv"
    config = _tiny.resolved_config()
    config["training"] = {**config["training"], "concept_mapping_path": mapping}
    repo, pushed = repo_with_origin(tmp_path)
    checkpoint, _ = _tiny.write_run(
        tmp_path / "run",
        config=config,
        checkpoint_lineage=deployable_lineage(pushed, resolved_config=config),
    )
    bundle = export_ckpt(checkpoint, tmp_path / "bundle", repo_dir=repo)

    _run(bundle, tmp_path, cache, data_config, evals=["coralscapes"])

    assert calls.one("coralscapes")["concepts"].taxonomy_csv == REPO_ROOT / mapping


def test_each_requested_eval_runs_once_even_when_listed_twice(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _run(
        bundle_dir,
        tmp_path,
        cache,
        data_config,
        evals=["coralscapes39", "pacific", "coralscapes39"],
    )

    assert summary["evals"] == ["coralscapes39", "pacific"]
    calls.one("coralscapes39")
    calls.one("pacific")
    assert calls.never("coralnet")


def test_the_hierarchy_is_skipped_unless_coralnet_or_pacific_run(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config
):
    monkeypatch.setattr(
        run,
        "load_benthic_hierarchy",
        lambda *_: pytest.fail("hierarchy is only for coralnet and pacific"),
    )

    _run(bundle_dir, tmp_path, cache, data_config, evals=["coralscapes39", "val_loss"])


@pytest.mark.parametrize(
    ("split", "groups"), [("val", ["s3"]), ("test", ["s4"]), ("train", ["s1", "s2"])]
)
def test_coralnet_is_scored_on_exactly_the_split_files_groups_of_the_split(
    calls, bundle_dir, tmp_path, cache, data_config, split, groups
):
    summary = _run(bundle_dir, tmp_path, cache, data_config, evals=["coralnet"], split=split)

    kwargs = calls.one("coralnet")
    assert _groups(kwargs["reader"]) == groups
    assert kwargs["name"] == f"coralnet_{split}"
    assert kwargs["removed_concepts"] == ["removed__a"]
    assert kwargs["hierarchy"] == {"hier": 1}
    assert kwargs["concept_names"] == _tiny.CONCEPT_NAMES
    assert summary[f"coralnet_{split}"] == _POOLED_RESULT
    assert summary["num_removed_concepts"] == 1


def test_a_trained_on_coralnet_split_is_warned_about_and_recorded(
    calls, bundle_dir, tmp_path, cache, data_config, caplog
):
    with caplog.at_level("WARNING", logger=_LOGGER):
        summary = _run(bundle_dir, tmp_path, cache, data_config, evals=["coralnet"], split="val")

    assert summary["coralnet_split_in_training"] is True
    assert any("train" in r.getMessage() and "s3" in r.getMessage() for r in caplog.records)


def test_no_coralnet_overlap_is_recorded_for_a_split_the_config_does_not_train_on(
    calls, bundle_dir, tmp_path, cache, data_config, caplog
):
    with caplog.at_level("WARNING", logger=_LOGGER):
        summary = _run(bundle_dir, tmp_path, cache, data_config, evals=["coralnet"], split="test")

    assert summary["coralnet_split_in_training"] is False
    assert not [
        r for r in caplog.records if r.levelname == "WARNING" and "trains on" in r.getMessage()
    ]


def test_pacific_is_scored_on_the_val_block_with_its_annotators(
    calls, bundle_dir, tmp_path, cache, data_config
):
    _run(bundle_dir, tmp_path, cache, data_config, evals=["pacific"])

    reader = calls.one("pacific")["reader"]
    assert _groups(reader) == ["heron_reef", "line_islands"]
    assert sorted(reader.points["annotator"]) == ["archived", "host"]


def test_the_coralscapes_probes_are_scored_on_the_pinned_version(
    calls, bundle_dir, tmp_path, cache, data_config
):
    _run(bundle_dir, tmp_path, cache, data_config, evals=["coralscapes", "coralscapes39"])

    pin = cache.data["datasets"]["coralscapes_v2"]["version"]
    for name in ("coralscapes", "coralscapes39"):
        manifest = calls.one(name)["version"].manifest
        assert (manifest.dataset, manifest.version) == ("coralscapes_v2", pin)


def test_images_per_source_hands_the_seeded_sample_to_the_coralnet_reader(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _run(
        bundle_dir,
        tmp_path,
        cache,
        data_config,
        evals=["coralnet"],
        split="train",
        images_per_source=2,
        sample_seed=7,
    )

    sample = calls.one("sample")
    expected_path = tmp_path / "out" / "coralnet_train_sample.parquet"
    assert (sample["path"], sample["images_per_source"], sample["seed"]) == (expected_path, 2, 7)
    assert _groups(sample["reader"]) == ["s1", "s2"]
    sampled = pd.read_parquet(expected_path)["image_id"].tolist()
    assert len(sampled) == 3 and "c2a" in sampled
    reader = calls.one("coralnet")["reader"]
    assert sorted(reader.images["image_id"]) == sorted(sampled)
    assert summary["images_per_source"] == 2
    assert summary["sample_seed"] == 7
    assert summary["sample"] == str(expected_path)


def test_no_sample_is_drawn_without_images_per_source(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _run(bundle_dir, tmp_path, cache, data_config, evals=["coralnet"])

    assert calls.never("sample")
    assert "sample" not in summary


def test_no_sample_is_drawn_when_no_eval_uses_it(calls, bundle_dir, tmp_path, cache, data_config):
    _run(bundle_dir, tmp_path, cache, data_config, evals=["pacific"], images_per_source=5)

    assert calls.never("sample")


def test_val_loss_on_val_scores_the_data_section_with_the_max_images_option(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _run(
        bundle_dir, tmp_path, cache, data_config, evals=["val_loss"], val_loss_max_images=17
    )

    kwargs = calls.one("val_loss")
    assert kwargs["section"] == cache.section
    assert kwargs["split"] == "val"
    assert kwargs["cache_dir"] == cache.cache_dir
    assert kwargs["image_ids"] is None
    assert kwargs["name"] == "val_loss"
    assert kwargs["apply_source_mask"] is False
    assert kwargs["max_images_per_dataset"] == 17
    assert summary["data_config"] == str(data_config)
    assert "val_loss" in summary
    assert "train_loss" not in summary


def test_train_loss_scores_exactly_the_split_files_train_groups_under_the_source_mask(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _run(bundle_dir, tmp_path, cache, data_config, evals=["val_loss"], split="train")

    kwargs = calls.one("val_loss")
    assert list(kwargs["section"].datasets) == ["coralnet"]
    assert kwargs["section"].datasets["coralnet"].train == SplitSelection(groups=["s1", "s2"])
    assert kwargs["split"] == "train"
    assert kwargs["name"] == "train_loss"
    assert kwargs["apply_source_mask"] is True
    assert kwargs["max_images_per_dataset"] == 1000
    assert kwargs["image_ids"] is None
    assert "train_loss" in summary


def test_test_loss_scores_the_test_blocks_and_is_recorded_under_test_loss(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _run(bundle_dir, tmp_path, cache, data_config, evals=["val_loss"], split="test")

    kwargs = calls.one("val_loss")
    assert kwargs["split"] == "test"
    assert kwargs["name"] == "test_loss"
    assert kwargs["apply_source_mask"] is False
    assert "test_loss" in summary
    assert "val_loss" not in summary


def test_train_loss_on_a_sample_is_uncapped_restricted_to_it_and_writes_no_config(
    calls, bundle_dir, tmp_path, cache, data_config
):
    _run(
        bundle_dir,
        tmp_path,
        cache,
        data_config,
        evals=["val_loss"],
        split="train",
        images_per_source=5,
    )

    kwargs = calls.one("val_loss")
    sampled = pd.read_parquet(tmp_path / "out" / "coralnet_train_sample.parquet")
    assert list(kwargs["section"].datasets) == ["coralnet"]
    assert kwargs["split"] == "train"
    assert kwargs["image_ids"] == {("coralnet", "val"): sampled["image_id"].tolist()}
    assert kwargs["max_images_per_dataset"] is None
    assert not list((tmp_path / "out").glob("*.yaml"))


def test_a_failed_eval_is_recorded_and_the_remaining_ones_run(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config
):
    def _boom_on_coralnet(name, *args, **kwargs):
        if name.startswith("coralnet"):
            raise RuntimeError("boom")
        return _POOLED_RESULT

    monkeypatch.setattr(run, "evaluate_point_dataset", _boom_on_coralnet)

    summary = _run(bundle_dir, tmp_path, cache, data_config, evals=["coralnet", "pacific"])

    assert summary["coralnet_val"] == {"error": "RuntimeError: boom"}
    assert summary["pacific"] == _POOLED_RESULT
    markdown = (tmp_path / "out" / "summary.md").read_text()
    assert "## coralnet_val: ERROR - RuntimeError: boom" in markdown


def test_a_concept_name_count_that_differs_from_the_models_channels_is_rejected(
    calls, bundle_dir, tmp_path, cache, data_config
):
    bundle = tmp_path / "bundle_extra_concept"
    shutil.copytree(bundle_dir, bundle)
    model_json = bundle / MODEL_JSON
    metadata = json.loads(model_json.read_text())
    metadata["concept_id2name"]["999"] = "phantom"
    model_json.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="concept names but the model has"):
        _run(bundle, tmp_path, cache, data_config, evals=["val_loss"])
    assert calls.never("val_loss")


@pytest.mark.parametrize("name", ["coralnet", "pacific", "coralscapes"])
def test_the_concept_evals_require_the_reported_binary_concepts(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config, name
):
    monkeypatch.setattr(metrics, "REPORTED_BINARY_CONCEPTS", ("bleached", "anthropogenic"))

    with pytest.raises(RuntimeError, match=r"\['anthropogenic'\]"):
        _run(bundle_dir, tmp_path, cache, data_config, evals=[name])
    assert calls.never(name)


@pytest.mark.parametrize("name", ["coralscapes39", "val_loss"])
def test_the_other_evals_do_not_require_the_reported_binary_concepts(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config, name
):
    monkeypatch.setattr(metrics, "REPORTED_BINARY_CONCEPTS", ("bleached", "anthropogenic"))

    _run(bundle_dir, tmp_path, cache, data_config, evals=[name])

    calls.one(name)


def test_a_trained_on_coralscapes_split_is_warned_about_and_recorded(
    calls, bundle_dir, tmp_path, cache, caplog
):
    config = _write_config(
        tmp_path, _with_coralscapes_train(cache, {"groups": ["train", "validation"]})
    )
    with caplog.at_level("WARNING", logger=_LOGGER):
        summary = _run(bundle_dir, tmp_path, cache, config, evals=["coralscapes"], split="val")

    assert summary["coralscapes_split_in_training"] is True
    assert any(
        "trains on" in r.getMessage() and "validation" in r.getMessage() for r in caplog.records
    )


def test_no_coralscapes_overlap_is_recorded_for_a_split_the_config_does_not_train_on(
    calls, bundle_dir, tmp_path, cache, caplog
):
    config = _write_config(
        tmp_path, _with_coralscapes_train(cache, {"groups": ["train", "validation"]})
    )
    with caplog.at_level("WARNING", logger=_LOGGER):
        summary = _run(bundle_dir, tmp_path, cache, config, evals=["coralscapes39"], split="test")

    assert summary["coralscapes_split_in_training"] is False
    assert not [
        r for r in caplog.records if r.levelname == "WARNING" and "trains on" in r.getMessage()
    ]


def test_a_coralscapes_error_is_recorded_when_the_config_has_no_coralscapes_entry(
    calls, bundle_dir, tmp_path, cache
):
    datasets = {k: v for k, v in cache.data["datasets"].items() if k != "coralscapes_v2"}
    config = _write_config(tmp_path, {**cache.data, "datasets": datasets})

    summary = _run(
        bundle_dir, tmp_path, cache, config, evals=["coralnet", "coralscapes"], split="val"
    )

    assert "error" not in summary["coralnet_val"]
    assert "error" in summary["coralscapes_val"]
    assert "coralscapes_split_in_training" not in summary
