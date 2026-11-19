"""build_concept_source_mask: main feeds the CoralNet split's annotations to the package and writes the mask."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

import scripts.build_concept_source_mask as script
from mermaidseg.dataset_reconciliation.concept_source_mask import MASK_COLUMNS

_COUNTS = pd.DataFrame(
    {
        "source_id": ["a", "a", "b"],
        "concept": ["bleached", "anthropogenic", "bleached"],
        "n_true": [0, 2, 4],
        "n_false": [3, 0, 1],
        "n_not_given": [1, 1, 0],
    }
)
_ANNOTATIONS = pd.DataFrame({"source_id": ["a", "b"], "source_label_name": ["x", "y"]})


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace dataset building, schema and counting with recorders; keep ``only_false_pairs``."""
    seen: dict[str, Any] = {}
    dataset = SimpleNamespace(df_annotations=_ANNOTATIONS, SOURCE_NAME="coralnet")

    def _build_datasets(data: Any, **kwargs: Any) -> dict:
        seen["build"] = kwargs
        return {(kwargs["names"][0], kwargs["splits"][0]): dataset}

    def _from_csv(path: Path, *, sources: set[str]) -> str:
        seen["schema"] = {"path": Path(path), "sources": sources}
        return "SCHEMA"

    def _counts(df: pd.DataFrame, schema: str, *, dataset_source: str) -> pd.DataFrame:
        seen["counts"] = {"df": df, "schema": schema, "dataset_source": dataset_source}
        return _COUNTS

    monkeypatch.setattr(script, "build_datasets", _build_datasets)
    monkeypatch.setattr(script, "ConceptSchema", SimpleNamespace(from_csv=_from_csv))
    monkeypatch.setattr(script, "source_concept_counts", _counts)
    monkeypatch.setattr(
        script,
        "setup_local_cache",
        lambda data: seen.update(
            cache_dir=data.get("local_cache_dir"), has_cache_key="local_cache_dir" in data
        ),
    )
    monkeypatch.setattr(script, "_git_commit", lambda: "abc1234")
    return seen


def _write_schema_config(tmp_path: Path) -> Path:
    """A data config with two configured datasets and one whose splits are all skipped."""
    path = tmp_path / "schema_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "local_cache_write_through": False,
                    "mermaid": {"train": {"annotation_date_cutoff": "2026-08-12"}, "val": None},
                    "catlin_seaview": {"train": None, "val": {"whitelist_regions": ["ATL"]}},
                    "benthos_yuval": {"train": None, "val": None},
                }
            }
        )
    )
    return path


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["build_concept_source_mask.py", *argv])
    script.main()


def test_main_builds_the_requested_coralnet_split_without_reading_images(
    monkeypatch, seen, tmp_path
):
    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"), "--split", "val")

    assert seen["build"]["names"] == ["coralnet"]
    assert seen["build"]["splits"] == ["val"]


def test_main_counts_the_datasets_annotations_against_the_schema_of_the_configured_sources(
    monkeypatch, seen, tmp_path
):
    schema_config = _write_schema_config(tmp_path)
    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"), "--schema-config", str(schema_config))

    assert seen["counts"]["df"] is _ANNOTATIONS
    assert seen["counts"]["schema"] == "SCHEMA"
    assert seen["counts"]["dataset_source"] == "coralnet"
    assert seen["schema"]["sources"] == {"mermaid", "catlin_seaview"}


def test_main_writes_only_the_pairs_with_false_points_and_no_true_points(
    monkeypatch, seen, tmp_path
):
    out = tmp_path / "nested" / "mask.csv"

    _run(monkeypatch, "--out", str(out))

    written = pd.read_csv(out)
    assert list(written.columns) == MASK_COLUMNS
    assert written.to_dict("records") == [
        {"source_id": "a", "concept": "bleached", "n_true": 0, "n_false": 3, "n_not_given": 1}
    ]


def test_main_writes_every_count_to_counts_out_when_given(monkeypatch, seen, tmp_path):
    counts_out = tmp_path / "counts.csv"

    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"), "--counts-out", str(counts_out))

    pd.testing.assert_frame_equal(pd.read_csv(counts_out), _COUNTS)


def test_main_writes_no_counts_file_without_counts_out(monkeypatch, seen, tmp_path):
    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["mask.csv", "mask.meta.yaml"]


def test_main_records_the_inputs_and_the_masked_counts_in_the_meta_sidecar(
    monkeypatch, seen, tmp_path
):
    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"), "--split", "val")

    meta = yaml.safe_load((tmp_path / "mask.meta.yaml").read_text())
    assert meta["rule"] == "only_false"
    assert meta["dataset"] == "coralnet"
    assert meta["split"] == "val"
    assert meta["n_sources"] == 2
    assert meta["n_binary_concepts"] == 2
    assert meta["n_pairs"] == 4
    assert meta["n_masked"] == 1
    assert meta["n_points"] == len(_ANNOTATIONS)
    assert meta["git_commit"] == "abc1234"
    assert len(meta["data_config_sha1"]) == 40
    assert len(meta["concepts_csv_sha1"]) == 40


def test_main_reports_the_masked_pair_count(monkeypatch, seen, tmp_path, capsys):
    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"))

    assert "1 of 2 sources x 2 binary concepts = 4 pairs are only_false" in capsys.readouterr().out


def test_main_cache_dir_flag_reaches_the_local_cache_setup(monkeypatch, seen, tmp_path):
    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"), "--cache-dir", "/the/cache")

    assert seen["cache_dir"] == "/the/cache"


def test_main_leaves_the_cache_unconfigured_without_cache_dir(monkeypatch, seen, tmp_path):
    _run(monkeypatch, "--out", str(tmp_path / "mask.csv"))

    assert seen["has_cache_key"] is False
