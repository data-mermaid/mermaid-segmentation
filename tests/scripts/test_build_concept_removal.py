"""build_concept_removal: main joins CoralNet sources onto the label counts and writes the removed channels."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

import scripts.build_concept_removal as script
from mermaidseg.dataset_reconciliation.concept_removal import REMOVAL_COLUMNS, CoverageRule

_DATA_CONFIG = Path(script.__file__).resolve().parents[1] / "configs" / "data_config.yaml"

_COUNTS = pd.DataFrame(
    {
        "dataset": ["coralnet", "coralnet", "mermaid"],
        "image_id": [1, 2, 7],
        "label": ["a", "b", "c"],
        "n": [5, 6, 7],
    }
)


def _coverage() -> pd.DataFrame:
    """Two channels, one removed for too few images."""
    rows = {
        "genus__acropora": ("genus", 900, 120, 60, 5, 1, False, False, False),
        "anthropogenic": ("binary", 20, 4, 0, 1, 0, True, True, False),
    }
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=REMOVAL_COLUMNS[1:])
    frame.index.name = "concept"
    frame["removed"] = frame[["fails_images", "fails_annotations", "fails_sources"]].any(axis=1)
    return frame


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Replace the CoralNet lookup, schema and coverage computation with recorders."""
    seen: dict[str, Any] = {}
    counts_path = tmp_path / "counts.parquet"
    _COUNTS.to_parquet(counts_path)
    seen["counts_path"] = counts_path

    def _image_sources(data_config: Path, cache_dir: str | None) -> pd.Series:
        seen["sources_args"] = (Path(data_config), cache_dir)
        return pd.Series({"1": "s10", "2": "s20"}, name="source_id")

    def _from_csv(path: Path, *, sources: set[str]) -> str:
        seen["schema"] = {"path": Path(path), "sources": sources}
        return "SCHEMA"

    def _coverage_fn(counts: pd.DataFrame, schema: str, rule: CoverageRule) -> pd.DataFrame:
        seen["coverage"] = {"counts": counts, "schema": schema, "rule": rule}
        return _coverage()

    monkeypatch.setattr(script, "coralnet_image_sources", _image_sources)
    monkeypatch.setattr(script, "ConceptSchema", SimpleNamespace(from_csv=_from_csv))
    monkeypatch.setattr(script, "concept_coverage", _coverage_fn)
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


def _run(
    monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any], tmp_path: Path, *extra: str
) -> Path:
    out = tmp_path / "nested" / "removal.csv"
    argv = [
        "build_concept_removal.py",
        "--data-config",
        str(_DATA_CONFIG),
        "--label-counts",
        str(seen["counts_path"]),
        "--out",
        str(out),
        *extra,
    ]
    monkeypatch.setattr("sys.argv", argv)
    script.main()
    return out


def test_main_maps_coralnet_images_to_sources_and_keeps_other_datasets_as_their_own_source(
    monkeypatch, seen, tmp_path
):
    _run(monkeypatch, seen, tmp_path)

    counts = seen["coverage"]["counts"]
    assert counts["source"].tolist() == ["s10", "s20", "mermaid"]
    assert counts["image_id"].tolist() == ["1", "2", "7"]
    assert seen["coverage"]["schema"] == "SCHEMA"


def test_main_passes_the_data_config_and_cache_dir_to_the_coralnet_lookup(
    monkeypatch, seen, tmp_path
):
    _run(monkeypatch, seen, tmp_path, "--cache-dir", "/the/cache")

    data_config, cache_dir = seen["sources_args"]
    assert data_config == _DATA_CONFIG
    assert cache_dir == "/the/cache"


def test_main_builds_the_schema_from_the_datasets_of_the_schema_config(monkeypatch, seen, tmp_path):
    schema_config = _write_schema_config(tmp_path)
    _run(monkeypatch, seen, tmp_path, "--schema-config", str(schema_config))

    assert seen["schema"]["sources"] == {"mermaid", "catlin_seaview"}


def test_main_threshold_flags_set_the_coverage_rule(monkeypatch, seen, tmp_path):
    _run(monkeypatch, seen, tmp_path, "--min-images", "7", "--min-coralnet-sources", "2")

    rule = seen["coverage"]["rule"]
    assert rule == CoverageRule(min_images=7, min_coralnet_sources=2)


def test_main_uses_the_default_rule_without_threshold_flags(monkeypatch, seen, tmp_path):
    _run(monkeypatch, seen, tmp_path)

    assert seen["coverage"]["rule"] == CoverageRule()


def test_main_writes_only_the_removed_channels(monkeypatch, seen, tmp_path):
    out = _run(monkeypatch, seen, tmp_path)

    written = pd.read_csv(out)
    assert list(written.columns) == REMOVAL_COLUMNS
    assert written["concept"].tolist() == ["anthropogenic"]


def test_main_writes_the_full_coverage_table_when_coverage_out_is_given(
    monkeypatch, seen, tmp_path
):
    coverage_out = tmp_path / "coverage.csv"

    _run(monkeypatch, seen, tmp_path, "--coverage-out", str(coverage_out))

    assert pd.read_csv(coverage_out)["concept"].tolist() == ["genus__acropora", "anthropogenic"]


def test_main_records_the_rule_and_the_removal_counts_in_the_meta_sidecar(
    monkeypatch, seen, tmp_path
):
    out = _run(monkeypatch, seen, tmp_path, "--min-images", "7")

    meta = yaml.safe_load(out.with_name("removal.meta.yaml").read_text())
    assert meta["rule"]["min_images"] == 7
    assert meta["n_channels"] == 2
    assert meta["n_removed"] == 1
    assert meta["removed_by_kind"] == {"binary": 1, "genus": 0}
    assert meta["channels_by_kind"] == {"binary": 1, "genus": 1}
    assert meta["datasets_counted"] == ["coralnet", "mermaid"]
    assert meta["git_commit"] == "abc1234"


def test_main_reports_how_many_taxonomic_and_binary_channels_were_removed(
    monkeypatch, seen, tmp_path, capsys
):
    _run(monkeypatch, seen, tmp_path)

    assert "removed 0 of 1 taxonomic and 1 of 1 binary concept channels" in capsys.readouterr().out


def test_main_exits_when_a_coralnet_image_has_no_source_in_the_data_config(
    monkeypatch, seen, tmp_path
):
    pd.concat([_COUNTS, _COUNTS.assign(image_id=[99, 98, 97])]).to_parquet(seen["counts_path"])

    with pytest.raises(SystemExit, match="2 CoralNet count rows have no source"):
        _run(monkeypatch, seen, tmp_path)
    assert "coverage" not in seen
