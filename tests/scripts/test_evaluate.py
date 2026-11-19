"""Evaluation entrypoint: one loading path, through an exported bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import scripts.evaluate as evaluate_script
from mermaidseg.evaluation.coralscapes_probe import load_legacy_probes
from mermaidseg.export import export_bundle
from mermaidseg.inference.bundle import load_bundle
from mermaidseg.io import REPO_ROOT
from tests.mermaidseg.inference import _tiny

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
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    checkpoint, _ = _tiny.write_run(tmp_path_factory.mktemp("run"))
    return export_bundle(checkpoint, tmp_path_factory.mktemp("out") / "bundle")


def _run_main_capturing_predictor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bundle_dir: Path, *extra: str
) -> Any:
    from mermaidseg.evaluation import coralscapes_probe

    seen: dict[str, Any] = {}

    def _probe(*, predictor, **_):
        seen["predictor"] = predictor
        return {}

    monkeypatch.setattr(coralscapes_probe, "evaluate_coralscapes39_probe", _probe)
    evaluate_script.main(
        [
            "--bundle",
            str(bundle_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--evals",
            "coralscapes39",
            "--device",
            "cpu",
            *extra,
        ]
    )
    return seen["predictor"]


@pytest.mark.parametrize(("extra", "expected"), [([], None), (["--amp"], torch.bfloat16)])
def test_amp_dtype_is_resolved_from_the_bundle_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bundle_dir: Path, extra, expected
):
    predictor = _run_main_capturing_predictor(monkeypatch, tmp_path, bundle_dir, *extra)
    assert predictor.amp_dtype is expected


def test_concept_names_are_the_bundles_in_channel_order(bundle_dir: Path):
    bundle = load_bundle(bundle_dir, "cpu")
    assert evaluate_script.concept_names_of(bundle) == _tiny.CONCEPT_NAMES


def test_main_passes_the_bundles_concept_mapping_path_resolved_against_the_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from mermaidseg.evaluation import coralscapes_probe

    mapping = "configs/not_the_default_mapping.csv"
    config = _tiny.resolved_config()
    config["training"] = {**config["training"], "concept_mapping_path": mapping}
    lineage = {"git_sha": "abc123", "git_dirty": False, "resolved_config": config}
    checkpoint, _ = _tiny.write_run(tmp_path / "run", config=config, checkpoint_lineage=lineage)
    bundle = export_bundle(checkpoint, tmp_path / "bundle")
    seen: dict[str, Any] = {}

    def _probe(*, taxonomy_csv, **_):
        seen["taxonomy_csv"] = taxonomy_csv
        return {}

    monkeypatch.setattr(coralscapes_probe, "evaluate_coralscapes_probe", _probe)
    monkeypatch.setattr(evaluate_script, "run_removed_concepts", lambda *_: [])
    # The tiny bundle has no reported binary concept channels; the eval stage is under test.
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", [])
    evaluate_script.main(
        [
            "--bundle",
            str(bundle),
            "--output-dir",
            str(tmp_path / "out"),
            "--evals",
            "coralscapes",
            "--device",
            "cpu",
        ]
    )
    assert seen["taxonomy_csv"] == REPO_ROOT / mapping


def test_split_field():
    f = evaluate_script._split_field
    assert f(DATA, "coralnet", "val", "whitelist_sources") == [3]
    assert f(DATA, "coralnet", "test", "whitelist_sources") == [4, 5]
    assert f(DATA, "coralnet", "test", "annotation_overrides") == "/ov.parquet"
    assert f(DATA, "coralnet", "val", "annotation_overrides") is None
    assert f(DATA, "benthos_yuval", "val", "whitelist_sites") == ["RS24"]
    assert f(DATA, "benthos_yuval", "train", "whitelist_sites") is None
    assert f(DATA, "missing", "val", "whitelist_sources") is None


def test_coralnet_split_without_sources_fails_early():
    data = {"coralnet": {"val": {"whitelist_sources": [3]}}}
    with pytest.raises(ValueError, match=r"no coralnet\.test\.whitelist_sources"):
        evaluate_script._coralnet_sources(data, "test")


def test_coralnet_manifest_returns_the_configured_key():
    data = {"coralnet": {"val": {"manifest_path": "etl/runA/m.parquet"}}}
    assert evaluate_script._coralnet_manifest(data, "val") == "etl/runA/m.parquet"


@pytest.mark.parametrize(
    "data",
    [
        {"coralnet": {"val": {"whitelist_sources": [3]}}},
        {"coralnet": {"val": None}},
    ],
)
def test_coralnet_split_without_manifest_fails_early(data):
    with pytest.raises(ValueError, match=r"coralnet\.val\.manifest_path"):
        evaluate_script._coralnet_manifest(data, "val")


def test_train_split_reads_the_train_sources():
    assert "train" in evaluate_script.SPLITS
    assert evaluate_script._coralnet_sources(DATA, "train") == [1, 2]


def test_train_split_allows_only_coralnet_and_loss():
    check = evaluate_script.check_split_evals
    check("train", ["coralnet", "val_loss"])
    check("val", list(evaluate_script.ALL_EVALS))
    for other in ("pacific", "coralscapes", "coralscapes39"):
        with pytest.raises(ValueError, match="--split train supports only"):
            check("train", ["coralnet", other])


def test_train_loss_data_config_is_coralnet_train_without_a_transform(tmp_path):
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


def _parse_defaults() -> Any:
    return evaluate_script.build_parser().parse_args(["--bundle", "b", "--output-dir", "out"])


def test_default_probe_points_file_loads():
    path = Path(_parse_defaults().probe_points)

    probe = json.loads(path.read_text())

    assert probe["num_classes"] == 95
    assert probe["images"]


def test_default_coralscapes39_probe_file_loads():
    path = _parse_defaults().coralscapes39_probe

    probes = load_legacy_probes([path])

    assert 50 in probes
