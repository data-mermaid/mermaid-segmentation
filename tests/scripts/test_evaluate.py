"""Evaluation entrypoint: one loading path, through an exported bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import scripts.evaluate as evaluate_script
from mermaidseg.export import export_bundle
from mermaidseg.inference.bundle import load_bundle
from tests.mermaidseg.inference import _tiny

DELETED_FLAGS = [
    "--checkpoint",
    "--model-config",
    "--id2label",
    "--concept-id2name",
    "--training-config",
    "--resolution",
]


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    checkpoint, _ = _tiny.write_run(tmp_path_factory.mktemp("run"))
    return export_bundle(checkpoint, tmp_path_factory.mktemp("out") / "bundle")


def test_bundle_is_required():
    with pytest.raises(SystemExit):
        evaluate_script.build_parser().parse_args(["--output-dir", "out"])


@pytest.mark.parametrize("flag", DELETED_FLAGS)
def test_a_deleted_flag_is_rejected(flag: str):
    argv = ["--bundle", "b", "--output-dir", "out", flag, "x"]
    with pytest.raises(SystemExit):
        evaluate_script.build_parser().parse_args(argv)


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


def test_predictor_takes_size_and_normalisation_from_the_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bundle_dir: Path
):
    bundle = load_bundle(bundle_dir, "cpu")
    predictor = _run_main_capturing_predictor(monkeypatch, tmp_path, bundle_dir)
    assert predictor.input_size == bundle.input_size
    assert predictor.mean == bundle.mean
    assert predictor.std == bundle.std
    assert predictor.amp_dtype is None


def test_amp_dtype_is_resolved_from_the_bundle_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bundle_dir: Path
):
    predictor = _run_main_capturing_predictor(monkeypatch, tmp_path, bundle_dir, "--amp")
    assert predictor.amp_dtype is torch.bfloat16


def test_concept_names_are_the_bundles_in_channel_order(bundle_dir: Path):
    bundle = load_bundle(bundle_dir, "cpu")
    assert evaluate_script.concept_names_of(bundle) == _tiny.CONCEPT_NAMES


def test_removed_concepts_are_empty_without_a_concept_removal_file():
    config = _tiny.resolved_config()
    assert evaluate_script.run_removed_concepts(config, "unused.csv") == []
