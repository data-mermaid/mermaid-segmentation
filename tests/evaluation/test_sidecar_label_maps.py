"""``scripts/evaluate.py --sidecar``: label maps come from the run's model.json."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from mermaidseg.inference import ManifestError
from tests.inference._tiny import (
    CLASS_NAMES,
    concept_id2name,
    save_tiny_encoder,
    tiny_model_cfg,
    write_tiny_sidecar,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evaluate_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_script_sidecar", _REPO_ROOT / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory):
    root = tmp_path_factory.mktemp("sidecar")
    return write_tiny_sidecar(root / "run", tiny_model_cfg(save_tiny_encoder(root / "encoder")))


def _args(evaluate_script, *argv):
    return evaluate_script.build_parser().parse_args(
        ["--output-dir", "o", "--checkpoint", "c", *argv]
    )


def test_sidecar_supplies_both_label_maps(evaluate_script, sidecar):
    id2label, concepts = evaluate_script._sidecar_label_maps(
        _args(evaluate_script, "--sidecar", str(sidecar))
    )
    assert id2label == dict(enumerate(CLASS_NAMES))
    assert concepts == concept_id2name()


def test_no_sidecar_leaves_the_label_maps_to_the_caller(evaluate_script):
    assert evaluate_script._sidecar_label_maps(_args(evaluate_script)) is None


@pytest.mark.parametrize("flag", ["--id2label", "--concept-id2name"])
def test_sidecar_with_explicit_maps_is_an_error(evaluate_script, sidecar, flag):
    args = _args(evaluate_script, "--sidecar", str(sidecar), flag, "x.json")
    with pytest.raises(ValueError, match="--sidecar"):
        evaluate_script._sidecar_label_maps(args)


def test_bundle_with_explicit_label_maps_is_an_error(evaluate_script, tmp_path):
    (tmp_path / "model.json").write_text("{}")
    for flag in ("--id2label", "--concept-id2name", "--sidecar"):
        args = evaluate_script.build_parser().parse_args(
            ["--output-dir", "o", "--checkpoint", str(tmp_path), flag, "x.json"]
        )
        with pytest.raises(ValueError, match="carries its own"):
            evaluate_script._load_model(args, torch.device("cpu"))


def test_sidecar_of_another_run_is_refused_for_a_checkpoint(evaluate_script, sidecar, tmp_path):
    ckpt = tmp_path / "model_epoch1"
    torch.save(
        {
            "model_state_dict": {},
            "config": {"run_name": "some-other-run", "model": {}},
        },
        ckpt,
    )
    args = evaluate_script.build_parser().parse_args(
        [
            "--output-dir",
            "o",
            "--checkpoint",
            str(ckpt),
            "--sidecar",
            str(sidecar),
            "--model-config",
            "unused.yaml",
        ]
    )
    with pytest.raises(ManifestError, match="does not belong"):
        evaluate_script._load_model(args, torch.device("cpu"))
