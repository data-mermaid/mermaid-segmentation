"""``scripts/evaluate.py --sidecar``: label maps come from the run's model.json."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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
    return evaluate_script.build_parser().parse_args(["--output-dir", "o", "--checkpoint", "c", *argv])


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
