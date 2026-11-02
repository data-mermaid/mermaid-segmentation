"""``demo.inference.load_artifacts``: a bundle carries its own label maps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_inference():
    spec = importlib.util.spec_from_file_location(
        "demo_inference_bundle_test", _REPO_ROOT / "demo" / "inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kwarg", ["id2label", "concept_id2name"])
def test_bundle_with_explicit_label_maps_is_an_error(demo_inference, tmp_path, kwarg):
    (tmp_path / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="carries its own"):
        demo_inference.load_artifacts(tmp_path, **{kwarg: "x.json"})
