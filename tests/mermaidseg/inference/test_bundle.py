"""Loading a bundle offline: the sidecar contract, the weights, and refusals of foreign bundles."""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from mermaidseg.export import export_bundle
from mermaidseg.inference.bundle import MODEL_JSON, WEIGHTS, load_bundle
from mermaidseg.inference.predictor import CBMPredictor
from tests.mermaidseg.inference import _tiny


def test_round_trip_reproduces_the_trained_model_outputs(run, bundle_dir):
    trained = run[1]
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = trained(x)
        actual = load_bundle(bundle_dir, "cpu").model(x)
    torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=0)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=1e-5, rtol=0)


def test_loaded_bundle_reads_the_binary_concepts_bool_keys_as_strings(bundle_dir):
    loaded = load_bundle(bundle_dir, "cpu").model
    assert loaded.concept_value2id["bleached"] == {"not_given": 0, "False": 1, "True": 2}


def test_loaded_bundle_exposes_the_sidecar_contract(bundle_dir):
    bundle = load_bundle(bundle_dir, "cpu")
    assert bundle.path == bundle_dir
    assert bundle.id2label == {0: "ignore", 1: "acropora", 2: "porites", 3: "sand", 4: "algae"}
    assert bundle.concept_id2name == _tiny.concept_id2name()
    assert bundle.input_size == (64, 64)
    assert bundle.mean == (0.485, 0.456, 0.406)
    assert bundle.std == (0.229, 0.224, 0.225)
    assert bundle.config == _tiny.resolved_config()


def test_load_never_calls_the_hub(bundle_dir):
    with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("hub")):
        load_bundle(bundle_dir, "cpu")


def test_load_refuses_weights_missing_a_key(bundle_dir, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MODEL_JSON).write_text((bundle_dir / MODEL_JSON).read_text())
    weights = load_file(bundle_dir / WEIGHTS)
    del weights["concept_proj.bias"]
    save_file(weights, broken / WEIGHTS)
    with pytest.raises(RuntimeError, match="concept_proj.bias"):
        load_bundle(broken, "cpu")


@pytest.mark.parametrize(("key", "value"), [("schema_version", 99), ("task", "other")])
def test_load_refuses_a_foreign_schema_version_or_task(bundle_dir, tmp_path, key, value):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    metadata = json.loads((bundle_dir / MODEL_JSON).read_text())
    (foreign / MODEL_JSON).write_text(json.dumps({**metadata, key: value}))
    with pytest.raises(ValueError, match=key):
        load_bundle(foreign, "cpu")


_MISSING = object()


@pytest.mark.parametrize(
    "weights", ["../weights.safetensors", "/etc/weights.safetensors", _MISSING]
)
def test_load_refuses_a_weights_path_other_than_the_bundle_weights(bundle_dir, tmp_path, weights):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    metadata = json.loads((bundle_dir / MODEL_JSON).read_text())
    if weights is _MISSING:
        del metadata["weights"]
    else:
        metadata["weights"] = weights
    (foreign / MODEL_JSON).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="weights"):
        load_bundle(foreign, "cpu")


def test_predictor_preprocesses_with_the_bundles_own_statistics(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run", mean=[0.5] * 3, std=[0.25] * 3)
    bundle = load_bundle(export_bundle(checkpoint, tmp_path / "bundle"), "cpu")
    predictor = CBMPredictor.from_bundle(bundle, amp_dtype=None)
    image = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    expected = (resized.astype(np.float32) / 255 - 0.5) / 0.25
    np.testing.assert_allclose(
        predictor.preprocess(image).numpy(), expected.transpose(2, 0, 1), atol=1e-5
    )


@pytest.mark.parametrize(
    ("key", "foreign"), [("channel_order", "BGR"), ("interpolation", "nearest")]
)
def test_load_rejects_a_foreign_preprocessing_key(bundle_dir, tmp_path, key, foreign):
    copied = tmp_path / "copy"
    shutil.copytree(bundle_dir, copied)
    metadata = json.loads((copied / MODEL_JSON).read_text())
    metadata["preprocessing"][key] = foreign
    (copied / MODEL_JSON).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=key):
        load_bundle(copied, "cpu")
