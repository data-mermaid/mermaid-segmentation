"""Loading a bundle offline: the sidecar contract, the weights, and refusals of foreign bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from safetensors.torch import load_file, save_file

from mermaidseg.export import export_bundle
from mermaidseg.inference.bundle import (
    MODEL_JSON,
    WEIGHTS,
    WEIGHTS_SHA256,
    load_bundle,
    sha256_file,
)
from mermaidseg.inference.predictor import CBMPredictor
from tests.mermaidseg.inference import _tiny


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
    weights = load_file(bundle_dir / WEIGHTS)
    del weights["concept_proj.bias"]
    save_file(weights, broken / WEIGHTS)
    metadata = json.loads((bundle_dir / MODEL_JSON).read_text())
    metadata[WEIGHTS_SHA256] = hashlib.sha256((broken / WEIGHTS).read_bytes()).hexdigest()
    (broken / MODEL_JSON).write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="concept_proj.bias"):
        load_bundle(broken, "cpu")


def test_load_refuses_weights_with_a_changed_value(bundle_dir, tmp_path):
    copied = tmp_path / "copy"
    shutil.copytree(bundle_dir, copied)
    weights = load_file(copied / WEIGHTS)
    name = next(iter(weights))
    weights[name] = weights[name] + 1
    save_file(weights, copied / WEIGHTS)
    with pytest.raises(ValueError, match="weights_sha256"):
        load_bundle(copied, "cpu")


def test_load_refuses_truncated_weights_before_parsing_them(bundle_dir, tmp_path):
    copied = tmp_path / "copy"
    shutil.copytree(bundle_dir, copied)
    data = (copied / WEIGHTS).read_bytes()
    (copied / WEIGHTS).write_bytes(data[:-8])
    with pytest.raises(ValueError, match="weights_sha256"):
        load_bundle(copied, "cpu")


def test_load_refuses_a_bundle_without_a_recorded_checksum(bundle_dir, tmp_path):
    copied = tmp_path / "copy"
    shutil.copytree(bundle_dir, copied)
    metadata = json.loads((copied / MODEL_JSON).read_text())
    del metadata[WEIGHTS_SHA256]
    (copied / MODEL_JSON).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="weights_sha256"):
        load_bundle(copied, "cpu")


def test_sha256_file_matches_hashlib_across_chunk_boundaries(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 123))
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("key", "value"),
    [("schema_version", 99), ("task", "other"), ("weights", "../weights.safetensors")],
)
def test_load_refuses_a_foreign_schema_version_task_or_weights_path(
    bundle_dir, tmp_path, key, value
):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    metadata = json.loads((bundle_dir / MODEL_JSON).read_text())
    (foreign / MODEL_JSON).write_text(json.dumps({**metadata, key: value}))
    with pytest.raises(ValueError, match=key):
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
