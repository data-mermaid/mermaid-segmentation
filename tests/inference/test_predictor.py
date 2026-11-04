"""CBMPredictor.predict / predict_points over both the eager and the graph bundle."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mermaidseg.inference import CBMPredictor, load_bundle, preprocess


def _image(height: int, width: int) -> np.ndarray:
    return np.random.default_rng(3).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("fmt", ["checkpoint", "graph"])
def test_predict_points_equals_upsampled_predict_indexed_at_the_points(graph_bundle, fmt):
    predictor = CBMPredictor.from_bundle(load_bundle(graph_bundle, format=fmt))
    image = _image(100, 130)
    points = np.array([[0, 0], [99, 129], [50, 7], [13, 88], [71, 64]])

    class_probs, concept_probs = predictor.predict(image)
    up_class = F.interpolate(class_probs[None], size=(100, 130), mode="bilinear")[0]
    up_concept = F.interpolate(concept_probs[None], size=(100, 130), mode="bilinear")[0]

    point_class, point_concept = predictor.predict_points(image, points)
    rows, cols = points[:, 0], points[:, 1]
    assert torch.allclose(point_class, up_class[:, rows, cols].T, atol=1e-5, rtol=0)
    assert torch.allclose(point_concept, up_concept[:, rows, cols].T, atol=1e-5, rtol=0)


def test_predict_is_softmaxed_model_output_on_the_preprocessed_image(graph_bundle):
    bundle = load_bundle(graph_bundle, format="checkpoint")
    image = _image(80, 90)
    class_probs, concept_probs = CBMPredictor.from_bundle(bundle).predict(image)

    with torch.no_grad():
        out = bundle.model(preprocess(image, bundle.input_size)[None])
    assert class_probs.shape == (len(bundle.classes), *bundle.input_size)
    assert torch.allclose(class_probs, torch.softmax(out.logits, dim=1)[0], atol=1e-6, rtol=0)
    assert torch.allclose(concept_probs, out.hidden_states[0], atol=1e-6, rtol=0)


def test_graph_predictor_refuses_amp(graph_bundle):
    with pytest.raises(ValueError, match="amp"):
        CBMPredictor.from_bundle(load_bundle(graph_bundle, format="graph"), amp=True)


def test_graph_predictor_rejects_a_batch_of_two(graph_bundle):
    bundle = load_bundle(graph_bundle, format="graph")
    predictor = CBMPredictor.from_bundle(bundle)
    with pytest.raises(ValueError, match='format="checkpoint"'):
        predictor.forward_probs(torch.zeros(2, 3, *bundle.input_size))


@pytest.mark.parametrize(
    "call",
    [
        lambda predictor, x: predictor.forward_features(x, "concepts"),
        lambda predictor, x: predictor.forward_features(x, "dpt"),
        lambda predictor, x: predictor.forward_concept_logits(x),
    ],
    ids=["concept_features", "dpt_features", "concept_logits"],
)
def test_graph_predictor_rejects_feature_calls(graph_bundle, call):
    bundle = load_bundle(graph_bundle, format="graph")
    predictor = CBMPredictor.from_bundle(bundle)
    with pytest.raises(NotImplementedError, match='format="checkpoint"'):
        call(predictor, torch.zeros(1, 3, *bundle.input_size))
