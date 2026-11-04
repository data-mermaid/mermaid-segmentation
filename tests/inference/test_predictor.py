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


def _eager_predictor(graph_bundle):
    bundle = load_bundle(graph_bundle, format="checkpoint")
    return CBMPredictor.from_bundle(bundle), bundle.model


def _images(batch: int) -> torch.Tensor:
    return torch.randn(batch, 3, 64, 64, generator=torch.Generator().manual_seed(7))


def _expected_concepts(model, x):
    return model(x).hidden_states


def _expected_dpt(model, x):
    feat = model.dpt_head(model._encode(x))
    return F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)


def _expected_backbone(model, x):
    # Last selected hidden state, CLS dropped, laid out as the 4x4 patch grid of a 64x64 input
    # with 16-pixel patches.
    tokens = model._encode(x)[-1][:, 1:, :]
    grid = tokens.permute(0, 2, 1).reshape(x.shape[0], 32, 4, 4)
    return F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)


@pytest.mark.parametrize(
    ("kind", "channels", "expected"),
    [
        ("concepts", 14, _expected_concepts),  # 6 ranks x 2 values + 2 tail concepts
        ("dpt", 8, _expected_dpt),  # fusion_hidden_size
        ("backbone", 32, _expected_backbone),  # encoder hidden_size
    ],
)
def test_eager_forward_features_are_the_models_own_submodule_output_at_input_resolution(
    graph_bundle, kind, channels, expected
):
    # Catches: "concepts" returning another output, "dpt" skipping the resize or reading the
    # wrong submodule, "backbone" taking a different hidden state or keeping the CLS token.
    predictor, model = _eager_predictor(graph_bundle)
    x = _images(2)

    got = predictor.forward_features(x, kind)

    assert got.shape == (2, channels, 64, 64)
    with torch.no_grad():
        reference = expected(model, x)
    assert torch.allclose(got, reference, atol=1e-5, rtol=0)


@pytest.mark.parametrize("kind", ["bogus", "Concepts"])
def test_forward_features_rejects_an_unknown_kind(graph_bundle, kind):
    # Catches: dropping the kind guard, which sends any unknown kind down the backbone path.
    predictor, _ = _eager_predictor(graph_bundle)
    with pytest.raises(ValueError, match="Unknown feature kind"):
        predictor.forward_features(_images(1), kind)


def test_eager_forward_concept_logits_are_the_models_pre_activation_logits(graph_bundle):
    # Catches: returning the activated concept probabilities instead of the raw logits.
    predictor, model = _eager_predictor(graph_bundle)
    x = _images(2)

    got = predictor.forward_concept_logits(x)

    assert got.shape == (2, 14, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(got, out.concept_logits, atol=1e-5, rtol=0)
    assert not torch.allclose(got, out.hidden_states, atol=1e-3)


@pytest.mark.parametrize(
    "points",
    [[1.0, 2.0], [[1, 2, 3]], [[[1, 2]]]],
    ids=["one_dimensional", "three_columns", "three_dimensional"],
)
def test_predict_points_rejects_points_that_are_not_n_by_2(graph_bundle, points):
    # Catches: dropping the shape guard, which would silently read the first two columns.
    predictor, _ = _eager_predictor(graph_bundle)
    with pytest.raises(ValueError, match=r"must be \(N, 2\)"):
        predictor.predict_points(_image(40, 50), points)


def test_activate_concept_logits_softmaxes_each_rank_and_sigmoids_the_tail(graph_bundle):
    # Six rank pairs (each width 2) then two tail concepts, over two pixels. Pixel 0 is all
    # zeros. At pixel 1 even ranks hold (0, ln 3) and odd ranks (ln 3, 0); the tail holds
    # (ln 3, -ln 3). softmax(0, ln 3) = (1/4, 3/4); sigmoid(ln 3) = 3/4, sigmoid(-ln 3) = 1/4.
    # Catches: sigmoid on a rank (0.5, 0.75), softmax on the tail (0.9, 0.1), a shifted offset.
    predictor, _ = _eager_predictor(graph_bundle)
    ln3 = 1.0986123
    column = [0.0, ln3, ln3, 0.0] * 3 + [ln3, -ln3]
    logits = torch.zeros(14, 1, 2)
    logits[:, 0, 1] = torch.tensor(column)

    got = predictor.activate_concept_logits(logits)

    quarter_pairs = [0.25, 0.75, 0.75, 0.25] * 3
    expected = torch.zeros(14, 1, 2)
    expected[:, 0, 0] = 0.5
    expected[:, 0, 1] = torch.tensor(quarter_pairs + [0.75, 0.25])
    assert got.shape == (14, 1, 2)
    assert torch.allclose(got, expected, atol=1e-6, rtol=0)
