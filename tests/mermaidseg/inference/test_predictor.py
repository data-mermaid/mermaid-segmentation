"""CBMPredictor, its shared class postprocessing, and exact source-resolution point sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mermaidseg.inference.predictor import CBMPredictor, class_predictions, class_probabilities
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from tests.mermaidseg.inference import _tiny


@pytest.fixture(scope="module")
def tiny_model():
    return _tiny.build_trained_model()


def _record_autocast(model: nn.Module) -> dict[str, object]:
    """Record the CPU autocast state seen at the start of each forward."""
    seen: dict[str, object] = {}

    def hook(_module, _args) -> None:
        seen["enabled"] = torch.is_autocast_enabled("cpu")
        seen["dtype"] = torch.get_autocast_dtype("cpu")

    model.register_forward_pre_hook(hook)
    return seen


def _images(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, 3, *_tiny.INPUT_SIZE)


def _predictor(model: nn.Module, amp_dtype: torch.dtype | None = None) -> CBMPredictor:
    return CBMPredictor(
        model,  # type: ignore[arg-type]
        input_size=_tiny.INPUT_SIZE,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device="cpu",
        amp_dtype=amp_dtype,
    )


def test_class_predictions_is_the_argmax_over_the_class_dimension():
    logits = torch.tensor([[[[1.0, 5.0]], [[3.0, 2.0]], [[0.0, 9.0]]]])  # (1, 3, 1, 2)
    assert class_predictions(logits).tolist() == [[[1, 2]]]


def test_class_probabilities_is_a_float32_softmax_over_the_class_dimension():
    logits = torch.tensor([[[[0.0]], [[0.0]], [[float(np.log(2.0))]]]], dtype=torch.bfloat16)
    probs = class_probabilities(logits)
    assert probs.dtype == torch.float32
    assert probs[0, :, 0, 0].tolist() == pytest.approx([0.25, 0.25, 0.5], abs=1e-2)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


def test_amp_dtype_runs_forward_under_cpu_autocast():
    model = _tiny.build_trained_model()
    seen = _record_autocast(model)
    _predictor(model, torch.bfloat16).forward_probs(_images())
    assert seen == {"enabled": True, "dtype": torch.bfloat16}


def test_no_amp_dtype_runs_forward_without_autocast():
    model = _tiny.build_trained_model()
    seen = _record_autocast(model)
    _predictor(model).forward_probs(_images())
    assert seen["enabled"] is False


def test_forward_runs_in_eval_mode_under_inference_mode():
    model = _tiny.build_trained_model().train()
    seen: dict[str, bool] = {}
    model.register_forward_pre_hook(
        lambda m, _a: seen.update(training=m.training, inference=torch.is_inference_mode_enabled())
    )
    _predictor(model).forward_probs(_images())
    assert seen == {"training": False, "inference": True}


def test_predict_points_equals_forward_probs_upsampled_and_indexed(tiny_model):
    predictor = _predictor(tiny_model)
    image = np.random.default_rng(0).integers(0, 256, (21, 30, 3), dtype=np.uint8)
    points = np.array([[0, 0], [10, 15], [20, 29], [5, 28]])

    class_pts, concept_pts = predictor.predict_points(image, points)

    class_probs, concept_probs = predictor.forward_probs(predictor.preprocess(image)[None])
    size = image.shape[:2]
    up_class = F.interpolate(class_probs, size=size, mode="bilinear", align_corners=False)[0]
    up_concept = F.interpolate(concept_probs, size=size, mode="bilinear", align_corners=False)[0]
    rows, cols = points[:, 0], points[:, 1]
    assert class_pts.shape == (4, len(_tiny.CLASS_NAMES))
    assert concept_pts.shape == (4, len(_tiny.CONCEPT_NAMES))
    assert torch.allclose(class_pts, up_class[:, rows, cols].T, atol=1e-5)
    assert torch.allclose(concept_pts, up_concept[:, rows, cols].T, atol=1e-5)


@pytest.mark.parametrize("points", [np.zeros((4, 3)), np.zeros(4), np.zeros((2, 2, 2))])
def test_predict_points_rejects_points_that_are_not_n_by_2(tiny_model, points):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        _predictor(tiny_model).predict_points(image, points)


def test_sample_at_points_equals_interpolate_and_index():
    torch.manual_seed(0)
    c, h, w = 6, 9, 11
    height, width = 71, 83
    maps = torch.randn(c, h, w)
    up = F.interpolate(
        maps.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )[0]
    rows = np.array([0, 5, 40, 70, height - 1, 33])
    cols = np.array([0, 6, 82, 1, width - 1, 50])
    ref = up[:, rows, cols]
    got = CBMPredictor.sample_at_points(maps, rows, cols, height, width)
    assert torch.allclose(ref, got, atol=1e-5)


def test_forward_concept_logits_returns_the_models_concept_logits_in_float32(tiny_model):
    images = _images()
    with torch.inference_mode():
        expected = tiny_model(images).concept_logits
    out = _predictor(tiny_model).forward_concept_logits(images)
    assert out.dtype == torch.float32
    assert out.shape == (1, len(_tiny.CONCEPT_NAMES), *_tiny.INPUT_SIZE)
    assert torch.equal(out, expected)


def test_forward_concept_logits_runs_under_the_configured_autocast():
    model = _tiny.build_trained_model()
    seen = _record_autocast(model)
    _predictor(model, torch.bfloat16).forward_concept_logits(_images())
    assert seen["enabled"] is True


def _activating(model: nn.Module, activation) -> CBMPredictor:
    model.concept_outputs_activation = activation  # type: ignore[assignment]
    return _predictor(model)


def test_activate_concept_logits_applies_the_activation_to_a_batched_copy_and_unbatches():
    seen: list[tuple[int, ...]] = []

    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(x.shape))
        return torch.sigmoid(x)

    out = _activating(_tiny.build_trained_model(), sigmoid).activate_concept_logits(
        torch.zeros(2, 3, 4)
    )
    assert seen == [(1, 2, 3, 4)]
    assert out.shape == (2, 3, 4)
    assert torch.all(out == 0.5)


def test_activate_concept_logits_rejects_logits_that_are_not_k_h_w():
    with pytest.raises(ValueError, match=r"must be \(K, H, W\)"):
        _activating(_tiny.build_trained_model(), torch.sigmoid).activate_concept_logits(
            torch.zeros(1, 2, 3, 4)
        )


@pytest.mark.parametrize(
    ("activation", "match"),
    [
        (lambda x: x[0], r"must return \(1, K, H, W\)"),
        (lambda x: x.repeat(2, 1, 1, 1), r"must return \(1, K, H, W\)"),
        (lambda x: x[:, :1], "changed shape"),
    ],
    ids=["three_dimensional", "batch_of_two", "channels_dropped"],
)
def test_activate_concept_logits_rejects_an_activation_that_changes_the_shape(activation, match):
    with pytest.raises(RuntimeError, match=match):
        _activating(_tiny.build_trained_model(), activation).activate_concept_logits(
            torch.zeros(2, 3, 4)
        )


@pytest.mark.parametrize("shape", [(5, 5), (1, 2, 5, 5)])
def test_sample_at_points_rejects_maps_that_are_not_c_h_w(shape):
    with pytest.raises(ValueError, match=r"must be \(C, h, w\)"):
        CBMPredictor.sample_at_points(torch.zeros(shape), [0], [0], 5, 5)
