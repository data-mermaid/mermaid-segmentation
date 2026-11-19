"""CBMPredictor, its shared class postprocessing, and exact source-resolution point sampling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mermaidseg.inference.predictor import CBMPredictor, class_predictions, class_probabilities
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD


class _StubModel(nn.Module):
    """CBM-shaped stub that records the CPU autocast state during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.concept_classifier = nn.Conv2d(3, 2, 1)
        self.seen: dict[str, object] = {}

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        self.seen["enabled"] = torch.is_autocast_enabled("cpu")
        self.seen["dtype"] = torch.get_autocast_dtype("cpu")
        return SimpleNamespace(logits=self.concept_classifier(x), hidden_states=x)


def _predictor(
    model: nn.Module, amp_dtype: torch.dtype | None = None, size: tuple[int, int] = (8, 8)
) -> CBMPredictor:
    return CBMPredictor(
        model,
        input_size=size,
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
    model = _StubModel()
    _predictor(model, torch.bfloat16).forward_probs(torch.zeros(1, 3, 8, 8))
    assert model.seen == {"enabled": True, "dtype": torch.bfloat16}


def test_no_amp_dtype_runs_forward_without_autocast():
    model = _StubModel()
    _predictor(model).forward_probs(torch.zeros(1, 3, 8, 8))
    assert model.seen["enabled"] is False


def test_forward_runs_in_eval_mode_under_inference_mode():
    model = _StubModel()
    seen: dict[str, bool] = {}
    model.register_forward_pre_hook(
        lambda m, _a: seen.update(training=m.training, inference=torch.is_inference_mode_enabled())
    )
    _predictor(model).forward_probs(torch.zeros(1, 3, 8, 8))
    assert seen == {"training": False, "inference": True}


def test_predict_points_equals_forward_probs_upsampled_and_indexed():
    torch.manual_seed(0)
    predictor = _predictor(_StubModel())
    image = np.random.default_rng(0).integers(0, 256, (21, 30, 3), dtype=np.uint8)
    points = np.array([[0, 0], [10, 15], [20, 29], [5, 28]])

    class_pts, concept_pts = predictor.predict_points(image, points)

    class_probs, concept_probs = predictor.forward_probs(predictor.preprocess(image)[None])
    size = image.shape[:2]
    up_class = F.interpolate(class_probs, size=size, mode="bilinear", align_corners=False)[0]
    up_concept = F.interpolate(concept_probs, size=size, mode="bilinear", align_corners=False)[0]
    rows, cols = points[:, 0], points[:, 1]
    assert class_pts.shape == (4, 2)
    assert concept_pts.shape == (4, 3)
    assert torch.allclose(class_pts, up_class[:, rows, cols].T, atol=1e-5)
    assert torch.allclose(concept_pts, up_concept[:, rows, cols].T, atol=1e-5)


@pytest.mark.parametrize("points", [np.zeros((4, 3)), np.zeros(4), np.zeros((2, 2, 2))])
def test_predict_points_rejects_points_that_are_not_n_by_2(points):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        _predictor(_StubModel()).predict_points(image, points)


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


class _LogitModel(nn.Module):
    """CBM-shaped stub whose forward output carries ``concept_logits``."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.concept_classifier = nn.Conv2d(3, 2, 1)
        self._logits = logits

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=x, hidden_states=x, concept_logits=self._logits)


def test_forward_concept_logits_returns_the_models_concept_logits_in_float32():
    raw = torch.tensor([[[[1.5]], [[-2.0]]]], dtype=torch.bfloat16)  # (1, 2, 1, 1)
    out = _predictor(_LogitModel(raw)).forward_concept_logits(torch.zeros(1, 3, 8, 8))
    assert out.dtype == torch.float32
    assert out.tolist() == [[[[1.5]], [[-2.0]]]]


def test_forward_concept_logits_runs_under_the_configured_autocast():
    class Recording(_LogitModel):
        enabled: bool | None = None

        def forward(self, x):
            self.enabled = torch.is_autocast_enabled("cpu")
            return super().forward(x)

    model = Recording(torch.zeros(1, 2, 1, 1))
    _predictor(model, torch.bfloat16).forward_concept_logits(torch.zeros(1, 3, 8, 8))
    assert model.enabled is True


def test_forward_concept_logits_raises_when_the_output_has_no_such_field():
    # _StubModel's output carries only ``logits`` and ``hidden_states``.
    with pytest.raises(RuntimeError, match="did not return concept_logits"):
        _predictor(_StubModel()).forward_concept_logits(torch.zeros(1, 3, 8, 8))


def _activating(activation) -> CBMPredictor:
    model = _StubModel()
    model.concept_outputs_activation = activation  # type: ignore[assignment]
    return _predictor(model)


def test_activate_concept_logits_applies_the_activation_to_a_batched_copy_and_unbatches():
    seen: list[tuple[int, ...]] = []

    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(x.shape))
        return torch.sigmoid(x)

    out = _activating(sigmoid).activate_concept_logits(torch.zeros(2, 3, 4))
    assert seen == [(1, 2, 3, 4)]
    assert out.shape == (2, 3, 4)
    assert torch.all(out == 0.5)


def test_activate_concept_logits_rejects_logits_that_are_not_k_h_w():
    with pytest.raises(ValueError, match=r"must be \(K, H, W\)"):
        _activating(torch.sigmoid).activate_concept_logits(torch.zeros(1, 2, 3, 4))


def test_activate_concept_logits_requires_the_model_to_define_an_activation():
    with pytest.raises(RuntimeError, match="no concept_outputs_activation"):
        _predictor(_StubModel()).activate_concept_logits(torch.zeros(2, 3, 4))


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
        _activating(activation).activate_concept_logits(torch.zeros(2, 3, 4))


def test_a_model_without_a_concept_classifier_is_rejected():
    with pytest.raises(ValueError, match="concept_classifier"):
        _predictor(nn.Identity())


@pytest.mark.parametrize("shape", [(5, 5), (1, 2, 5, 5)])
def test_sample_at_points_rejects_maps_that_are_not_c_h_w(shape):
    with pytest.raises(ValueError, match=r"must be \(C, h, w\)"):
        CBMPredictor.sample_at_points(torch.zeros(shape), [0], [0], 5, 5)
