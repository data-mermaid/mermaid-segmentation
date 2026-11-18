"""CBM predictor shared by the evaluation suite and serving.

Wraps a loaded concept-bottleneck model with:

- preprocessing that matches the training/val transform (Resize to the model ``input_size`` +
  normalize), through ``mermaidseg.inference.preprocess``,
- batched forward returning class + concept probabilities at input resolution,
- exact source-resolution point sampling via ``grid_sample`` (equivalent to a bilinear upsample
  of the input-resolution map followed by indexing, without materialising the full-resolution
  map).

``class_probabilities`` and ``class_predictions`` are the postprocessing that in-training
validation and inference share. This module imports only torch, numpy and the preprocess module
so serving can load it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from mermaidseg.inference.preprocess import preprocess

if TYPE_CHECKING:
    from mermaidseg.inference.bundle import ModelBundle


def class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax of ``(B, C, h, w)`` class logits over the class dimension, in float32."""
    return torch.softmax(logits.float(), dim=1)


def class_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Argmax of ``(B, C, h, w)`` class logits over the class dimension: ``(B, h, w)`` ids."""
    return logits.argmax(dim=1)


class CBMPredictor:
    """Inference wrapper around a concept-bottleneck model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        input_size: tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float],
        device: torch.device | str,
        amp_dtype: torch.dtype | None,
    ):
        """Wrap a model for inference.

        Args:
            model: Concept-bottleneck model exposing ``concept_classifier``.
            input_size: ``(height, width)`` the images are resized to.
            mean: Per-channel normalisation mean.
            std: Per-channel normalisation standard deviation.
            device: Device the inputs are moved to.
            amp_dtype: Autocast dtype for the forward pass, or ``None`` to run without autocast.

        Raises:
            ValueError: If ``model`` has no ``concept_classifier``.
        """
        # Duck-typed: CBM variants expose attributes (concept_outputs_activation) that nn.Module hides.
        self.model: Any = model.eval()
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype

        if not hasattr(model, "concept_classifier"):
            raise ValueError("Model does not expose a `concept_classifier`; is this a CBM model?")
        classifier = cast("Any", model.concept_classifier)
        self.num_concepts = int(classifier.in_channels)
        self.num_classes = int(classifier.out_channels)

    @classmethod
    def from_bundle(cls, bundle: ModelBundle, *, amp_dtype: torch.dtype | None) -> CBMPredictor:
        """Build a predictor from a loaded bundle's model, preprocessing and device."""
        return cls(
            bundle.model,
            input_size=bundle.input_size,
            mean=bundle.mean,
            std=bundle.std,
            device=bundle.device,
            amp_dtype=amp_dtype,
        )

    # -- preprocessing ---------------------------------------------------------
    def preprocess(self, image_rgb_uint8: NDArray[np.uint8]) -> torch.Tensor:
        """Resize and normalize an RGB image to the model input size."""
        return preprocess(image_rgb_uint8, self.input_size, self.mean, self.std)

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        )

    # -- forward passes --------------------------------------------------------
    @torch.inference_mode()
    def forward_probs(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class_probs, concept_probs)`` at input resolution.

        Shapes: ``class_probs (B, num_classes, h, w)``, ``concept_probs
        (B, num_concepts, h, w)``. Both are fp32 on the model device.
        """
        images = images.to(self.device, non_blocking=True)
        with self._autocast():
            outputs = self.model(images)
        return class_probabilities(outputs.logits), outputs.hidden_states.float()

    @torch.inference_mode()
    def predict_points(
        self,
        image_rgb_uint8: NDArray[np.uint8],
        points_rowcol: Sequence[Sequence[float]] | NDArray[np.number],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-point ``(class_probs (N, C), concept_probs (N, K))`` on CPU.

        ``points_rowcol`` is ``(N, 2)`` ``(row, col)`` pixel coordinates in the source image.
        Values equal bilinearly upsampling the input-resolution maps to the source size and
        indexing there.

        Raises:
            ValueError: If ``points_rowcol`` is not ``(N, 2)``.
        """
        points = np.asarray(points_rowcol)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points_rowcol must be (N, 2) (row, col); got shape {points.shape}")
        height, width = image_rgb_uint8.shape[:2]
        class_probs, concept_probs = self.forward_probs(self.preprocess(image_rgb_uint8)[None])
        rows, cols = points[:, 0], points[:, 1]
        point_class = self.sample_at_points(class_probs[0], rows, cols, height, width)
        point_concept = self.sample_at_points(concept_probs[0], rows, cols, height, width)
        return point_class.T.cpu(), point_concept.T.cpu()

    @torch.inference_mode()
    def forward_concept_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw concept logits ``(B, K, h, w)`` at input resolution, fp32.

        These are the pre-activation values (softmax has not been applied to
        taxonomic groups, and sigmoid has not been applied to the binary tail).
        """
        images = images.to(self.device, non_blocking=True)
        with self._autocast():
            outputs = self.model(images)
        logits = getattr(outputs, "concept_logits", None)
        if logits is None:
            raise RuntimeError(
                "Model forward did not return concept_logits. "
                "Strided Coralscapes eval averages those logits before activation."
            )
        return logits.float()

    def activate_concept_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the model's concept activation to ``(K, H, W)`` averaged logits.

        Taxonomic groups are softmaxed and the binary tail is sigmoided, matching
        ``concept_outputs_activation`` used at training time. The result stays on
        the same device as ``logits``.
        """
        if logits.dim() != 3:
            raise ValueError(f"logits must be (K, H, W); got {tuple(logits.shape)}")
        activate = getattr(self.model, "concept_outputs_activation", None)
        if activate is None:
            raise RuntimeError(
                "Model has no concept_outputs_activation, so averaged concept logits "
                "cannot be turned into the features the linear probe expects."
            )
        activated = activate(logits.unsqueeze(0))
        if activated.dim() != 4 or activated.shape[0] != 1:
            raise RuntimeError(
                f"concept_outputs_activation must return (1, K, H, W); got {tuple(activated.shape)}"
            )
        activated = activated[0]
        if activated.shape != logits.shape:
            raise RuntimeError(
                f"concept_outputs_activation changed shape from {tuple(logits.shape)} "
                f"to {tuple(activated.shape)}"
            )
        return activated

    # -- source-resolution point sampling -------------------------------------
    @staticmethod
    def sample_at_points(
        maps: torch.Tensor,
        rows: Sequence[int] | NDArray[np.integer],
        cols: Sequence[int] | NDArray[np.integer],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Sample a ``(C, h, w)`` map at source-resolution ``(row, col)`` points.

        Uses ``grid_sample(mode="bilinear", align_corners=False)`` with
        ``x = (col + 0.5) / width * 2 - 1`` and ``y = (row + 0.5) / height * 2 - 1``.
        This is *exactly* equal to bilinearly upsampling ``maps`` from ``(h, w)``
        to ``(height, width)`` (align_corners=False) and indexing ``[.., row,
        col]`` -- but ~orders of magnitude cheaper for large source images.

        Returns a ``(C, N)`` tensor on the same device as ``maps``.
        """
        if maps.dim() != 3:
            raise ValueError(f"maps must be (C, h, w); got {tuple(maps.shape)}")
        device = maps.device
        cols_t = torch.as_tensor(np.asarray(cols), dtype=torch.float32, device=device)
        rows_t = torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=device)
        xs = (cols_t + 0.5) / float(width) * 2.0 - 1.0
        ys = (rows_t + 0.5) / float(height) * 2.0 - 1.0
        grid = torch.stack([xs, ys], dim=-1).view(1, -1, 1, 2)  # (1, N, 1, 2)
        sampled = F.grid_sample(
            maps.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )  # (1, C, N, 1)
        return sampled[0, :, :, 0]
