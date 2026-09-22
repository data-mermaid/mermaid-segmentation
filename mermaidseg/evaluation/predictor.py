"""CBM inference helpers for the evaluation suite.

Wraps a loaded concept-bottleneck model with:

- preprocessing that matches the training/demo val transform (Resize to the
  model ``input_size`` + ImageNet normalize),
- batched forward returning class + concept probabilities at input resolution,
- per-pixel feature extraction for linear probing (concepts / DPT / backbone),
- exact source-resolution point sampling via ``grid_sample`` (equivalent to a
  bilinear upsample of the input-resolution map followed by indexing, but
  without materialising the huge full-resolution map).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

FEATURE_KINDS = ("concepts", "dpt", "backbone")


def make_eval_transform(input_size: tuple[int, int]) -> A.Compose:
    """Albumentations transform matching the demo/val pipeline (Resize + Normalize)."""
    height, width = int(input_size[0]), int(input_size[1])
    return A.Compose(
        [
            A.Resize(height=height, width=width, p=1),
            A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def preprocess_image(image_rgb_uint8: NDArray[np.uint8], transform: A.Compose) -> torch.Tensor:
    """Resize + normalize an HxWx3 uint8 RGB image to a (3, h, w) float tensor."""
    normalized = transform(image=image_rgb_uint8)["image"]
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous().float()


class CBMPredictor:
    """Inference wrapper around a concept-bottleneck model."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_size: tuple[int, int],
        device: torch.device | str = "cuda",
        amp: bool = False,
    ):
        self.model = model.eval()
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.device = torch.device(device)
        self.amp = bool(amp)
        self.transform = make_eval_transform(self.input_size)

        if not hasattr(model, "concept_classifier"):
            raise ValueError("Model does not expose a `concept_classifier`; is this a CBM model?")
        self.num_concepts = int(model.concept_classifier.in_channels)
        self.num_classes = int(model.concept_classifier.out_channels)
        self.patch_size = int(getattr(model, "patch_size", 16))

    # -- preprocessing ---------------------------------------------------------
    def preprocess(self, image_rgb_uint8: NDArray[np.uint8]) -> torch.Tensor:
        return preprocess_image(image_rgb_uint8, self.transform)

    def _autocast(self):
        if self.amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    # -- forward passes --------------------------------------------------------
    @torch.no_grad()
    def forward_probs(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class_probs, concept_probs)`` at input resolution.

        Shapes: ``class_probs (B, num_classes, h, w)``, ``concept_probs
        (B, num_concepts, h, w)``. Both are fp32 on the model device.
        """
        images = images.to(self.device, non_blocking=True)
        with self._autocast():
            outputs = self.model(images)
        class_probs = torch.softmax(outputs.logits.float(), dim=1)
        concept_probs = outputs.hidden_states.float()
        return class_probs, concept_probs

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor, kind: str = "concepts") -> torch.Tensor:
        """Return dense per-pixel features ``(B, D, h, w)`` at input resolution."""
        if kind not in FEATURE_KINDS:
            raise ValueError(f"Unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")
        images = images.to(self.device, non_blocking=True)

        if kind == "concepts":
            with self._autocast():
                outputs = self.model(images)
            return outputs.hidden_states.float()

        if not (hasattr(self.model, "_encode") and hasattr(self.model, "dpt_head")):
            raise NotImplementedError(
                f"Feature kind {kind!r} requires a DPT-based CBM model with `_encode`/`dpt_head`."
            )

        with self._autocast():
            hidden_states = self.model._encode(images)

        if kind == "dpt":
            feat = self.model.dpt_head(hidden_states)
            feat = F.interpolate(
                feat.float(), size=images.shape[-2:], mode="bilinear", align_corners=False
            )
            return feat

        # kind == "backbone": last selected hidden state, patch tokens only.
        hs = hidden_states[-1]  # (B, 1 + P, C) -- CLS at index 0, register tokens already dropped
        patch = hs[:, 1:, :]
        b, num_patches, channels = patch.shape
        th = images.shape[-2] // self.patch_size
        tw = images.shape[-1] // self.patch_size
        if th * tw != num_patches:
            raise ValueError(
                f"Cannot reshape {num_patches} patch tokens into {th}x{tw} grid "
                f"(input {tuple(images.shape[-2:])}, patch_size {self.patch_size})."
            )
        feat = patch.transpose(1, 2).reshape(b, channels, th, tw)
        feat = F.interpolate(
            feat.float(), size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        return feat

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

    @staticmethod
    def upsample_scores(scores: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Bilinearly upsample a ``(K, h, w)`` score map to ``(K, height, width)``."""
        if scores.dim() != 3:
            raise ValueError(f"scores must be (K, h, w); got {tuple(scores.shape)}")
        out = F.interpolate(
            scores.unsqueeze(0).float(),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )
        return out[0]
