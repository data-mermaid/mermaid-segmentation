"""CBM inference wrapper shared by the demos, the evaluation suite and serving.

Wraps a concept-bottleneck model (eager module or ``torch.export`` program) with:

- preprocessing that matches the training val transform (``preprocess``),
- batched forward returning class + concept probabilities at input resolution,
- per-pixel feature extraction for linear probing (concepts / DPT / backbone; eager only),
- exact source-resolution point sampling via ``grid_sample`` (equivalent to a bilinear upsample
  of the input-resolution map followed by indexing, without materialising the full-resolution
  map).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess

if TYPE_CHECKING:
    from mermaidseg.inference.bundle import ModelBundle

logger = logging.getLogger(__name__)

FEATURE_KINDS = ("concepts", "dpt", "backbone")
_GRAPH_SCOPE = (
    "an exported graph serves single images only; load the bundle with "
    'format="checkpoint" for batched or feature evaluation'
)


class CBMPredictor:
    """Inference wrapper around a concept-bottleneck model or its exported graph.

    A graph-backed predictor (a ``torch.export`` program, e.g. from ``load_bundle`` with
    ``format="auto"`` on a bundle that has ``model.pt2``) is single-image serving only: it runs
    ``predict``/``predict_points`` and ``forward_probs`` on a batch of one, in fp32. Batched
    forwards, concept logits, feature extraction and ``amp`` need the eager model, so evaluation
    loads the bundle with ``format="checkpoint"``.
    """

    def __init__(
        self,
        model: torch.nn.Module | torch.export.ExportedProgram,
        input_size: tuple[int, int],
        device: torch.device | str = "cpu",
        amp: bool = False,
        *,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        num_classes: int | None = None,
        num_concepts: int | None = None,
        patch_size: int | None = None,
    ):
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.device = torch.device(device)
        self.amp = bool(amp)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

        self.is_graph = isinstance(model, torch.export.ExportedProgram)
        if self.is_graph and self.amp:
            # The graph obeys ambient autocast, so amp would run its fp32 concept tail in bf16.
            raise ValueError(
                "amp=True is not supported for an exported graph; use amp=False, or load the "
                'bundle with format="checkpoint" for mixed precision.'
            )
        if self.is_graph:
            # Exported modules reject .eval(); the program is traced in eval mode already.
            self.model = model.module().to(self.device)
        else:
            self.model = model.eval()

        if hasattr(self.model, "concept_classifier"):
            self.num_concepts = int(self.model.concept_classifier.in_channels)
            self.num_classes = int(self.model.concept_classifier.out_channels)
        elif num_classes is not None and num_concepts is not None:
            self.num_classes = int(num_classes)
            self.num_concepts = int(num_concepts)
        else:
            raise ValueError(
                "Model does not expose a `concept_classifier`; pass num_classes and "
                "num_concepts for an exported graph, or check this is a CBM model."
            )
        self.patch_size = int(
            patch_size if patch_size is not None else getattr(self.model, "patch_size", 16)
        )

    @classmethod
    def from_bundle(
        cls, bundle: ModelBundle, device: torch.device | str | None = None, amp: bool = False
    ) -> CBMPredictor:
        """Predictor over a loaded bundle, using the bundle's input size and normalisation."""
        return cls(
            bundle.model,
            input_size=bundle.input_size,
            device=bundle.device if device is None else device,
            amp=amp,
            mean=bundle.mean,
            std=bundle.std,
            num_classes=bundle.num_classes,
            num_concepts=bundle.num_concepts,
            patch_size=bundle.patch_size,
        )

    # -- preprocessing ---------------------------------------------------------
    def preprocess(self, image_rgb_uint8: NDArray[np.uint8]) -> torch.Tensor:
        return preprocess(image_rgb_uint8, self.input_size, self.mean, self.std)

    def _autocast(self):
        if self.amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    # -- forward passes --------------------------------------------------------
    def _forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class_logits, concept_probs)`` from either an eager model or a graph."""
        if self.is_graph and images.shape[0] != 1:
            raise ValueError(f"Got a batch of {images.shape[0]} images, but {_GRAPH_SCOPE}.")
        images = images.to(self.device, non_blocking=True)
        with self._autocast():
            outputs = self.model(images)
        if isinstance(outputs, tuple | list):
            class_logits, concept_probs = outputs
        else:
            class_logits, concept_probs = outputs.logits, outputs.hidden_states
        return class_logits.float(), concept_probs.float()

    @torch.no_grad()
    def forward_probs(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class_probs, concept_probs)`` at input resolution.

        Shapes: ``class_probs (B, num_classes, h, w)``, ``concept_probs
        (B, num_concepts, h, w)``. Both are fp32 on the model device.
        """
        class_logits, concept_probs = self._forward(images)
        return torch.softmax(class_logits, dim=1), concept_probs

    @torch.no_grad()
    def predict(self, image_rgb_uint8: NDArray[np.uint8]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class_probs (C, h, w), concept_probs (K, h, w))`` at input resolution, on CPU."""
        class_probs, concept_probs = self.forward_probs(self.preprocess(image_rgb_uint8)[None])
        return class_probs[0].cpu(), concept_probs[0].cpu()

    @torch.no_grad()
    def predict_points(
        self,
        image_rgb_uint8: NDArray[np.uint8],
        points_rowcol: Sequence[Sequence[float]] | NDArray[np.number],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-point ``(class_probs (N, C), concept_probs (N, K))`` on CPU.

        ``points_rowcol`` is ``(N, 2)`` ``(row, col)`` pixel coordinates in the source image.
        Values equal bilinearly upsampling ``predict`` to the source size and indexing there.
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

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor, kind: str = "concepts") -> torch.Tensor:
        """Return dense per-pixel features ``(B, D, h, w)`` at input resolution."""
        if kind not in FEATURE_KINDS:
            raise ValueError(f"Unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")
        if self.is_graph:
            raise NotImplementedError(f"forward_features is unavailable: {_GRAPH_SCOPE}.")
        images = images.to(self.device, non_blocking=True)

        if kind == "concepts":
            return self._forward(images)[1]

        if not (hasattr(self.model, "_encode") and hasattr(self.model, "dpt_head")):
            raise NotImplementedError(
                f"Feature kind {kind!r} requires a DPT-based CBM model with `_encode`/`dpt_head`."
            )

        with self._autocast():
            hidden_states = self.model._encode(images)

        if kind == "dpt":
            feat = self.model.dpt_head(hidden_states)
            return F.interpolate(
                feat.float(), size=images.shape[-2:], mode="bilinear", align_corners=False
            )

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
        return F.interpolate(
            feat.float(), size=images.shape[-2:], mode="bilinear", align_corners=False
        )

    @torch.no_grad()
    def forward_concept_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw concept logits ``(B, K, h, w)`` at input resolution, fp32.

        These are the pre-activation values (softmax has not been applied to
        taxonomic groups, and sigmoid has not been applied to the binary tail).
        """
        if self.is_graph:
            raise NotImplementedError(f"forward_concept_logits is unavailable: {_GRAPH_SCOPE}.")
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
