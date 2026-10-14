"""Native-resolution evaluation for sparse point-annotated datasets.

The shared val transform resizes masks to the model's input size, which drops almost all point
annotations (``pacific_labeled_corals``: 864 annotated pixels at 2112x2816, 9 at 256x256). Instead,
each annotated native pixel ``(y, x)`` is mapped onto the model's output grid and the
prediction is read there::

    y_model = floor(y * h_model / H_native)
    x_model = floor(x * w_model / W_native)

Each image becomes a ``1 x P`` strip of its ``P`` points, which is fed to the caller's
:class:`~mermaidseg.model.eval.Evaluator`, so metrics (including any
``rank_output_consumers``) are defined in one place.

``batch_predict``'s ``target_dim`` argument is unused in ``meta.py``; outputs are always at
the model's input resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
import tqdm

from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SparseEvalCoverage:
    """Ground truth scored by a sparse evaluation run.

    Attributes:
        images: Images evaluated.
        annotated_pixels: Annotated pixels found at native resolution.
        scored_points: Points actually scored (equals ``annotated_pixels`` unless capped).
        empty_images: Images with no annotation at all.
        points_per_image: Per-image annotated-pixel counts.
    """

    images: int = 0
    annotated_pixels: int = 0
    scored_points: int = 0
    empty_images: int = 0
    points_per_image: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        mean = float(np.mean(self.points_per_image)) if self.points_per_image else 0.0
        return {
            "coverage/images": float(self.images),
            "coverage/annotated_pixels": float(self.annotated_pixels),
            "coverage/scored_points": float(self.scored_points),
            "coverage/empty_images": float(self.empty_images),
            "coverage/mean_points_per_image": mean,
        }


def strip_transform(split_cfg: Any) -> dict[str, Any]:
    """Return ``split_cfg`` without ``transform``/``augmentation``, so the dataset yields native-
    resolution images and masks."""
    cfg = dict(split_cfg)
    cfg.pop("transform", None)
    cfg.pop("augmentation", None)
    return cfg


def checkpoint_input_size(
    checkpoint: Any, dataset: str, split: str = "val"
) -> tuple[int, int] | None:
    """``(height, width)`` of the val ``Resize`` saved in a training checkpoint's config.

    Prefers ``dataset``/``split`` and otherwise takes any dataset's val ``Resize``: the input size
    belongs to the model, not the dataset. Returns None if the checkpoint saved no config or no
    ``Resize``.
    """
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    data = config.get("data") if isinstance(config, dict) else None
    if not isinstance(data, dict):
        return None
    for name, split_name in [(dataset, split), *((name, "val") for name in data)]:
        split_cfg = data.get(name)
        split_cfg = split_cfg.get(split_name) if isinstance(split_cfg, dict) else None
        transform = split_cfg.get("transform") if isinstance(split_cfg, dict) else None
        for t in getattr(transform, "transforms", []):
            if type(t).__name__ == "Resize":
                return (int(t.height), int(t.width))
    return None


def preprocess_image(
    image: np.ndarray,
    model_size: tuple[int, int],
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Native ``HWC uint8`` image -> normalised ``(1, 3, h, w)`` float tensor.

    Matches the config's val ``Resize`` + ``Normalize``: ``cv2.INTER_LINEAR`` is albumentations'
    default, and ``Normalize`` divides by 255 before standardising.
    """
    height, width = model_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def map_to_model_grid(coords: np.ndarray, native: int, model: int) -> np.ndarray:
    """Map native pixel indices onto the model's output grid.

    Uses ``floor(coord * model / native)``, the same convention as a nearest-neighbour resize,
    clamped to the valid range.
    """
    mapped = np.floor(coords.astype(np.float64) * model / native).astype(np.int64)
    return np.clip(mapped, 0, model - 1)


@torch.no_grad()
def evaluate_sparse_native(
    dataset: Any,
    meta_model: MetaModel,
    evaluator: Evaluator,
    model_size: tuple[int, int] = (256, 256),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    subset_n: int | None = None,
    max_points_per_image: int | None = None,
    progress: bool = True,
    indices: Sequence[int] | None = None,
) -> tuple[dict[str, float], SparseEvalCoverage]:
    """Evaluate a sparse dataset at native annotation resolution.

    Args:
        dataset: Dataset built WITHOUT a transform (see :func:`strip_transform`), yielding
            ``(image HWC uint8, mask HW)`` at native resolution.
        meta_model: The model wrapper.
        evaluator: Supplies the metrics: its ``metric_dict``, concept metrics and
            ``rank_output_consumers`` are updated directly.
        model_size: Spatial size fed to the model; use the training run's val ``Resize``
            (:func:`checkpoint_input_size`).
        mean / std: Normalisation statistics, matching the config's ``Normalize``.
        subset_n: Evaluate only the first ``n`` items. Not a random sample if the dataset
            is sorted (``pacific_labeled_corals`` is ordered by site); prefer ``indices``.
        indices: Explicit item indices to evaluate, overriding ``subset_n``.
        max_points_per_image: Randomly subsample points beyond this cap. ``None`` scores
            every annotated pixel.
        progress: Show a progress bar.

    Returns:
        ``(metric_results, coverage)``.
    """
    meta_model.model.eval()
    device = evaluator.device
    coverage = SparseEvalCoverage()

    if indices is None:
        total = len(dataset) if subset_n is None else min(subset_n, len(dataset))
        indices = range(total)
    iterator = tqdm.tqdm(indices, desc="sparse native eval") if progress else indices

    for index in iterator:
        image, mask = dataset[index]
        mask = np.asarray(mask)
        native_h, native_w = mask.shape[:2]

        ys, xs = np.nonzero(mask)
        coverage.images += 1
        if ys.size == 0:
            coverage.empty_images += 1
            continue
        coverage.annotated_pixels += int(ys.size)

        if max_points_per_image is not None and ys.size > max_points_per_image:
            pick = np.random.default_rng(index).choice(
                ys.size, size=max_points_per_image, replace=False
            )
            ys, xs = ys[pick], xs[pick]
        coverage.scored_points += int(ys.size)
        coverage.points_per_image.append(int(ys.size))

        inputs = preprocess_image(image, model_size, mean, std).to(device)
        logits, concept_outputs = meta_model.batch_predict(inputs)

        model_h, model_w = logits.shape[-2], logits.shape[-1]
        my = torch.from_numpy(map_to_model_grid(ys, native_h, model_h)).to(device)
        mx = torch.from_numpy(map_to_model_grid(xs, native_w, model_w)).to(device)

        # (1, C, 1, P): each image becomes a one-row strip of its annotated points.
        point_logits = logits[0, :, my, mx].unsqueeze(0).unsqueeze(2)
        source_labels = torch.from_numpy(mask[ys, xs].astype(np.int64)).to(device).view(1, 1, -1)
        target_labels = meta_model._to_target_labels(source_labels)

        if evaluator.metric_dict:
            predictions = point_logits.argmax(dim=1)
            for metric in evaluator.metric_dict.values():
                metric.update(predictions, target_labels)

        if meta_model.training_mode in ("concept-bottleneck", "concept") and (
            concept_outputs is not None
        ):
            point_concepts = concept_outputs[0, :, my, mx].unsqueeze(0).unsqueeze(2)
            concept_labels = meta_model._to_concept_labels(source_labels)
            evaluator.evaluate_concepts(point_concepts, concept_labels)

    results: dict[str, float] = {}
    for name, metric in evaluator.metric_dict.items():
        value = metric.compute().cpu().numpy()
        results[name] = float(value.item()) if value.ndim == 0 else value
        metric.reset()
    if meta_model.training_mode in ("concept-bottleneck", "concept"):
        results.update(evaluator._compute_concept_metric_results())
    results.update(coverage.summary())
    return results, coverage
