"""torch.export graph export of a DPT concept-bottleneck model, behind a parity gate."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

GRAPH_TOLERANCE = 1e-5
# Upper bound on the patch-grid side the exported graph accepts (8192 px at patch 16).
MAX_TOKEN_GRID = 512


class ParityError(RuntimeError):
    """The exported graph diverges from the eager model beyond tolerance."""


class _GraphForward(torch.nn.Module):
    """Eager CBM forward that returns ``(class_logits, concept_probs)`` as a plain tuple.

    It composes the model's own submodules, taking the patch grid from the input shape: the
    eager ``DPTHead`` infers the grid from its configured aspect ratio, which ``torch.export``
    specialises to the export-time size.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        hidden_states = model._encode(x)
        token_height = x.shape[-2] // model.patch_size
        token_width = x.shape[-1] // model.patch_size
        features = model.dpt_head.neck(hidden_states, token_height, token_width)
        concept_features = model.dpt_head.head(features[-1])
        concept_features = F.interpolate(
            concept_features, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        class_logits, concept_probs, _ = model._forward_concept_tail(concept_features)
        return class_logits, concept_probs


def _parity_input(size_hw: Sequence[int]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(1, 3, int(size_hw[0]), int(size_hw[1]), generator=generator)


def export_graph(
    model: torch.nn.Module,
    size_hw: Sequence[int],
    out_path: str | Path,
    *,
    atol: float = GRAPH_TOLERANCE,
) -> Path:
    """Export ``model`` with ``torch.export`` (dynamic H/W, multiples of the patch size).

    The program takes a ``(1, 3, H, W)`` float tensor and returns ``(class_logits,
    concept_probs)``. Before anything is written, it is compared with the eager model on a fixed
    ``size_hw`` tensor; a max abs difference above ``atol`` raises ``ParityError``.
    """
    model = model.eval()
    patch_size = int(model.patch_size)
    if size_hw[0] % patch_size or size_hw[1] % patch_size:
        raise ValueError(f"size_hw {tuple(size_hw)} must be multiples of patch size {patch_size}")

    wrapper = _GraphForward(model).eval()
    example = _parity_input(size_hw)
    grid_h = torch.export.Dim("grid_h", min=1, max=MAX_TOKEN_GRID)
    grid_w = torch.export.Dim("grid_w", min=1, max=MAX_TOKEN_GRID)
    # Non-strict tracing: transformers' hidden-state capture rebinds module.forward, which the
    # strict (dynamo) tracer rejects.
    with torch.no_grad():
        program = torch.export.export(
            wrapper,
            (example,),
            dynamic_shapes={"x": {2: patch_size * grid_h, 3: patch_size * grid_w}},
            strict=False,
        )
        eager = model(example)
        class_logits, concept_probs = program.module()(example)

    max_diff = max(
        (class_logits - eager.logits).abs().max().item(),
        (concept_probs - eager.hidden_states).abs().max().item(),
    )
    if not max_diff <= atol:
        raise ParityError(
            f"Exported graph diverges from the eager model: max|diff|={max_diff:.3e} "
            f"exceeds atol={atol:.3e} at size {tuple(size_hw)}. Refusing to write it."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, out_path)
    return out_path
