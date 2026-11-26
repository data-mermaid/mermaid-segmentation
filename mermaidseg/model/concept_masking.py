"""Per-source masking of concept targets (the tensor op; the keep table is built from the sidecar)."""

from __future__ import annotations

import torch


def apply_concept_source_mask(
    target_concepts: torch.Tensor,
    keep: torch.Tensor,
    source_rows: torch.Tensor,
) -> torch.Tensor:
    """Set masked channels of each sample's concept targets to 0 (not_given).

    Args:
        target_concepts: ``(B, C, H, W)`` concept targets.
        keep: ``(S + 1, C)`` bool keep table (see ``ConceptSourceMask``).
        source_rows: ``(B,)`` keep-table row of each sample.
    """
    keep_b = keep.to(target_concepts.device)[source_rows.to(target_concepts.device).long()]
    return target_concepts * keep_b.to(target_concepts.dtype)[:, :, None, None]
