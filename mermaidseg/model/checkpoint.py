"""Owns the on-disk checkpoint format so *save* and *load* agree in one place.

Two facts about our checkpoints must be understood identically wherever one is written or
read (training resume in :mod:`mermaidseg.model.meta`, the demo, tests):

1. **Frozen params are slimmed out.** A frozen backbone is byte-identical across every
   checkpoint of a run and is re-created when the model is constructed, so the ``Logger``
   drops those keys (``excludes_frozen_params=True``). Such a checkpoint must load with
   ``strict=False`` — the missing keys are exactly the (re-initialised) frozen params.
2. **PEFT (LoRA) wrapping rewrites keys.** ``peft.get_peft_model`` nests the encoder, so a
   checkpoint's keys can carry extra ``base_model``/``encoder`` nesting relative to a
   freshly-wrapped model. Keys are normalised before loading so they line up.

Loading never touches ``requires_grad``: the frozen/trainable split comes from how the model
is constructed (``freeze_encoder``), not from the state dict.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def frozen_param_names(model: torch.nn.Module) -> set[str]:
    """State-dict keys of parameters with ``requires_grad=False`` (e.g. a frozen
    backbone)."""
    return {name for name, param in model.named_parameters() if not param.requires_grad}


def slim_state_dict(
    model: torch.nn.Module, *, save_full: bool = False
) -> tuple[dict[str, Any], bool]:
    """Return ``(state_dict, excludes_frozen)`` for saving.

    Frozen parameters are dropped unless ``save_full`` is set. ``excludes_frozen``
    records whether anything was dropped, so the loader knows to use ``strict=False``.
    """
    # Copy tensors to CPU for a portable checkpoint without moving the model itself.
    full = {
        key: (value.cpu() if isinstance(value, torch.Tensor) else value)
        for key, value in model.state_dict().items()
    }
    if save_full:
        return full, False
    frozen = frozen_param_names(model)
    slim = {key: value for key, value in full.items() if key not in frozen}
    return slim, bool(frozen)


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile PEFT/backbone key nesting so a checkpoint loads into a freshly-built
    model.

    A no-op for keys that don't carry the extra nesting, so it is always safe to apply.
    """
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.model."):
            key = key.replace("encoder.model.", "encoder.", 1)
        if ".base_model.model.model." in key:
            key = key.replace(".base_model.model.model.", ".base_model.model.", 1)
        normalized[key] = value
    return normalized


def load_into(model: torch.nn.Module, checkpoint: Any) -> Any:
    """Load a saved ``checkpoint`` into a freshly-constructed ``model``.

    Accepts either a full checkpoint dict (with ``model_state_dict`` and optional
    ``excludes_frozen_params``) or a bare state dict. Slimmed checkpoints load
    ``strict=False`` (the missing keys are the re-initialised frozen backbone);
    bare/legacy state dicts load ``strict=True`` to still catch real mismatches. Keys
    are normalised for PEFT nesting first. Returns torch's ``_IncompatibleKeys``
    (``.missing_keys`` / ``.unexpected_keys``) so callers can assert that nothing
    *unexpected* was dropped.
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        strict = not checkpoint.get("excludes_frozen_params", False)
    else:
        state_dict = checkpoint
        strict = True
    return model.load_state_dict(normalize_state_dict_keys(state_dict), strict=strict)


def restore_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: Mapping[str, Any],
    *,
    scheduler: Any | None = None,
) -> int:
    """Restore training state from a full checkpoint for resume; return the epoch to
    resume FROM.

    Loads model weights (frozen-slim aware, via :func:`load_into`), then restores optimizer state
    and — when a scheduler is supplied and its state was saved — scheduler state, in place on the
    freshly-constructed instances so training continues with the correct momentum and LR schedule.
    Returns ``checkpoint["epoch"] + 1`` (the checkpoint records the last *completed* epoch).

    Minimal-resume by design: RNG, AMP ``GradScaler``, the LR-warmup counter, and early-stopping
    state are *not* restored (they are not persisted), so a resumed run continues correctly but is
    not bit-for-bit identical to an uninterrupted one. Requires a *full* checkpoint (the one logged
    under ``artifact_path="checkpoints"``); the lean ``best-model`` snapshot has no optimizer state
    and will raise ``KeyError`` here — which is intended, since it cannot resume training.
    """
    load_into(model, checkpoint)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    return int(checkpoint["epoch"]) + 1
