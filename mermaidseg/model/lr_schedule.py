"""Learning-rate schedules that are not stock PyTorch schedulers."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step


class ConstantThenLinearLR(LRScheduler):
    """Hold the base learning rate, then decay it linearly to ``end_factor``.

    The constructor's initial step leaves the learning rate at the base rate and
    sets ``last_epoch`` to 0. Each later :meth:`step` is one optimizer update.
    The first ``constant_iters`` of those stay at the base rate. The next
    ``decay_iters`` move linearly from the base rate to ``end_factor`` times the
    base rate, reaching ``end_factor`` on the last of them.

    Warmup is not part of this scheduler. :class:`~mermaidseg.model.meta.MetaModel`
    runs its own linear warmup and only starts stepping this scheduler once that
    warmup has finished, on the following update. For a 4k warmup, a plateau
    through update 275k, and a 25k decay that lands on zero at update 300k, use
    ``constant_iters=271_000`` and ``decay_iters=25_000``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        constant_iters: int,
        decay_iters: int,
        end_factor: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if constant_iters < 0:
            raise ValueError(f"constant_iters must be >= 0, got {constant_iters}")
        if decay_iters <= 0:
            raise ValueError(f"decay_iters must be > 0, got {decay_iters}")
        if end_factor < 0.0 or end_factor > 1.0:
            raise ValueError(f"end_factor must be in [0, 1], got {end_factor}")
        self.constant_iters = int(constant_iters)
        self.decay_iters = int(decay_iters)
        self.end_factor = float(end_factor)
        super().__init__(optimizer, last_epoch)

    def _factor(self) -> float:
        if self.last_epoch <= self.constant_iters:
            return 1.0
        progress = self.last_epoch - self.constant_iters
        if progress >= self.decay_iters:
            return self.end_factor
        return 1.0 + (self.end_factor - 1.0) * (progress / self.decay_iters)

    def get_lr(self) -> list[float]:
        """Closed form from ``base_lrs``, so a resume does not depend on the current rate."""
        _warn_get_lr_called_within_step(self)
        factor = self._factor()
        return [base_lr * factor for base_lr in self.base_lrs]
