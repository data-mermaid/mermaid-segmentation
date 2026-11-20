"""Class-id and palette construction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def build_classes_and_colors(
    names: Iterable[str], seed: int = 1337
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Return ``classes.json`` and ``colors.json`` content.

    Names are deduplicated and numbered alphabetically from 1; ``unlabeled`` is 0 and black.
    Colors are drawn in alphabetical order from a seeded generator, so they are reproducible.
    """
    ordered = sorted(set(names))
    classes = {"unlabeled": 0}
    classes.update({name: i for i, name in enumerate(ordered, start=1)})
    rng = np.random.default_rng(seed)
    colors: dict[str, list[int]] = {"unlabeled": [0, 0, 0]}
    for name in ordered:
        colors[name] = [int(x) for x in rng.integers(0, 256, size=3)]
    return classes, colors
