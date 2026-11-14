"""The published CoralscapesV2 95-class to Coralscapes 39-class map."""

from __future__ import annotations

import numpy as np

from mermaidseg.datasets.coralscapes.coralscapes_dataset import CORALSCAPES_ID2NAME
from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import (
    CORALSCAPES_V2_ID2NAME,
    CORALSCAPES_V2_TO_LEGACY_NAME,
    coralscapes_v2_to_legacy_lut,
)


def _legacy_id(name: str) -> int:
    return next(class_id for class_id, label in CORALSCAPES_ID2NAME.items() if label == name)


def _fine_id(name: str) -> int:
    return next(class_id for class_id, label in CORALSCAPES_V2_ID2NAME.items() if label == name)


def test_legacy_map_covers_every_fine_and_coarse_class():
    assert set(CORALSCAPES_V2_TO_LEGACY_NAME) == set(CORALSCAPES_V2_ID2NAME.values())
    assert set(CORALSCAPES_V2_TO_LEGACY_NAME.values()) == set(CORALSCAPES_ID2NAME.values())

    lut = coralscapes_v2_to_legacy_lut()
    assert lut.shape == (96,)
    assert lut.dtype == np.uint8
    assert int(lut[0]) == 0
    assert {int(value) for value in lut[1:]} == set(range(1, 40))
    for class_id, name in CORALSCAPES_V2_ID2NAME.items():
        assert int(lut[class_id]) == _legacy_id(CORALSCAPES_V2_TO_LEGACY_NAME[name])


def test_legacy_map_spot_checks():
    lut = coralscapes_v2_to_legacy_lut()
    assert CORALSCAPES_V2_TO_LEGACY_NAME["acropora alive"] == "acropora alive"
    assert int(lut[_fine_id("acropora alive")]) == _legacy_id("acropora alive")

    algae = [name for name in CORALSCAPES_V2_ID2NAME.values() if name.startswith("algae covered")]
    assert len(algae) == 8
    assert {CORALSCAPES_V2_TO_LEGACY_NAME[name] for name in algae} == {"algae covered substrate"}

    assert CORALSCAPES_V2_TO_LEGACY_NAME["water column"] == "background"
    assert CORALSCAPES_V2_TO_LEGACY_NAME["water surface"] == "background"
    assert int(lut[_fine_id("water column")]) == _legacy_id("background")

    assert CORALSCAPES_V2_TO_LEGACY_NAME["galaxea dead"] == "massive/meandering dead"
    assert CORALSCAPES_V2_TO_LEGACY_NAME["galaxea alive"] == "other coral alive"
    assert CORALSCAPES_V2_TO_LEGACY_NAME["galaxea bleached"] == "other coral bleached"

    # The dataset card's legacy column says "hard substrate".
    assert CORALSCAPES_V2_TO_LEGACY_NAME["hard substrate"] == "unknown hard substrate"
    assert CORALSCAPES_V2_TO_LEGACY_NAME["human-made structure"] == "unknown hard substrate"
