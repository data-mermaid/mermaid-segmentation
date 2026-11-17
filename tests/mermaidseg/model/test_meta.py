"""Run settings read from the training config by mermaidseg.model.meta."""

from __future__ import annotations

import pytest

import mermaidseg.model.meta as meta_module
from mermaidseg.io import ConfigDict


@pytest.mark.parametrize(("raw", "expected"), [(180, 180.0), (45, 45.0), (0, 0.0), (-1, 0.0)])
def test_stall_dump_seconds_reads_training_config(raw, expected):
    assert meta_module.stall_dump_seconds(ConfigDict({"stall_dump_sec": raw})) == expected
