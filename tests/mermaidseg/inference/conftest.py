"""Shared fixtures: a tiny trained run and the bundle exported from it."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mermaidseg.inference.bundle import export_bundle
from tests.mermaidseg.inference import _tiny


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, torch.nn.Module]:
    return _tiny.write_run(tmp_path_factory.mktemp("run"))


@pytest.fixture(scope="module")
def bundle_dir(run: tuple[Path, torch.nn.Module], tmp_path_factory: pytest.TempPathFactory) -> Path:
    return export_bundle(run[0], tmp_path_factory.mktemp("out") / "bundle")
