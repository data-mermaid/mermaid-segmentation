"""Module-scoped tiny-model fixtures for the inference package tests."""

from __future__ import annotations

import pytest
import torch

from tests.inference._tiny import (
    RUN_NAME,
    build_trained_lora_model,
    save_tiny_encoder,
    tiny_model_cfg,
    write_tiny_sidecar,
)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A run folder holding a tiny encoder config, the LoRA model, and its sidecar inputs."""
    root = tmp_path_factory.mktemp("run")
    encoder_dir = save_tiny_encoder(root / "encoder")
    model_cfg = tiny_model_cfg(encoder_dir)
    model = build_trained_lora_model(model_cfg)
    return {"root": root, "model_cfg": model_cfg, "model": model}


@pytest.fixture(scope="module")
def sidecar_path(trained_run):
    return write_tiny_sidecar(trained_run["root"] / "checkpoints", trained_run["model_cfg"])


@pytest.fixture(scope="module")
def checkpoint_path(trained_run):
    from mermaidseg.io import ConfigDict

    path = trained_run["root"] / "checkpoints" / "model_epoch3"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": trained_run["model"].state_dict(),
            "config": ConfigDict({"run_name": RUN_NAME, "model": trained_run["model_cfg"]}),
            "epoch": 3,
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def graph_bundle(trained_run, checkpoint_path, sidecar_path):
    from mermaidseg.inference import export_bundle

    return export_bundle(
        checkpoint_path, sidecar_path, trained_run["root"] / "graph_bundle", graph=True
    )
