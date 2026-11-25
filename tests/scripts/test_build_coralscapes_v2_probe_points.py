"""Unit test for the CoralscapesV2 probe-point builder entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.build_coralscapes_v2_probe_points as builder_script
from tests.mermaidseg.data.conftest import ImageSpec, write_version


def test_main_opens_the_pinned_version_and_forwards_the_cli_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="dense",
        images=[ImageSpec("site1_a", group="train", mask=np.ones((8, 8), np.uint8))],
        labels={1: "acanthaster planci"},
        name="coralscapes_v2",
    )
    data_config = tmp_path / "data_config.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "datasets": {
                        "coralscapes_v2": {
                            "version": pin.version,
                            "manifest_sha256": pin.manifest_sha256,
                            "train": {"groups": ["train"]},
                        }
                    }
                }
            }
        )
    )
    captured: dict = {}
    monkeypatch.setattr(builder_script, "build_probe_points", lambda **kw: captured.update(kw))

    builder_script.main(
        [
            "--cache-dir",
            str(cache_dir),
            "--data-config",
            str(data_config),
            "--output",
            str(tmp_path / "out.json"),
            "--per-class",
            "50",
            "--min-component-size",
            "7",
            "--exclusion-radius",
            "9",
            "--label-space",
            "legacy39",
        ]
    )

    version = captured.pop("version")
    assert (version.manifest.dataset, version.manifest.version) == ("coralscapes_v2", pin.version)
    assert captured == {
        "output": tmp_path / "out.json",
        "per_class": 50,
        "min_component_size": 7,
        "exclusion_radius": 9,
        "label_space": "legacy39",
    }
