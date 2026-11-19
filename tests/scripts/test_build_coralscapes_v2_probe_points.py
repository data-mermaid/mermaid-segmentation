"""Unit test for the CoralscapesV2 probe-point builder entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.build_coralscapes_v2_probe_points as builder_script


def test_main_forwards_cli_flags_to_build_probe_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    monkeypatch.setattr(builder_script, "build_probe_points", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_coralscapes_v2_probe_points.py",
            "--local-root",
            str(tmp_path),
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
        ],
    )
    builder_script.main()
    assert captured == {
        "local_root": tmp_path,
        "output": tmp_path / "out.json",
        "per_class": 50,
        "min_component_size": 7,
        "exclusion_radius": 9,
        "label_space": "legacy39",
    }
