"""The export entrypoint writes a bundle directory from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.export as export_script
from mermaidseg.inference.bundle import MODEL_JSON, WEIGHTS
from tests.mermaidseg.inference import _tiny


def test_main_writes_a_bundle_with_the_sidecar_and_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    checkpoint, _ = _tiny.write_run(tmp_path / "run")
    out = tmp_path / "bundle"
    export_script.main(["--checkpoint", str(checkpoint), "--output-dir", str(out)])
    assert sorted(p.name for p in out.iterdir()) == [MODEL_JSON, WEIGHTS]
    assert capsys.readouterr().out.strip() == str(out)


def test_main_refuses_a_dirty_run(tmp_path: Path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run", git_dirty=True)
    out = tmp_path / "bundle"
    with pytest.raises(ValueError, match="clean tree"):
        export_script.main(["--checkpoint", str(checkpoint), "--output-dir", str(out)])
    assert not out.exists()
