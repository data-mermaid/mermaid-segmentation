"""The export entrypoint writes a bundle directory from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.export as export_script
from mermaidseg.inference.bundle import MODEL_JSON, WEIGHTS
from tests.mermaidseg.conftest import deployable_lineage, repo_with_origin, seeded_s3
from tests.mermaidseg.inference import _tiny


def test_main_writes_a_bundle_with_the_sidecar_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    repo, pushed = repo_with_origin(tmp_path)
    monkeypatch.chdir(repo)
    checkpoint, _ = _tiny.write_run(tmp_path / "run", checkpoint_lineage=deployable_lineage(pushed))
    out = tmp_path / "bundle"
    export_script.main(
        ["--checkpoint", str(checkpoint), "--output-dir", str(out)], s3_client=seeded_s3()
    )
    assert sorted(p.name for p in out.iterdir()) == [MODEL_JSON, WEIGHTS]
    assert capsys.readouterr().out.strip() == str(out)
