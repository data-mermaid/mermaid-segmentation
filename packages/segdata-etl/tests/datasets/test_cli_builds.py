"""``segdata build <external>`` through ``cli.main``, so each subcommand reaches its own builder."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from s3_fake import FakeS3

from segdata_etl import cli
from segdata_spec import DATASET_BUCKET

from . import (
    test_benthos_yuval,
    test_catlin_seaview,
    test_moorea_labeled_corals,
    test_pacific_labeled_corals,
)
from .conftest import TAXONOMY_VERSION, make_repo, publish_fixture_taxonomy

_INGESTS: dict[str, Callable[[FakeS3], object]] = {
    "catlin_seaview": test_catlin_seaview._ingested,
    "moorea_labeled_corals": test_moorea_labeled_corals._ingested,
    "pacific_labeled_corals": test_pacific_labeled_corals._ingested,
    "benthos_yuval": test_benthos_yuval._ingested,
}


@pytest.mark.parametrize("dataset", list(_INGESTS))
def test_build_subcommand_stages_a_version_of_its_own_dataset(
    dataset: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = make_repo(tmp_path, monkeypatch)
    s3 = FakeS3()
    _INGESTS[dataset](s3)
    taxonomy_sha = publish_fixture_taxonomy(s3, repo)
    monkeypatch.setattr(cli, "make_s3_client", lambda **_kw: s3)
    stage = tmp_path / "stage"
    argv = ["build", dataset, "--stage-dir", str(stage), "--cache-dir", str(tmp_path / "cache")]
    argv += ["--taxonomy-version", TAXONOMY_VERSION, "--taxonomy-manifest-sha256", taxonomy_sha]

    assert cli.main([*argv, "--workers", "2"]) == 0

    version = dict(line.split(": ", 1) for line in capsys.readouterr().out.splitlines())["version"]
    assert [p.name for p in (stage / DATASET_BUCKET / "datasets").iterdir()] == [dataset]
    manifest = stage / DATASET_BUCKET / "datasets" / dataset / version / "manifest.json"
    assert json.loads(manifest.read_text())["dataset"] == dataset
