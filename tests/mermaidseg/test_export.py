"""The sidecar ``model.json`` a run writes at start, and the bundle exported from a run."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from botocore.exceptions import ClientError
from safetensors.torch import load_file, save_file

from mermaidseg.export import _untied_cpu_state, write_sidecar
from mermaidseg.inference.bundle import MODEL_JSON, WEIGHTS
from tests.mermaidseg.conftest import (
    DS_A,
    DS_B,
    MANIFESTS,
    TAX,
    FakeS3,
    commit,
    deployable_lineage,
    export_ckpt,
    git,
    manifest_key,
    repo_with_origin,
    seeded_s3,
    sha,
)
from tests.mermaidseg.inference import _tiny


@pytest.fixture(scope="module")
def run(deployable: tuple[Path, Path, str]) -> tuple[Path, Path]:
    """The exportable run as (checkpoint, repo), shadowing the shared (checkpoint, model) fixture."""
    return deployable[0], deployable[1]


def _sidecar_kwargs(**overrides):
    kwargs = {
        "run_name": "run-a",
        "model_class": "ConceptBottleneckDPTLoRADINOv3",
        "model_kwargs": {
            "encoder_name": "enc",
            "input_size": (56, 56),
            "num_classes": 3,
            "concept_value2id": {"order": {"not_given": [0, 0]}, "bleached": {False: 1}},
        },
        "encoder_config": {"hidden_size": 16},
        "id2label": {0: "ignore", 1: "coral"},
        "concept_id2name": {0: "a", 1: "b"},
        "input_size": (56, 56),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.25, 0.25, 0.25),
        "label_space": {"classes.csv": "aa", "concepts.csv": "bb"},
        "lineage": {"seed": 7},
    }
    return {**kwargs, **overrides}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_sidecar_records_encoder_name_outside_kwargs_without_mutating_the_caller(tmp_path: Path):
    kwargs = _sidecar_kwargs()
    before = copy.deepcopy(kwargs["model_kwargs"])

    architecture = _read(write_sidecar(tmp_path, **kwargs))["architecture"]

    assert architecture["encoder_name"] == "enc"
    assert "encoder_name" not in architecture["kwargs"]
    assert kwargs["model_kwargs"] == before


def test_resume_with_the_same_contract_refreshes_the_lineage(tmp_path: Path):
    write_sidecar(tmp_path, **_sidecar_kwargs())
    first = _read(tmp_path / MODEL_JSON)
    write_sidecar(tmp_path, **_sidecar_kwargs(lineage={"seed": 7, "git_sha": "new"}))
    second = _read(tmp_path / MODEL_JSON)
    assert second["lineage"] == {"seed": 7, "git_sha": "new"}
    assert {k: v for k, v in second.items() if k != "lineage"} == {
        k: v for k, v in first.items() if k != "lineage"
    }


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"id2label": {0: "ignore", 1: "sand"}}, "id2label"),
        ({"model_class": "Other"}, "architecture"),
    ],
)
def test_resume_with_a_different_contract_names_the_differing_key(tmp_path, override, key):
    write_sidecar(tmp_path, **_sidecar_kwargs())
    before = (tmp_path / MODEL_JSON).read_text()
    with pytest.raises(ValueError, match=key):
        write_sidecar(tmp_path, **_sidecar_kwargs(**override))
    assert (tmp_path / MODEL_JSON).read_text() == before


def test_bundle_directory_holds_the_sidecar_weights_and_label_space(deployable, tmp_path):
    checkpoint, repo, _ = deployable
    run_dir = tmp_path / "run"
    shutil.copytree(checkpoint.parent, run_dir)
    run_space = run_dir / "label_space"
    (run_space / "extra.csv").write_text("unlisted\n")
    bundle_dir = export_ckpt(run_dir / checkpoint.name, tmp_path / "bundle", repo_dir=repo)
    assert sorted(p.name for p in bundle_dir.iterdir()) == ["label_space", MODEL_JSON, WEIGHTS]
    assert sorted(p.name for p in (bundle_dir / "label_space").iterdir()) == [
        "classes.csv",
        "concepts.csv",
    ]
    assert [p.name for p in bundle_dir.parent.iterdir()] == [bundle_dir.name, "run"] or sorted(
        p.name for p in bundle_dir.parent.iterdir()
    ) == ["bundle", "run"]
    for name in ("classes.csv", "concepts.csv"):
        assert (bundle_dir / "label_space" / name).read_bytes() == (run_space / name).read_bytes()


def test_exported_metadata_extends_the_sidecar_with_weights_and_checkpoint(deployable, bundle_dir):
    sidecar = json.loads((deployable[0].parent / MODEL_JSON).read_text())
    exported = json.loads((bundle_dir / MODEL_JSON).read_text())
    assert exported == {
        **sidecar,
        "lineage": deployable_lineage(deployable[2]),
        "weights": "weights.safetensors",
        "checkpoint": {"epoch": 3, "global_step": 42},
        "weights_sha256": hashlib.sha256((bundle_dir / WEIGHTS).read_bytes()).hexdigest(),
    }


def test_exported_lineage_is_the_checkpoints_not_the_sidecars(deployable, tmp_path):
    checkpoint, repo, pushed = deployable
    lineage = deployable_lineage(pushed)
    export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo)
    exported = json.loads((tmp_path / "out" / MODEL_JSON).read_text())
    sidecar = json.loads((checkpoint.parent / MODEL_JSON).read_text())
    assert sidecar["lineage"]["git_sha"] == "abc123"
    assert exported["lineage"] == lineage


def test_export_refuses_a_dirty_checkpoint_after_the_sidecar_was_refreshed_clean(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run", git_dirty=True)
    sidecar_path = checkpoint.parent / MODEL_JSON
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["lineage"]["git_dirty"] = False
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="clean tree"):
        export_ckpt(checkpoint, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_of_a_checkpoint_without_lineage_raises_naming_its_path(tmp_path):
    checkpoint, _ = _tiny.write_run(tmp_path / "run")
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del contents["lineage"]
    torch.save(contents, checkpoint)
    with pytest.raises(RuntimeError, match="lineage") as excinfo:
        export_ckpt(checkpoint, tmp_path / "out")
    assert str(checkpoint) in str(excinfo.value)


def test_export_refuses_a_non_empty_output_directory(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        export_ckpt(run[0], out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_export_accepts_an_existing_empty_output_directory(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    export_ckpt(run[0], out, repo_dir=run[1])
    assert sorted(p.name for p in out.iterdir()) == sorted(["label_space", MODEL_JSON, WEIGHTS])


@pytest.mark.parametrize("key", ["config"])
def test_export_of_a_checkpoint_missing_a_required_key_names_the_key_and_its_path(tmp_path, key):
    checkpoint, _ = _tiny.write_run(tmp_path / "run")
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del contents[key]
    torch.save(contents, checkpoint)
    with pytest.raises(RuntimeError, match=key) as raised:
        export_ckpt(checkpoint, tmp_path / "out")
    assert str(checkpoint) in str(raised.value)
    assert not (tmp_path / "out").exists()


def test_export_without_a_sidecar_raises_naming_its_path(run, tmp_path):
    lone = tmp_path / "lone"
    lone.mkdir()
    checkpoint = lone / "model_step1"
    checkpoint.write_bytes(run[0].read_bytes())
    with pytest.raises(FileNotFoundError, match=str(lone / MODEL_JSON)):
        export_ckpt(checkpoint, tmp_path / "out")


def test_export_requires_the_output_parent_to_exist_and_creates_nothing(run, tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        export_ckpt(run[0], tmp_path / "missing" / "out")
    assert list(tmp_path.iterdir()) == []


def test_untied_state_round_trips_through_save_file(tmp_path):
    shared = torch.arange(6, dtype=torch.float32)
    untied = _untied_cpu_state({"a": shared, "b": shared[:3]})
    save_file(untied, tmp_path / "w.safetensors")
    loaded = load_file(tmp_path / "w.safetensors")
    torch.testing.assert_close(loaded["a"], shared)
    torch.testing.assert_close(loaded["b"], shared[:3])


def test_export_refuses_a_dataset_manifest_missing_from_the_bucket_naming_dataset_and_version(
    deployable, tmp_path
):
    checkpoint, repo, _ = deployable
    s3 = seeded_s3(without=DS_B)
    with pytest.raises(ValueError, match="ds-b") as raised:
        export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo, s3_client=s3)
    assert "20261121-bbbbbbb" in str(raised.value)
    assert not (tmp_path / "out").exists()


def test_export_refuses_a_missing_taxonomy_manifest(deployable, tmp_path):
    checkpoint, repo, _ = deployable
    s3 = seeded_s3(without=TAX)
    with pytest.raises(ValueError, match="mermaid-benthic") as raised:
        export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo, s3_client=s3)
    assert "20261119-ccccccc" in str(raised.value)
    assert not (tmp_path / "out").exists()


def test_export_refuses_a_manifest_whose_sha_differs_naming_both_hashes(deployable, tmp_path):
    checkpoint, repo, _ = deployable
    s3 = seeded_s3()
    s3.objects[manifest_key(DS_A)] = b"tampered"
    with pytest.raises(ValueError, match="ds-a") as raised:
        export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo, s3_client=s3)
    assert sha(MANIFESTS[DS_A]) in str(raised.value)
    assert sha(b"tampered") in str(raised.value)
    assert not (tmp_path / "out").exists()


def test_export_refuses_a_commit_only_on_a_local_branch_telling_the_user_to_fetch(tmp_path):
    repo, _ = repo_with_origin(tmp_path)
    git(repo, "checkout", "-b", "local-only")
    local_sha = commit(repo, "local.txt")
    lineage = deployable_lineage(local_sha)
    checkpoint, _ = _tiny.write_run(tmp_path / "run", checkpoint_lineage=lineage)
    s3 = seeded_s3()
    with pytest.raises(ValueError, match="not on any remote branch") as raised:
        export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo, s3_client=s3)
    assert local_sha in str(raised.value)
    assert s3.requested == []
    assert not (tmp_path / "out").exists()


def test_export_accepts_a_commit_on_a_remote_other_than_origin(tmp_path):
    repo, _ = repo_with_origin(tmp_path)
    other = tmp_path / "other.git"
    other.mkdir()
    git(other, "init", "--bare", "--initial-branch=main")
    git(repo, "remote", "add", "other", str(other))
    git(repo, "checkout", "-b", "elsewhere")
    sha_on_other = commit(repo, "elsewhere.txt")
    git(repo, "push", "other", "elsewhere")
    git(repo, "checkout", "main")
    git(repo, "branch", "-D", "elsewhere")
    checkpoint, _ = _tiny.write_run(
        tmp_path / "run", checkpoint_lineage=deployable_lineage(sha_on_other)
    )
    export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo)
    assert (tmp_path / "out" / MODEL_JSON).is_file()


def test_export_refuses_a_commit_the_repository_does_not_know(deployable, tmp_path):
    _, repo, _ = deployable
    unknown = "0123456789abcdef0123456789abcdef01234567"
    lineage = deployable_lineage(unknown)
    checkpoint, _ = _tiny.write_run(tmp_path / "run", checkpoint_lineage=lineage)
    with pytest.raises(ValueError, match="not on any remote branch"):
        export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            ClientError(
                {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
                "GetObject",
            ),
            id="access-denied",
        ),
    ],
)
def test_export_propagates_a_non_missing_key_s3_error_unchanged(deployable, tmp_path, error):
    checkpoint, repo, _ = deployable
    with pytest.raises(ClientError):
        export_ckpt(checkpoint, tmp_path / "out", repo_dir=repo, s3_client=FakeS3({}, error=error))
    assert not (tmp_path / "out").exists()
