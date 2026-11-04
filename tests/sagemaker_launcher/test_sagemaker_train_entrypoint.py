"""Tests for the in-container training entrypoint
(scripts/sagemaker_train_entrypoint.py).

Regression guard: ``_find_run_yaml`` used to sniff for a run YAML by checking whether
the literal substring ``"job:"`` appeared in the file's first 200 characters. Any run
YAML with a comment header longer than that (e.g. the taxonomical-loss-ablation configs)
pushed the ``job:`` key past the sniff window, so the container failed at start with "No
run YAML with a job: block" even though the file was a perfectly valid run YAML. The fix
parses each YAML and checks for a top-level ``job`` key instead.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sagemaker_train_entrypoint as entrypoint  # type: ignore  # noqa: E402

RUN_YAML_BODY = """
job:
  name_prefix: test-run
  image: mermaid-segmentation-jobs:training-latest
  entrypoint: scripts/sagemaker_train_entrypoint.py
  instance_type: ml.g5.2xlarge
  volume_gb: 200
  max_runtime_hours: 48

config:
  config_data: configs/data_config_dedup.yaml
  config_model: configs/model_config_cbm_dpt_lora_vitl_256.yaml
  config_training: configs/training_config_cbm_vitsize.yaml
"""

LONG_COMMENT_HEADER = "\n".join(f"# {'x' * 76}" for _ in range(6))  # > 200 chars


def test_finds_run_yaml_behind_a_long_comment_header(tmp_path, monkeypatch):
    """A comment header longer than 200 chars must not hide the job: block (the bug)."""
    (tmp_path / "run.yaml").write_text(LONG_COMMENT_HEADER + "\n" + RUN_YAML_BODY)
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    found = entrypoint._find_run_yaml()

    assert found == tmp_path / "run.yaml"


def test_ignores_non_run_yaml_in_the_same_directory(tmp_path, monkeypatch):
    """A sibling YAML without a job: key (e.g. a split config) is not mistaken for the
    run YAML."""
    (tmp_path / "run.yaml").write_text(RUN_YAML_BODY)
    (tmp_path / "data_config.yaml").write_text("data:\n  default:\n    train: {}\n")
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    found = entrypoint._find_run_yaml()

    assert found == tmp_path / "run.yaml"


def test_raises_when_no_run_yaml_present(tmp_path, monkeypatch):
    (tmp_path / "data_config.yaml").write_text("data:\n  default: {}\n")
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    with pytest.raises(SystemExit, match="No run YAML with a job: block"):
        entrypoint._find_run_yaml()


def test_raises_on_multiple_run_yamls(tmp_path, monkeypatch):
    (tmp_path / "run.yaml").write_text(RUN_YAML_BODY)
    (tmp_path / "run2.yaml").write_text(RUN_YAML_BODY)
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    with pytest.raises(SystemExit, match="Multiple candidate run YAMLs"):
        entrypoint._find_run_yaml()


TRAINER = "scripts/train.py"


def test_env_maps_cache_channel_and_model_dir():
    env = entrypoint.build_trainer_env(
        {"SM_CHANNEL_CACHE": "/opt/ml/input/data/cache", "SM_MODEL_DIR": "/opt/ml/model"}
    )

    assert env == {
        "MERMAIDSEG_LOCAL_CACHE_DIR": "/opt/ml/input/data/cache",
        "MERMAIDSEG_OUTPUT_DIR": "/opt/ml/model",
    }


def test_env_does_not_override_values_already_set():
    env = entrypoint.build_trainer_env(
        {
            "SM_CHANNEL_CACHE": "/c",
            "SM_MODEL_DIR": "/m",
            "MERMAIDSEG_LOCAL_CACHE_DIR": "/mine",
            "MERMAIDSEG_OUTPUT_DIR": "/out",
            "MERMAIDSEG_RUN_ID": "job",
            "WANDB_RUN_ID": "mine",
            "WANDB_RESUME": "must",
        }
    )

    assert env == {}


def test_argv_runs_the_cbm_trainer_with_config_flags_and_overrides():
    seg = {
        "config_model": "configs/m.yaml",
        "config_training": "configs/t.yaml",
        "config_data": "configs/d.yaml",
        "overrides": {
            "run-name": "cbm_vitl",
            "per-pixel-loss-weight": 0.5,
            "preempt-checkpoint-every-steps": 1000,
        },
    }

    argv = entrypoint.build_trainer_argv(seg, python="py")

    assert argv == [
        "py",
        "-u",
        TRAINER,
        "--model-config",
        "configs/m.yaml",
        "--training-config",
        "configs/t.yaml",
        "--data-config",
        "configs/d.yaml",
        "--run-name",
        "cbm_vitl",
        "--per-pixel-loss-weight",
        "0.5",
        "--preempt-checkpoint-every-steps",
        "1000",
    ]


def test_argv_emits_only_the_config_flags_given_and_handles_bools():
    argv = entrypoint.build_trainer_argv({"overrides": {"flag-on": True, "flag-off": False}}, "py")

    assert argv == ["py", "-u", TRAINER, "--flag-on"]


def test_argv_rejects_unknown_config_keys():
    with pytest.raises(SystemExit, match="config_logger"):
        entrypoint.build_trainer_argv({"config_logger": "x"}, "py")


@pytest.mark.parametrize("key", ["model-config", "training-config", "data-config"])
def test_argv_rejects_an_override_that_shadows_a_config_flag(key):
    with pytest.raises(SystemExit, match=key):
        entrypoint.build_trainer_argv({"overrides": {key: "x.yaml"}}, "py")


def test_missing_config_files_reports_paths_that_do_not_exist(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "configs").mkdir()
    (repo / "configs" / "t.yaml").write_text("x: 1\n")
    seg = {"config_training": "configs/t.yaml", "config_model": "configs/nope.yaml"}

    missing = entrypoint.missing_config_files(seg, repo)

    assert missing == ["configs/nope.yaml"]


class FakeSecretsClient:
    def __init__(self, secrets):
        self.secrets = secrets
        self.requested = []

    def get_secret_value(self, SecretId):
        self.requested.append(SecretId)
        return {"SecretString": self.secrets[SecretId]}


def test_resolve_secrets_exports_the_real_variable_from_each_secret_id():
    environ = {"WANDB_API_KEY_SECRET_ID": "wb-secret", "HF_TOKEN_SECRET_ID": "hf-secret"}
    client = FakeSecretsClient({"wb-secret": "wandb-value", "hf-secret": "hf-value"})

    entrypoint.resolve_secrets(environ, client)

    assert environ["WANDB_API_KEY"] == "wandb-value"
    assert environ["HF_TOKEN"] == "hf-value"


def test_resolve_secrets_never_replaces_a_variable_already_set():
    environ = {"HF_TOKEN_SECRET_ID": "hf-secret", "HF_TOKEN": "explicit"}
    client = FakeSecretsClient({"hf-secret": "from-secret"})

    entrypoint.resolve_secrets(environ, client)

    assert environ["HF_TOKEN"] == "explicit"


def test_resolve_secrets_without_ids_makes_no_call():
    client = FakeSecretsClient({})
    entrypoint.resolve_secrets({}, client)
    assert client.requested == []


def test_resolve_secrets_lets_a_fetch_failure_raise():
    class Failing:
        def get_secret_value(self, SecretId):
            raise RuntimeError("AccessDenied")

    with pytest.raises(RuntimeError, match="AccessDenied"):
        entrypoint.resolve_secrets({"HF_TOKEN_SECRET_ID": "x"}, Failing())


def test_trainer_env_puts_the_cache_on_the_ml_volume_without_a_channel():
    env = entrypoint.build_trainer_env({})
    assert env["MERMAIDSEG_LOCAL_CACHE_DIR"] == "/opt/ml/cache"


def test_trainer_env_outputs_to_the_checkpoint_dir_when_checkpointing():
    env = entrypoint.build_trainer_env(
        {"SM_MODEL_DIR": "/opt/ml/model"}, checkpoint_dir=Path("/opt/ml/checkpoints")
    )
    assert env["MERMAIDSEG_OUTPUT_DIR"] == "/opt/ml/checkpoints"


def _touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x")


def test_latest_checkpoint_picks_the_highest_step_across_kinds(tmp_path):
    # epoch1 -> (1 + 1) * 1000 = 2000 steps, beating step1500 and preempt1999
    _touch(tmp_path, "model_step1500", "model_preempt1999", "model_epoch1", "model_epoch0")
    assert entrypoint.latest_checkpoint(tmp_path, 1000) == tmp_path / "model_epoch1"


def test_latest_checkpoint_ignores_partial_and_unrelated_files(tmp_path):
    _touch(tmp_path, "model_step100", "model_step900.partial", "model.json", "model_stepX")
    assert entrypoint.latest_checkpoint(tmp_path, 1000) == tmp_path / "model_step100"


def test_latest_checkpoint_keeps_the_first_of_equal_steps_preempt_first(tmp_path):
    _touch(tmp_path, "model_step500", "model_preempt500")
    assert entrypoint.latest_checkpoint(tmp_path, 1000) == tmp_path / "model_preempt500"


def test_latest_checkpoint_is_none_for_a_missing_or_empty_dir(tmp_path):
    assert entrypoint.latest_checkpoint(tmp_path / "absent", 1000) is None
    assert entrypoint.latest_checkpoint(tmp_path, 1000) is None


def test_training_complete_needs_a_final_step_file_that_finished(tmp_path):
    _touch(tmp_path, "model_step1000")
    done = lambda path: {"global_step": 1000, "epoch_complete": True}  # noqa: E731
    midway = lambda path: {"global_step": 1000, "epoch_complete": False}  # noqa: E731

    assert entrypoint.training_complete(tmp_path, 1000, 1000, load=done)
    assert not entrypoint.training_complete(tmp_path, 1000, 1000, load=midway)


def test_training_complete_accepts_the_last_epoch_file(tmp_path):
    _touch(tmp_path, "model_epoch0")
    done = lambda path: {"global_step": 1000, "epoch_complete": True}  # noqa: E731
    assert entrypoint.training_complete(tmp_path, 1000, 1000, load=done)


def test_training_complete_is_false_when_a_file_cannot_be_read(tmp_path):
    _touch(tmp_path, "model_step1000")

    def broken(path):
        raise EOFError("truncated")

    assert not entrypoint.training_complete(tmp_path, 1000, 1000, load=broken)


def _write_training_config(tmp_path, epochs=1, iters=1000):
    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"training:\n  epochs: {epochs}\n  iterations_per_train_epoch: {iters}\n")
    return cfg


def test_resume_args_pass_the_latest_checkpoint_of_the_named_run(tmp_path):
    cfg = _write_training_config(tmp_path, iters=1000)
    ckpt_root = tmp_path / "ckpt"
    _touch(ckpt_root / "model_checkpoints" / "myrun", "model_step300", "model_step700")
    seg = {"config_training": str(cfg), "overrides": {"run-name": "myrun"}}
    args = entrypoint.resume_args(seg, ckpt_root, repo_dir=tmp_path, load=_loadable)

    assert args == [
        "--checkpoint",
        str(ckpt_root / "model_checkpoints" / "myrun" / "model_step700"),
    ]


def test_resume_args_are_empty_with_no_checkpoint(tmp_path):
    cfg = _write_training_config(tmp_path)
    seg = {"config_training": str(cfg), "overrides": {"run-name": "myrun"}}
    assert entrypoint.resume_args(seg, tmp_path / "ckpt", repo_dir=tmp_path) == []


def test_resume_args_are_none_when_training_already_finished(tmp_path):
    cfg = _write_training_config(tmp_path, epochs=1, iters=1000)
    ckpt_root = tmp_path / "ckpt"
    _touch(ckpt_root / "model_checkpoints" / "myrun", "model_step1000")
    seg = {"config_training": str(cfg), "overrides": {"run-name": "myrun"}}
    load = lambda path: {"global_step": 1000, "epoch_complete": True}  # noqa: E731

    assert entrypoint.resume_args(seg, ckpt_root, repo_dir=tmp_path, load=load) is None


def _loadable(path):
    """A checkpoint the trainer can resume from (the keys resume.sh verify_ckpt requires)."""
    return {
        "global_step": 1,
        "epoch_complete": False,
        "model_state_dict": {"w": 1},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }


def test_resume_args_fall_back_past_an_unloadable_newest_checkpoint(tmp_path):
    cfg = _write_training_config(tmp_path, iters=1000)
    run_dir = tmp_path / "ckpt" / "model_checkpoints" / "myrun"
    _touch(run_dir, "model_step300", "model_step700")
    seg = {"config_training": str(cfg), "overrides": {"run-name": "myrun"}}

    def load(path):
        if path.name == "model_step700":
            raise EOFError("truncated")
        return _loadable(path)

    args = entrypoint.resume_args(seg, tmp_path / "ckpt", repo_dir=tmp_path, load=load)

    assert args == ["--checkpoint", str(run_dir / "model_step300")]


def test_resume_args_skip_a_checkpoint_without_scheduler_state(tmp_path):
    cfg = _write_training_config(tmp_path, iters=1000)
    run_dir = tmp_path / "ckpt" / "model_checkpoints" / "myrun"
    _touch(run_dir, "model_step300", "model_step700")
    seg = {"config_training": str(cfg), "overrides": {"run-name": "myrun"}}

    def load(path):
        ckpt = _loadable(path)
        if path.name == "model_step700":
            del ckpt["scheduler_state_dict"]
        return ckpt

    args = entrypoint.resume_args(seg, tmp_path / "ckpt", repo_dir=tmp_path, load=load)

    assert args == ["--checkpoint", str(run_dir / "model_step300")]


def test_resume_args_abort_when_checkpoints_exist_but_none_loads(tmp_path):
    cfg = _write_training_config(tmp_path, iters=1000)
    _touch(tmp_path / "ckpt" / "model_checkpoints" / "myrun", "model_step300", "model_step700")
    seg = {"config_training": str(cfg), "overrides": {"run-name": "myrun"}}

    def broken(path):
        raise EOFError("truncated")

    with pytest.raises(SystemExit, match="no resumable checkpoint") as exc_info:
        entrypoint.resume_args(seg, tmp_path / "ckpt", repo_dir=tmp_path, load=broken)

    assert "model_step300" in str(exc_info.value)
    assert "model_step700" in str(exc_info.value)


def test_resume_args_count_steps_with_the_epochs_override(tmp_path):
    # The config says 1 epoch of 1000; the override makes it 2, so step 1000 is not the end.
    cfg = _write_training_config(tmp_path, epochs=1, iters=1000)
    run_dir = tmp_path / "ckpt" / "model_checkpoints" / "myrun"
    _touch(run_dir, "model_step1000")
    seg = {"config_training": str(cfg), "overrides": {"run-name": "myrun", "epochs": 2}}

    def load(path):
        return {**_loadable(path), "global_step": 1000, "epoch_complete": True}

    args = entrypoint.resume_args(seg, tmp_path / "ckpt", repo_dir=tmp_path, load=load)

    assert args == ["--checkpoint", str(run_dir / "model_step1000")]


def test_resume_args_rank_epoch_files_with_the_iterations_override(tmp_path):
    # With 100 iterations per epoch, model_epoch1 is step 200: below model_step500.
    cfg = _write_training_config(tmp_path, epochs=10, iters=1000)
    run_dir = tmp_path / "ckpt" / "model_checkpoints" / "myrun"
    _touch(run_dir, "model_epoch1", "model_step500")
    seg = {
        "config_training": str(cfg),
        "overrides": {"run-name": "myrun", "iterations-per-train-epoch": 100},
    }

    args = entrypoint.resume_args(seg, tmp_path / "ckpt", repo_dir=tmp_path, load=_loadable)

    assert args == ["--checkpoint", str(run_dir / "model_step500")]


def test_check_output_dir_rejects_an_output_dir_override():
    with pytest.raises(SystemExit, match="output-dir"):
        entrypoint.check_output_dir_owned({"overrides": {"output-dir": "/x"}}, {})


def test_check_output_dir_rejects_mermaidseg_output_dir_in_job_env():
    with pytest.raises(SystemExit, match="MERMAIDSEG_OUTPUT_DIR"):
        entrypoint.check_output_dir_owned({}, {"MERMAIDSEG_OUTPUT_DIR": "/x"})


def test_check_output_dir_accepts_a_run_that_leaves_the_output_dir_alone():
    entrypoint.check_output_dir_owned({"overrides": {"run-name": "r"}}, {"OTHER": "1"})


def test_resume_argv_drops_init_overrides_and_appends_the_checkpoint():
    seg = {"overrides": {"run-name": "r", "init-weights": "/w", "init-checkpoint": "/c"}}

    argv = entrypoint.build_trainer_argv(seg, "py", resume=["--checkpoint", "/ckpt/model_step5"])

    assert argv == ["py", "-u", TRAINER, "--run-name", "r", "--checkpoint", "/ckpt/model_step5"]


def test_resume_argv_drops_a_checkpoint_override_for_the_resume_one():
    seg = {"overrides": {"checkpoint": "/old"}}

    argv = entrypoint.build_trainer_argv(seg, "py", resume=["--checkpoint", "/new"])

    assert argv == ["py", "-u", TRAINER, "--checkpoint", "/new"]


def test_fresh_argv_keeps_init_overrides():
    argv = entrypoint.build_trainer_argv({"overrides": {"init-weights": "/w"}}, "py", resume=[])
    assert argv == ["py", "-u", TRAINER, "--init-weights", "/w"]


def test_trainer_env_resumes_the_wandb_run_of_the_job():
    env = entrypoint.build_trainer_env({"MERMAIDSEG_RUN_ID": "cbm.vitl-20261102T000000Z-ab12"})
    assert env["WANDB_RUN_ID"] == "cbm-vitl-20261102T000000Z-ab12"
    assert env["WANDB_RESUME"] == "allow"


def _write(directory, name, content):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(content)


def test_sync_mirrors_the_newest_preempt_under_the_fixed_name_with_a_marker(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(local, "model_preempt1000", b"old")
    _write(local, "model_preempt2000", b"new")
    _write(local, "model.json", b"{}")

    entrypoint.sync_once(local, synced)

    assert sorted(p.name for p in synced.iterdir()) == ["model.json", "resume.ckpt", "resume.json"]
    assert (synced / "resume.ckpt").read_bytes() == b"new"
    assert json.loads((synced / "resume.json").read_text())["name"] == "model_preempt2000"


def test_sync_overwrites_the_same_objects_as_new_preempts_appear(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(local, "model_preempt1000", b"first")
    entrypoint.sync_once(local, synced)
    (local / "model_preempt1000").unlink()
    _write(local, "model_preempt2000", b"second")

    entrypoint.sync_once(local, synced)

    assert sorted(p.name for p in synced.iterdir()) == ["resume.ckpt", "resume.json"]
    assert (synced / "resume.ckpt").read_bytes() == b"second"
    assert json.loads((synced / "resume.json").read_text())["name"] == "model_preempt2000"


def test_sync_mirrors_an_interval_checkpoint_only_once_its_size_is_stable(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(local, "model_step500", b"part")

    first = entrypoint.sync_once(local, synced)
    assert not (synced / "model_step500").exists()

    _write(local, "model_step500", b"partial-more")
    second = entrypoint.sync_once(local, synced, first)
    assert not (synced / "model_step500").exists()

    entrypoint.sync_once(local, synced, second)
    assert (synced / "model_step500").read_bytes() == b"partial-more"


def test_sync_never_mirrors_partial_files(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(local, "model_preempt3000.partial", b"x")

    entrypoint.sync_once(local, synced, final=True)

    assert not synced.exists() or list(synced.iterdir()) == []


def test_final_sync_mirrors_interval_and_newest_preempt_under_real_names(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(local, "model_step1000", b"step")
    _write(local, "model_preempt1000", b"preempt")

    entrypoint.sync_once(local, synced, final=True)

    assert (synced / "model_step1000").read_bytes() == b"step"
    assert (synced / "model_preempt1000").read_bytes() == b"preempt"


def test_restore_puts_the_mirrored_checkpoints_back_under_their_names(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(synced, "resume.ckpt", b"preempt")
    _write(synced, "resume.json", json.dumps({"name": "model_preempt2000"}).encode())
    _write(synced, "model_step500", b"step")

    entrypoint.restore_checkpoints(synced, local)

    assert (local / "model_preempt2000").read_bytes() == b"preempt"
    assert (local / "model_step500").read_bytes() == b"step"
    assert entrypoint.latest_checkpoint(local, 1000) == local / "model_preempt2000"


def test_restore_then_sync_uploads_nothing_new(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(synced, "resume.ckpt", b"preempt")
    _write(synced, "resume.json", json.dumps({"name": "model_preempt2000"}).encode())
    entrypoint.restore_checkpoints(synced, local)
    before = (synced / "resume.ckpt").stat().st_mtime_ns

    entrypoint.sync_once(local, synced)

    assert (synced / "resume.ckpt").stat().st_mtime_ns == before
    assert sorted(p.name for p in synced.iterdir()) == ["resume.ckpt", "resume.json"]


def test_main_resumes_from_the_restored_mirror_and_mirrors_the_result(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = _write_training_config(tmp_path, epochs=1, iters=1000)
    config_dir, synced_root, local_root = tmp_path / "cfg", tmp_path / "ckpt", tmp_path / "local"
    config_dir.mkdir()
    (config_dir / "run.yaml").write_text(
        "job:\n  name_prefix: t\n"
        f"config:\n  config_training: {cfg}\n"
        "  overrides:\n    run-name: myrun\n    init-weights: /w\n"
    )
    synced = synced_root / "model_checkpoints" / "myrun"
    _write(synced, "resume.ckpt", b"preempt500")
    _write(synced, "resume.json", json.dumps({"name": "model_preempt500"}).encode())
    for name, value in {
        "CONFIG_DIR": config_dir,
        "CHECKPOINT_DIR": synced_root,
        "LOCAL_CHECKPOINT_DIR": local_root,
        "REPO_DIR": repo,
    }.items():
        monkeypatch.setattr(entrypoint, name, value)
    monkeypatch.setattr(entrypoint, "_torch_load", _loadable)
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    monkeypatch.delenv("MERMAIDSEG_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("MERMAIDSEG_LOCAL_CACHE_DIR", str(tmp_path / "cache"))
    calls = []

    def run_trainer(cmd):
        calls.append(cmd)
        _write(local_root / "model_checkpoints" / "myrun", "model_preempt1000", b"done")
        return 0

    monkeypatch.setattr(entrypoint.subprocess, "Popen", _fake_popen(run_trainer))

    with pytest.raises(SystemExit) as exit_info:
        entrypoint.main()

    assert exit_info.value.code == 0
    restored = local_root / "model_checkpoints" / "myrun" / "model_preempt500"
    assert "--init-weights" not in calls[0]
    assert calls[0][-2:] == ["--checkpoint", str(restored)]
    assert (synced / "resume.ckpt").read_bytes() == b"done"
    assert (synced / "model_preempt1000").read_bytes() == b"done"


def test_restore_links_preempts_but_copies_step_and_epoch_files(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(synced, "resume.ckpt", b"preempt")
    _write(synced, "resume.json", json.dumps({"name": "model_preempt2000"}).encode())
    _write(synced, "model_step500", b"step")
    _write(synced, "model_epoch0", b"epoch")

    entrypoint.restore_checkpoints(synced, local)

    # The trainer rewrites step/epoch files in place; that must never reach the synced copy.
    assert (local / "model_step500").stat().st_ino != (synced / "model_step500").stat().st_ino
    assert (local / "model_epoch0").stat().st_ino != (synced / "model_epoch0").stat().st_ino
    assert (local / "model_preempt2000").stat().st_ino == (synced / "resume.ckpt").stat().st_ino


def test_sync_hard_links_instead_of_a_second_copy(tmp_path):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(local, "model_preempt1000", b"preempt")
    _write(local, "model_step500", b"step")
    first = entrypoint.sync_once(local, synced)
    entrypoint.sync_once(local, synced, first)

    assert (synced / "resume.ckpt").stat().st_ino == (local / "model_preempt1000").stat().st_ino
    assert (synced / "model_step500").stat().st_ino == (local / "model_step500").stat().st_ino
    assert not (synced / entrypoint.MIRROR_TMP).exists()


def test_sync_copies_when_a_hard_link_is_refused(tmp_path, monkeypatch):
    local, synced = tmp_path / "local", tmp_path / "synced"
    _write(local, "model_preempt1000", b"preempt")

    def refuse(src, dst):
        raise OSError("cross-device")

    monkeypatch.setattr(entrypoint.os, "link", refuse)
    entrypoint.sync_once(local, synced)

    assert (synced / "resume.ckpt").read_bytes() == b"preempt"
    assert (synced / "resume.ckpt").stat().st_ino != (local / "model_preempt1000").stat().st_ino


class _FakeProc:
    def __init__(self, run):
        self._run, self.signals, self.returncode = run, [], None

    def wait(self):
        self.returncode = self._run(self)
        return self.returncode

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)


def _fake_popen(run_trainer):
    """A Popen stand-in whose wait() runs `run_trainer(cmd)` (or `run_trainer(proc)` below)."""

    def factory(cmd, cwd=None):
        return _FakeProc(lambda proc: run_trainer(cmd))

    return factory


def test_mirror_loop_syncs_then_forwards_sigterm_when_woken_by_the_handler():
    import threading

    wake, stop, terminated = threading.Event(), threading.Event(), threading.Event()
    order, forwarded = [], threading.Event()

    def forward():
        order.append("forward")
        forwarded.set()

    thread = threading.Thread(
        target=entrypoint._mirror_loop,
        args=(wake, stop, terminated, 3600, lambda: order.append("sync"), forward),
    )
    thread.start()
    terminated.set()
    wake.set()
    assert forwarded.wait(5)
    stop.set()
    wake.set()
    thread.join(5)

    assert order == ["sync", "forward"]


def test_mirror_loop_exits_on_stop_without_a_final_sync():
    import threading

    wake, stop, terminated = threading.Event(), threading.Event(), threading.Event()
    calls = []
    thread = threading.Thread(
        target=entrypoint._mirror_loop,
        args=(
            wake,
            stop,
            terminated,
            3600,
            lambda: calls.append("sync"),
            lambda: calls.append("f"),
        ),
    )
    thread.start()
    stop.set()
    wake.set()
    thread.join(5)

    assert not thread.is_alive()
    assert calls == []


def test_a_sigterm_syncs_before_it_is_forwarded_to_the_trainer(tmp_path, monkeypatch):
    import signal

    order = []
    monkeypatch.setattr(
        entrypoint, "sync_once", lambda *a, **kw: order.append(("sync", kw.get("final", False)))
    )

    def run(proc):
        os.kill(os.getpid(), signal.SIGTERM)
        assert _wait_for(lambda: proc.signals)
        order.append(("forwarded", proc.signals[0]))
        return -15

    monkeypatch.setattr(entrypoint.subprocess, "Popen", lambda cmd, cwd=None: _FakeProc(run))
    before = signal.getsignal(signal.SIGTERM)

    status = entrypoint.run_mirrored(["t"], tmp_path / "l", tmp_path / "s", cwd=tmp_path)

    assert status == -15
    assert order[:2] == [("sync", False), ("forwarded", signal.SIGTERM)]
    assert order[-1] == ("sync", True)
    assert signal.getsignal(signal.SIGTERM) is before


def _wait_for(predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_final_sync_after_a_failed_trainer_keeps_a_just_written_step_file(tmp_path, monkeypatch):
    local, synced = tmp_path / "local", tmp_path / "synced"

    def trainer(cmd):
        _write(local, "model_step500", b"fresh")
        return 1

    monkeypatch.setattr(entrypoint.subprocess, "Popen", _fake_popen(trainer))

    status = entrypoint.run_mirrored(["t"], local, synced, cwd=tmp_path)

    assert status == 1
    assert (synced / "model_step500").read_bytes() == b"fresh"
