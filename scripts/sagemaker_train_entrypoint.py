"""In-container entrypoint for a mermaid-segmentation TrainingJob.

Reads the run YAML from /opt/ml/input/data/config/, applies any top-level `env:` keys,
maps SageMaker's environment onto the trainer's, then runs from the repo root the same trainer the
Slurm jobs run (slurm/engaging/train_body.sh):

    python scripts/train.py <flags from the run YAML>

Secrets: `WANDB_API_KEY_SECRET_ID` / `HF_TOKEN_SECRET_ID` name Secrets Manager secrets; their
values are fetched here and exported as WANDB_API_KEY / HF_TOKEN, so the job's Environment never
carries the raw values.

Environment mapping:
  SM_CHANNEL_CACHE -> MERMAIDSEG_LOCAL_CACHE_DIR; without a `cache` channel the trainer's
                      write-through S3 mirror goes to /opt/ml/cache on the ML volume (size
                      `volume_gb` for the mirror)
  SM_MODEL_DIR     -> MERMAIDSEG_OUTPUT_DIR (checkpoints land in
                      <model dir>/model_checkpoints/<run-name>, which SageMaker uploads)
  MERMAIDSEG_RUN_ID -> WANDB_RUN_ID (sanitised) and WANDB_RESUME=allow, so a spot restart
                      continues the same wandb run, as slurm/engaging/train_body.sh does

Spot resume: when the launcher configures SageMaker checkpointing, /opt/ml/checkpoints exists.
SageMaker syncs that directory to S3 and never deletes there, so the trainer writes instead to
/opt/ml/local_checkpoints/model_checkpoints/<run-name> on the ML volume, and a background thread
mirrors into /opt/ml/checkpoints/model_checkpoints/<run-name> only what resume needs
(`sync_once`): the newest model_preempt* as `resume.ckpt` plus a `resume.json` marker naming it,
every finished model_step* / model_epoch* under its own name, and model.json. After the trainer exits
the newest preempt is also kept under its own name. After a restart the mirror is restored into
the local directory, the trainer resumes from the newest checkpoint that loads (as
slurm/engaging/resume.sh does), and exits without training when the run already finished. The
run's artifacts are then in the checkpoint S3 prefix, not model.tar.gz. The entrypoint owns the
output directory then: an `output-dir` override or MERMAIDSEG_OUTPUT_DIR in job.env is an error.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path

import yaml

log = logging.getLogger("seg_train_entrypoint")
CONFIG_DIR = Path("/opt/ml/input/data/config")
CHECKPOINT_DIR = Path("/opt/ml/checkpoints")
# On the ML volume but outside CHECKPOINT_DIR, so SageMaker syncs only what sync_once mirrors.
LOCAL_CHECKPOINT_DIR = Path("/opt/ml/local_checkpoints")
RESUME_NAME = "resume.ckpt"
RESUME_MARKER = "resume.json"
SIDECAR_NAME = "model.json"
# One fixed temporary name, so an interrupted copy never leaves a new object in S3.
MIRROR_TMP = ".mirror.tmp"
SYNC_INTERVAL_S = 60
# Outside /opt/ml/model and /opt/ml/output so SageMaker does not upload the mirror.
LOCAL_CACHE_FALLBACK = "/opt/ml/cache"
DEFAULT_RUN_NAME = "mermaid_base_run_dinov3_lora_dpt_all"
DEFAULT_TRAINING_CONFIG = "configs/training_config_cbm.yaml"

# Secret-id variable -> the variable the trainer reads.
SECRET_ENV = {
    "WANDB_API_KEY_SECRET_ID": "WANDB_API_KEY",
    "HF_TOKEN_SECRET_ID": "HF_TOKEN",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = REPO_ROOT
TRAINER_SCRIPT = "scripts/train.py"

# Override keys that start a run from a checkpoint; a resume replaces them with --checkpoint.
_INIT_OVERRIDES = ("checkpoint", "init-checkpoint", "init-weights")
# What resume.sh verify_ckpt requires before handing a checkpoint to the trainer.
_RESUME_KEYS = ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")

# Run-YAML `config:` keys that name config files (paths relative to the repo root) -> trainer flags.
_CONFIG_FLAGS = {
    "config_model": "--model-config",
    "config_training": "--training-config",
    "config_data": "--data-config",
}


def resolve_secrets(environ: MutableMapping[str, str], client) -> None:
    """Export each `*_SECRET_ID` secret's value as its real variable, keeping any already set.

    A failed fetch raises: a run that needs a key must not start without it.
    """
    for id_var, var in SECRET_ENV.items():
        secret_id = environ.get(id_var)
        if secret_id:
            value = client.get_secret_value(SecretId=secret_id)["SecretString"]
            environ.setdefault(var, value)


def build_trainer_env(
    environ: Mapping[str, str], checkpoint_dir: Path | None = None
) -> dict[str, str]:
    """Return the MERMAIDSEG_* variables to add, never replacing ones already set.

    `checkpoint_dir` replaces SM_MODEL_DIR as the trainer's output root.
    """
    mapping = {
        "MERMAIDSEG_LOCAL_CACHE_DIR": environ.get("SM_CHANNEL_CACHE") or LOCAL_CACHE_FALLBACK,
        "MERMAIDSEG_OUTPUT_DIR": str(checkpoint_dir)
        if checkpoint_dir
        else environ.get("SM_MODEL_DIR"),
    }
    run_id = environ.get("MERMAIDSEG_RUN_ID")
    if run_id:
        # wandb run ids allow only alphanumerics, '_' and '-'.
        mapping["WANDB_RUN_ID"] = re.sub(r"[^A-Za-z0-9_-]", "-", run_id)
        mapping["WANDB_RESUME"] = "allow"
    return {k: v for k, v in mapping.items() if v and k not in environ}


def check_output_dir_owned(seg: Mapping, job_env: Mapping[str, str]) -> None:
    """Reject settings that move the trainer's output away from where resume looks."""
    if "output-dir" in (seg.get("overrides") or {}):
        raise SystemExit(
            "`output-dir` in config.overrides is not allowed with SageMaker checkpointing: "
            "the entrypoint sets the trainer's output directory."
        )
    if "MERMAIDSEG_OUTPUT_DIR" in job_env:
        raise SystemExit(
            "MERMAIDSEG_OUTPUT_DIR in job.env is not allowed with SageMaker checkpointing: "
            "the entrypoint sets the trainer's output directory."
        )


def _checkpoint_step(name: str, epoch_iters: int) -> int | None:
    """Optimizer step a checkpoint file name stands for; None for other names."""
    m = re.fullmatch(r"model_(preempt|step)(\d+)", name)
    if m:
        return int(m.group(2))
    m = re.fullmatch(r"model_epoch(\d+)", name)
    return (int(m.group(1)) + 1) * epoch_iters if m else None


def checkpoint_candidates(directory: Path, epoch_iters: int) -> list[Path]:
    """Checkpoints newest first by optimizer step (resume.sh latest_checkpoint order).

    model_epochE counts as (E + 1) * epoch_iters; `.partial` files are skipped; on equal
    steps the first of preempt, step, epoch (then name order) wins.
    """
    if not directory.is_dir():
        return []
    ranked = []
    for kind, prefix in enumerate(("model_preempt", "model_step", "model_epoch")):
        for path in sorted(directory.glob(f"{prefix}*")):
            step = _checkpoint_step(path.name, epoch_iters) if path.is_file() else None
            if step is not None:
                ranked.append((-step, kind, path.name, path))
    return [path for *_, path in sorted(ranked)]


def latest_checkpoint(directory: Path, epoch_iters: int) -> Path | None:
    """Checkpoint with the highest optimizer step, or None."""
    candidates = checkpoint_candidates(directory, epoch_iters)
    return candidates[0] if candidates else None


def _resumable(path: Path, load: Callable[[Path], dict]) -> bool:
    """True when `path` restores model, optimizer and scheduler (resume.sh verify_ckpt)."""
    try:
        ckpt = load(path)
    except Exception as exc:  # noqa: BLE001 - a truncated file is skipped, not fatal
        log.warning("cannot resume from %s: %s", path, exc)
        return False
    missing = [key for key in _RESUME_KEYS if key not in ckpt]
    state = ckpt.get("model_state_dict")
    if missing or not isinstance(state, dict) or not state:
        log.warning("cannot resume from %s: missing %s or model weights", path, missing)
        return False
    return True


def _torch_load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def training_complete(
    directory: Path, total_steps: int, epoch_iters: int, load: Callable[[Path], dict] | None = None
) -> bool:
    """True when a finished final-step checkpoint is on disk (resume.sh training_complete)."""
    load = load or _torch_load

    def finished(path: Path) -> bool:
        try:
            ckpt = load(path)
        except Exception as exc:  # noqa: BLE001 - a truncated file is not "done"
            log.warning("could not read %s: %s", path, exc)
            return False
        return int(ckpt.get("global_step") or 0) >= total_steps and bool(
            ckpt.get("epoch_complete", True)
        )

    for name in (f"model_step{total_steps}", f"model_preempt{total_steps}"):
        if (directory / name).is_file() and finished(directory / name):
            return True
    epochs = {
        int(p.name.removeprefix("model_epoch")): p
        for p in directory.glob("model_epoch*")
        if p.is_file() and p.name.removeprefix("model_epoch").isdigit()
    }
    if epochs:
        index = max(epochs)
        return (index + 1) * epoch_iters >= total_steps and finished(epochs[index])
    return False


def resume_args(
    seg: Mapping,
    checkpoint_root: Path,
    repo_dir: Path = REPO_DIR,
    load: Callable[[Path], dict] | None = None,
) -> list[str] | None:
    """Trainer args that resume the run from `checkpoint_root`; None when it already finished.

    Candidates are tried newest first; one that does not load is skipped for the next. Exits
    when checkpoints exist but none loads: restarting from step 0 would overwrite them.
    """
    load = load or _torch_load
    epoch_iters, total_steps = training_steps(seg, repo_dir)
    directory = checkpoint_root / "model_checkpoints" / run_name(seg)
    if training_complete(directory, total_steps, epoch_iters, load=load):
        return None
    candidates = checkpoint_candidates(directory, epoch_iters)
    for path in candidates:
        if _resumable(path, load):
            return ["--checkpoint", str(path)]
    if candidates:
        raise SystemExit(
            f"no resumable checkpoint among {[str(p) for p in candidates]}; refusing to restart "
            "from step 0 over an existing run"
        )
    return []


def run_name(seg: Mapping) -> str:
    return str((seg.get("overrides") or {}).get("run-name", DEFAULT_RUN_NAME))


def training_steps(seg: Mapping, repo_dir: Path = REPO_DIR) -> tuple[int, int]:
    """(iterations per epoch, total optimizer steps), with the run's overrides applied."""
    training = yaml.safe_load(
        (repo_dir / seg.get("config_training", DEFAULT_TRAINING_CONFIG)).read_text()
    )["training"]
    overrides = seg.get("overrides") or {}
    epoch_iters = int(
        overrides.get("iterations-per-train-epoch", training["iterations_per_train_epoch"])
    )
    epochs = int(overrides.get("epochs", training["epochs"]))
    return epoch_iters, epochs * epoch_iters


def _copy_into(src: Path, synced_dir: Path, name: str) -> None:
    """Hard-link `src` as `synced_dir/name` (copy when linking fails) via the fixed temp name.

    A link shares the inode, so a mirror costs no second copy on the ML volume. Safe because
    preempts are written to a new inode (.partial + rename) and step/epoch files are linked only
    once stable; restore copies step/epoch files, so the trainer never rewrites a linked one.
    """
    tmp = synced_dir / MIRROR_TMP
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, synced_dir / name)


def _snapshot(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def sync_once(
    local_dir: Path,
    synced_dir: Path,
    previous: Mapping[str, tuple[int, int]] | None = None,
    *,
    final: bool = False,
) -> dict[str, tuple[int, int]]:
    """Mirror what resume needs from the trainer's `local_dir` into the S3-synced `synced_dir`.

    The newest model_preempt* (written atomically) goes to `resume.ckpt` with a `resume.json`
    marker naming it. model_step* / model_epoch* are written in place, so each is copied
    under its own name only once its size and mtime match `previous`, the snapshot the last
    call returned. `final` (the trainer exited, any status) copies them without waiting and also
    keeps the newest preempt under its own name. Returns this call's snapshot.
    """
    snapshot: dict[str, tuple[int, int]] = {}
    if not local_dir.is_dir():
        return snapshot
    synced_dir.mkdir(parents=True, exist_ok=True)
    previous = previous or {}

    preempts = sorted(
        (int(m.group(1)), path)
        for path in local_dir.glob("model_preempt*")
        if path.is_file() and (m := re.fullmatch(r"model_preempt(\d+)", path.name))
    )
    if preempts:
        newest = preempts[-1][1]
        marker = synced_dir / RESUME_MARKER
        mirrored = json.loads(marker.read_text()).get("name") if marker.is_file() else None
        if mirrored != newest.name or not (synced_dir / RESUME_NAME).is_file():
            _copy_into(newest, synced_dir, RESUME_NAME)
            # The marker follows the data it names.
            (synced_dir / MIRROR_TMP).write_text(json.dumps({"name": newest.name}))
            os.replace(synced_dir / MIRROR_TMP, marker)
            log.info("mirrored %s as %s", newest.name, RESUME_NAME)
        if final and not (synced_dir / newest.name).is_file():
            _copy_into(newest, synced_dir, newest.name)

    for path in sorted(local_dir.glob("model_*")):
        if not path.is_file() or not re.fullmatch(r"model_(step|epoch)\d+", path.name):
            continue
        snapshot[path.name] = _snapshot(path)
        target = synced_dir / path.name
        if target.is_file() and target.stat().st_size == snapshot[path.name][0]:
            continue
        if final or previous.get(path.name) == snapshot[path.name]:
            _copy_into(path, synced_dir, path.name)
            log.info("mirrored %s", path.name)

    sidecar = local_dir / SIDECAR_NAME
    if sidecar.is_file():
        target = synced_dir / SIDECAR_NAME
        if not target.is_file() or target.read_bytes() != sidecar.read_bytes():
            _copy_into(sidecar, synced_dir, SIDECAR_NAME)
    return snapshot


def restore_checkpoints(synced_dir: Path, local_dir: Path) -> None:
    """Put the mirrored checkpoints back into `local_dir` under their original names."""
    if not synced_dir.is_dir():
        return
    local_dir.mkdir(parents=True, exist_ok=True)

    def place(src: Path, name: str) -> None:
        target = local_dir / name
        if target.exists():
            return
        # Preempts are replaced by rename, so a link is safe. The trainer can rewrite a step/epoch
        # file in place (an unresumable newer one it re-reaches), so those are copied.
        try:
            if not re.fullmatch(r"model_preempt\d+", name):
                raise OSError("step/epoch files are copied")
            os.link(src, target)
        except OSError:
            shutil.copyfile(src, target)
        log.info("restored %s", target)

    marker = synced_dir / RESUME_MARKER
    if marker.is_file() and (synced_dir / RESUME_NAME).is_file():
        place(synced_dir / RESUME_NAME, json.loads(marker.read_text())["name"])
    for path in sorted(synced_dir.glob("model_*")):
        if path.is_file() and re.fullmatch(r"model_(preempt|step|epoch)\d+", path.name):
            place(path, path.name)


def build_trainer_argv(
    seg: Mapping, python: str = sys.executable, resume: list[str] | None = None
) -> list[str]:
    """Translate the run YAML `config:` block into the trainer command line.

    A non-empty `resume` replaces any checkpoint/init overrides and is appended.
    """
    unknown = sorted(set(seg) - set(_CONFIG_FLAGS) - {"overrides"})
    if unknown:
        raise SystemExit(f"Unsupported keys in run YAML `config:` block: {unknown}")
    argv = [python, "-u", TRAINER_SCRIPT]
    for key, flag in _CONFIG_FLAGS.items():
        if key in seg:
            argv += [flag, str(seg[key])]
    shadowing = sorted(set(_CONFIG_FLAGS.values()) & {f"--{n}" for n in seg.get("overrides") or {}})
    if shadowing:
        raise SystemExit(
            f"`overrides:` must not set {shadowing}; use the `config_*` keys instead "
            "(a later flag would override the config the step counts are read from)."
        )
    for name, value in (seg.get("overrides") or {}).items():
        if resume and name in _INIT_OVERRIDES:
            continue
        # The trainer's flags are valued options; a bool true is a bare flag, false is omitted.
        if isinstance(value, bool):
            if value:
                argv.append(f"--{name}")
        else:
            argv += [f"--{name}", str(value)]
    return argv + list(resume or [])


def missing_config_files(seg: Mapping, repo_dir: Path = REPO_DIR) -> list[str]:
    """Config paths named in `seg` that do not exist (resolved from `repo_dir`, as the trainer runs)."""
    return [
        str(seg[key]) for key in _CONFIG_FLAGS if key in seg and not (repo_dir / seg[key]).is_file()
    ]


def _mirror_loop(
    wake: threading.Event,
    stop: threading.Event,
    terminated: threading.Event,
    interval: float,
    sync: Callable[[], None],
    forward: Callable[[], None],
) -> None:
    """Run `sync` every `interval` seconds, or at once when `wake` is set.

    After a SIGTERM (`terminated`) it syncs first and then calls `forward`, so the trainer is
    stopped only after the newest stable checkpoints are mirrored. `stop` ends the loop.
    """
    while True:
        wake.wait(interval)
        if stop.is_set():
            return
        wake.clear()
        try:
            sync()
        except Exception:  # noqa: BLE001 - a failed mirror must not stop training
            log.exception("checkpoint mirror failed; retrying next interval")
        if terminated.is_set():
            forward()
            terminated.clear()


def run_mirrored(
    cmd: list[str],
    local_dir: Path,
    synced_dir: Path,
    *,
    cwd: Path,
    interval: float = SYNC_INTERVAL_S,
) -> int:
    """Run the trainer while mirroring checkpoints; return its exit status.

    SIGTERM (StopTrainingJob, spot reclaim) triggers an immediate mirror and is then forwarded
    to the trainer. Once the trainer exits, a final sync runs whatever its status: a truncated
    file it left behind is skipped by `_resumable` on restore.
    """
    wake, stop, terminated = threading.Event(), threading.Event(), threading.Event()
    last: dict = {"snapshot": None}

    def sync() -> None:
        last["snapshot"] = sync_once(local_dir, synced_dir, last["snapshot"])

    proc = subprocess.Popen(cmd, cwd=cwd)

    def forward() -> None:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)

    def on_sigterm(signum, frame) -> None:
        terminated.set()
        wake.set()

    previous = signal.signal(signal.SIGTERM, on_sigterm)
    mirror = threading.Thread(
        target=_mirror_loop,
        args=(wake, stop, terminated, interval, sync, forward),
        name="checkpoint-mirror",
        daemon=True,
    )
    mirror.start()
    try:
        status = proc.wait()
    finally:
        stop.set()
        wake.set()
        mirror.join()
        signal.signal(signal.SIGTERM, previous)
    sync_once(local_dir, synced_dir, last["snapshot"], final=True)
    return status


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _find_run_yaml() -> Path:
    yamls = sorted(CONFIG_DIR.rglob("*.yaml")) + sorted(CONFIG_DIR.rglob("*.yml"))
    candidates = []
    for p in yamls:
        try:
            parsed = yaml.safe_load(p.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(parsed, dict) and "job" in parsed:
            candidates.append(p)
    if not candidates:
        raise SystemExit(f"No run YAML with a job: block under {CONFIG_DIR}")
    if len(candidates) > 1:
        raise SystemExit(f"Multiple candidate run YAMLs under {CONFIG_DIR}: {candidates}")
    return candidates[0]


def main():
    _configure_logging()
    run_yaml = _find_run_yaml()
    log.info("Loaded run YAML: %s", run_yaml)
    data = yaml.safe_load(run_yaml.read_text())

    # Apply env: from job block (launcher already injected these into
    # the container env, but if a user lands here via `docker run`
    # directly the YAML is the source of truth.)
    job_env = data.get("job", {}).get("env") or {}
    for k, v in job_env.items():
        os.environ.setdefault(k, str(v))

    seg = data.get("config")
    if seg is None:
        raise SystemExit("Run YAML missing top-level `config:` block.")

    cmd = build_trainer_argv(seg)
    missing = missing_config_files(seg)
    if missing:
        raise SystemExit(f"Config files not found under {REPO_DIR}: {missing}")

    # SageMaker creates this directory only when the job has a checkpoint config.
    checkpointing = CHECKPOINT_DIR.is_dir()
    if checkpointing:
        check_output_dir_owned(seg, job_env)

    # Fail before training starts when a configured secret cannot be read.
    if any(os.environ.get(v) for v in SECRET_ENV):
        import boto3

        resolve_secrets(os.environ, boto3.client("secretsmanager"))

    output_root = LOCAL_CHECKPOINT_DIR if checkpointing else None
    for k, v in build_trainer_env(os.environ, output_root).items():
        os.environ[k] = v
    Path(os.environ["MERMAIDSEG_LOCAL_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    local_run_dir = LOCAL_CHECKPOINT_DIR / "model_checkpoints" / run_name(seg)
    synced_run_dir = CHECKPOINT_DIR / "model_checkpoints" / run_name(seg)
    if checkpointing:
        restore_checkpoints(synced_run_dir, local_run_dir)
        resume = resume_args(seg, LOCAL_CHECKPOINT_DIR)
        if resume is None:
            log.info("Training already finished in %s; nothing to run", local_run_dir)
            return
        if resume:
            log.info("Resuming from %s", resume[-1])
        cmd = build_trainer_argv(seg, resume=resume)

    if os.getenv("SMOKE_TEST"):
        log.info("Smoke test OK — skipping trainer invocation")
        return

    log.info("Invoking (cwd=%s): %s", REPO_DIR, " ".join(cmd))
    if not checkpointing:
        sys.exit(subprocess.call(cmd, cwd=REPO_DIR))

    sys.exit(run_mirrored(cmd, local_run_dir, synced_run_dir, cwd=REPO_DIR))


if __name__ == "__main__":
    main()
