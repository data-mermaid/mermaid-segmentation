"""slurm/ job bodies, launchers and the experiment configs they point at.

The launchers run in --dry-run mode from a temporary copy of slurm/, so checkpoints
and profiles are faked per test and nothing is submitted.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.build_trainval_data_config import HEADER, build_final_data_config

_REPO = Path(__file__).resolve().parents[1]
_SLURM = _REPO / "slurm"
_CONFIGS = _REPO / "configs"

LOSSW = [
    ("1.0", "0.0"),
    ("0.9", "0.1"),
    ("0.75", "0.25"),
    ("0.5", "0.5"),
    ("0.25", "0.75"),
    ("0.1", "0.9"),
    ("0.0", "1.0"),
]
BRANCH_STEPS = [25000, 75000, 125000, 175000] + list(range(275000, 1_000_000, 100000))
DSIZE_PROFILES = ["exp1_drop0", "exp1_drop10", "exp1_drop20", "exp1_drop50"] + [
    f"exp{n}_drop{f}" for n in (2, 3, 4) for f in (10, 20, 50)
]
HIRES = {256: 1, 384: 1, 512: 4, 768: 4}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(_SLURM, root / "slurm")
    (root / "nbs" / "model_checkpoints").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text('#!/bin/bash\ncat "$QUEUE_FILE" 2>/dev/null || true\n')
    (bin_dir / "sbatch").write_text("#!/bin/bash\necho 'sbatch called in a test' >&2\nexit 1\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return root


def _launch(repo: Path, name: str, *args: str, queued: list[str] = ()) -> tuple[str, list]:
    env = {k: v for k, v in os.environ.items() if k not in ("DS_ROOT", "SPARSE_ROOT", "EVAL_ROOT")}
    env["PATH"] = f"{repo.parent / 'bin'}:{env['PATH']}"
    queue_file = repo.parent / "queue.txt"
    queue_file.write_text("".join(f"{job}\n" for job in queued))
    env["QUEUE_FILE"] = str(queue_file)
    result = subprocess.run(
        ["bash", str(repo / "slurm" / f"launch_{name}.sh"), "--dry-run", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "  would submit: sbatch "
    jobs = [
        shlex.split(line[len(prefix) :])
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    ]
    return result.stdout, jobs


def _old_file(path: Path) -> None:
    """Create PATH with an mtime an hour ago, so the launchers treat it as complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    past = path.stat().st_mtime - 3600
    os.utime(path, (past, past))


def _opt(job: list[str], name: str) -> str | None:
    for token in job:
        if token.startswith(f"--{name}="):
            return token.split("=", 1)[1]
    return None


def _arg(job: list[str], flag: str) -> str | None:
    return job[job.index(flag) + 1] if flag in job else None


def _runs(jobs: list[list[str]]) -> dict[str, list[str]]:
    """Jobs by name; every launcher submits one job per run."""
    runs = {_opt(job, "job-name"): job for job in jobs}
    assert len(runs) == len(jobs), "a run was submitted more than once"
    return runs


def _sbatch_header(name: str) -> dict[str, str]:
    header = {}
    for line in (_SLURM / name).read_text().splitlines():
        match = re.match(r"#SBATCH --([\w-]+)(?:=(.*))?$", line)
        if match:
            header[match.group(1)] = match.group(2) or ""
    return header


def _training_config(name: str) -> dict:
    return yaml.safe_load((_CONFIGS / f"training_config_cbm_{name}.yaml").read_text())["training"]


# ---------- scripts ----------


def test_every_script_parses():
    scripts = [
        *_SLURM.glob("*.sh"),
        *_SLURM.glob("*.sbatch"),
        *(_REPO / "scripts" / "cluster").glob("*.sh"),
    ]
    assert len(scripts) >= 16
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_slurm_has_only_the_unified_files():
    assert sorted(path.name for path in _SLURM.iterdir()) == sorted(
        [
            "common.sh",
            "lib_launch.sh",
            "train.sbatch",
            "eval.sbatch",
            "build_profiles.sbatch",
            *(
                f"launch_{name}.sh"
                for name in (
                    "lossw",
                    "taxloss",
                    "duration",
                    "hires",
                    "build_profiles",
                    "data_experiments",
                    "vitsize",
                    "final",
                    "evals",
                )
            ),
        ]
    )


def test_job_resources():
    train = _sbatch_header("train.sbatch")
    assert train["time"] == "71:59:00"
    assert train["partition"] == "vision-beery-a100"
    assert train["qos"] == "vision-beery-main"
    assert train["gres"] == "gpu:1"

    evaluation = _sbatch_header("eval.sbatch")
    assert evaluation["time"] == "4:00:00"
    assert evaluation["partition"] == "vision-beery-l40s"
    assert evaluation["qos"] == "vision-beery-free"
    assert "requeue" in evaluation

    assert _sbatch_header("build_profiles.sbatch")["partition"] == "vision-beery-cpu"
    for name in ("train.sbatch", "eval.sbatch", "build_profiles.sbatch"):
        assert "/scratch/sauder/mermaid-segmentation/logs/" in _sbatch_header(name)["output"]


def test_no_retired_paths_or_exports():
    for path in _SLURM.iterdir():
        text = path.read_text()
        assert "mermaid-segmentation-final" not in text, path.name
        assert "engaging" not in text.lower(), path.name
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        assert not any("--export" in line for line in code), path.name
    assert 'REPO="$SCRATCH/mermaid-segmentation"' in (_SLURM / "common.sh").read_text()


# ---------- configs ----------


@pytest.mark.parametrize(
    ("name", "updates", "warmup"),
    [
        ("100k", 100_000, 4000),
        ("final", 300_000, 4000),
        ("hires", 25_000, 4000),
        ("final_hires", 50_000, 4000),
    ],
)
def test_decaying_configs_reach_zero_at_the_last_update(name, updates, warmup):
    cfg = _training_config(name)
    scheduler = cfg["scheduler"]
    assert cfg["epochs"] * cfg["iterations_per_train_epoch"] == updates
    assert scheduler["type"] == "LinearLR" and scheduler["end_factor"] == 0.0
    assert scheduler["warmup_iters"] == warmup
    assert scheduler["warmup_iters"] + scheduler["total_iters"] == updates
    assert cfg["batch_size"] * cfg.get("grad_accum_steps", 1) == 32
    assert cfg["loss"]["per_pixel_loss_weight"] == cfg["loss"]["per_image_loss_weight"] == 0.5


def test_duration_configs():
    peak = _training_config("duration_peak")
    assert peak["epochs"] * peak["iterations_per_train_epoch"] == 1_000_000
    assert peak["iterations_per_train_epoch"] == 25000
    assert peak["scheduler"]["type"] == "ConstantLR" and peak["scheduler"]["warmup_iters"] == 4000

    cooldown = _training_config("duration_cooldown")
    assert cooldown["epochs"] * cooldown["iterations_per_train_epoch"] == 25000
    assert cooldown["scheduler"]["total_iters"] == 25000
    assert cooldown["scheduler"]["warmup_iters"] == 0


def test_only_the_final_configs_regularize_the_taxonomy():
    for name in ("100k", "duration_peak", "duration_cooldown", "hires"):
        cfg = _training_config(name)
        taxonomic = cfg.get("taxonomic_loss", {})
        assert taxonomic.get("kl_weight", 0) == 0 and taxonomic.get("partial_weight", 0) == 0
        assert cfg.get("distance_loss", {}).get("weight", 0) == 0
    for name in ("final", "final_hires"):
        assert _training_config(name)["taxonomic_loss"]["kl_weight"] == 0.5
    for path in _CONFIGS.glob("training_config_*.yaml"):
        assert "ramp" not in path.read_text(), path.name


def test_trainval_config_is_rebuilt_from_the_base_config():
    rebuilt = build_final_data_config((_CONFIGS / "data_config.yaml").read_text(), header=HEADER)
    assert rebuilt == (_CONFIGS / "data_config_trainval.yaml").read_text()


def test_launchers_use_existing_configs():
    referenced = set()
    for path in _SLURM.glob("launch_*.sh"):
        referenced |= set(re.findall(r"configs/([\w.]+\.yaml)", path.read_text()))
    referenced = {name for name in referenced if "$" not in name}
    assert referenced
    for name in referenced:
        assert (_CONFIGS / name).is_file(), name


# ---------- training launchers ----------


def test_lossw(repo):
    _, jobs = _launch(repo, "lossw")
    runs = _runs(jobs)
    assert list(runs) == [f"cbm_lossw_pix{p}_img{i}" for p, i in LOSSW]
    for (pixel, image), job in zip(LOSSW, runs.values(), strict=True):
        assert _opt(job, "dependency") is None
        assert _arg(job, "--training-config") == "../configs/training_config_cbm_100k.yaml"
        assert _arg(job, "--per-pixel-loss-weight") == pixel
        assert _arg(job, "--per-image-loss-weight") == image
        assert "--data-config" not in job and "--kl-weight" not in job


def test_taxloss(repo):
    _, jobs = _launch(repo, "taxloss")
    runs = _runs(jobs)
    flags = {
        "cbm_tax_kl": "--kl-weight",
        "cbm_tax_partial": "--partial-weight",
        "cbm_tax_tree": "--distance-weight",
    }
    assert list(runs) == list(flags)
    for run, flag in flags.items():
        job = runs[run]
        assert _arg(job, flag) == "0.5"
        assert sum(other in job for other in flags.values()) == 1
        assert _arg(job, "--training-config") == "../configs/training_config_cbm_100k.yaml"
        assert _arg(job, "--per-pixel-loss-weight") == _arg(job, "--per-image-loss-weight") == "0.5"


def test_duration_waits_for_each_peak_checkpoint(repo):
    _, jobs = _launch(repo, "duration")
    runs = _runs(jobs)
    assert list(runs) == ["cbm_duration_peak"]
    peak = runs["cbm_duration_peak"]
    assert _arg(peak, "--training-config") == "../configs/training_config_cbm_duration_peak.yaml"

    peak_dir = repo / "nbs" / "model_checkpoints" / "cbm_duration_peak"
    for step in BRANCH_STEPS[:-1]:
        _old_file(peak_dir / f"model_epoch{step // 25000 - 1}")
    (peak_dir / f"model_epoch{BRANCH_STEPS[-1] // 25000 - 1}").touch()  # still being written
    out, jobs = _launch(repo, "duration")
    runs = _runs(jobs)
    assert [run for run in runs if run != "cbm_duration_peak"] == [
        f"cbm_duration_cd{step}" for step in BRANCH_STEPS[:-1]
    ]
    assert f"waiting : cbm_duration_cd{BRANCH_STEPS[-1]}" in out
    for step in BRANCH_STEPS[:-1]:
        job = runs[f"cbm_duration_cd{step}"]
        assert _arg(job, "--init-checkpoint") == (
            f"nbs/model_checkpoints/cbm_duration_peak/model_epoch{step // 25000 - 1}"
        )
        assert (
            _arg(job, "--training-config")
            == "../configs/training_config_cbm_duration_cooldown.yaml"
        )
        assert _arg(job, "--data-seed") is not None


def test_hires(repo):
    out, jobs = _launch(repo, "hires")
    assert not jobs and out.count("waiting :") == 4

    source = "nbs/model_checkpoints/cbm_duration_cd275000/model_step25000"
    _old_file(repo / source)
    _, jobs = _launch(repo, "hires")
    runs = _runs(jobs)
    assert list(runs) == [f"cbm_hires_res{res}" for res in HIRES]
    for res, gpus in HIRES.items():
        job = runs[f"cbm_hires_res{res}"]
        assert (_opt(job, "gres") or "gpu:1") == f"gpu:{gpus}"
        assert _arg(job, "--init-weights") == source
        assert _arg(job, "--resolution") == str(res)
        assert _arg(job, "--training-config") == "../configs/training_config_cbm_hires.yaml"
        assert int(_arg(job, "--batch-size")) * int(_arg(job, "--grad-accum-steps")) == 32


def test_profiles_and_data_experiments(repo):
    _, jobs = _launch(repo, "build_profiles")
    assert [job[-5:] for job in jobs if "datasize" in job] == [
        ["datasize", "exp1", "exp2", "exp3", "exp4"]
    ]
    assert [job[-1] for job in jobs if "sparse" in job] == ["sparse"]
    out, jobs = _launch(repo, "data_experiments")
    assert not jobs and out.count("waiting :") == len(DSIZE_PROFILES) + 1

    for profile in DSIZE_PROFILES:
        _old_file(repo / "data_size" / "profiles" / profile / "data_config.yaml")
    _old_file(repo / "data_sparse" / "profile" / "data_config.yaml")
    _, jobs = _launch(repo, "build_profiles")
    assert not jobs

    _, jobs = _launch(repo, "data_experiments")
    runs = _runs(jobs)
    assert list(runs) == [f"cbm_dsize_{p}" for p in DSIZE_PROFILES] + ["cbm_sparse"]
    for profile in DSIZE_PROFILES:
        job = runs[f"cbm_dsize_{profile}"]
        assert _arg(job, "--data-config") == str(
            repo / "data_size" / "profiles" / profile / "data_config.yaml"
        )
        assert _arg(job, "--training-config") == "../configs/training_config_cbm_100k.yaml"
    sparse = runs["cbm_sparse"]
    assert _arg(sparse, "--data-config") == str(
        repo / "data_sparse" / "profile" / "data_config.yaml"
    )


def test_vitsize(repo):
    _, jobs = _launch(repo, "vitsize")
    runs = _runs(jobs)
    assert list(runs) == ["cbm_vitsize_vits", "cbm_vitsize_vitb", "cbm_vitsize_vithplus"]
    for run, job in runs.items():
        size = run.rsplit("_", 1)[1]
        assert _arg(job, "--model-config") == f"../configs/model_config_cbm_dpt_lora_{size}.yaml"
        assert _arg(job, "--training-config") == "../configs/training_config_cbm_100k.yaml"
        assert "--batch-size" not in job


def test_final_runs_the_768_fine_tune_after_the_final_run(repo):
    _, jobs = _launch(repo, "final")
    runs = _runs(jobs)
    assert list(runs) == ["cbm_final", "cbm_final_res768"]
    final, hires = runs["cbm_final"], runs["cbm_final_res768"]
    for job in (final, hires):
        assert _arg(job, "--data-config") == "../configs/data_config_trainval.yaml"
    assert _arg(final, "--training-config") == "../configs/training_config_cbm_final.yaml"
    assert _opt(final, "dependency") is None
    assert _opt(hires, "dependency") == "afterok:100001"
    assert _opt(hires, "gres") == "gpu:4"
    assert _arg(hires, "--init-weights") == "nbs/model_checkpoints/cbm_final/model_step300000"
    assert _arg(hires, "--resolution") == "768"
    assert _arg(hires, "--training-config") == "../configs/training_config_cbm_final_hires.yaml"
    assert int(_arg(hires, "--batch-size")) * int(_arg(hires, "--grad-accum-steps")) == 32


def test_launchers_skip_finished_and_queued_runs(repo):
    (repo / "nbs" / "model_checkpoints" / "cbm_tax_kl").mkdir()
    (repo / "nbs" / "model_checkpoints" / "cbm_tax_kl" / ".done").touch()
    out, jobs = _launch(repo, "taxloss", queued=["cbm_tax_partial"])
    assert "done    : cbm_tax_kl" in out
    assert "queued  : cbm_tax_partial" in out
    assert list(_runs(jobs)) == ["cbm_tax_tree"]


def test_launchers_reject_unknown_options(repo):
    result = subprocess.run(
        ["bash", str(repo / "slurm" / "launch_taxloss.sh"), "--chain", "2"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown option: --chain" in result.stderr


# ---------- eval launcher ----------


def _expected_eval_cells() -> list[tuple[str, str, str, str, int]]:
    vitl = "configs/model_config_cbm_dpt_lora_vitl.yaml"
    final = "model_step100000"
    cells = [("lossw", f"cbm_lossw_pix{p}_img{i}", final, vitl, 512) for p, i in LOSSW]
    cells += [
        ("taxloss", f"cbm_tax_{loss}", final, vitl, 512) for loss in ("kl", "partial", "tree")
    ]
    cells += [("datasize", f"cbm_dsize_{p}", final, vitl, 512) for p in DSIZE_PROFILES]
    cells += [("sparse", "cbm_sparse", final, vitl, 512)]
    cells += [
        ("vitsize", f"cbm_vitsize_{s}", final, f"configs/model_config_cbm_dpt_lora_{s}.yaml", 512)
        for s in ("vits", "vitb", "vithplus")
    ]
    for step in BRANCH_STEPS:
        cells.append(("duration", f"cbm_duration_cd{step}", "model_step25000", vitl, 512))
        cells.append(
            ("duration", "cbm_duration_peak", f"model_epoch{step // 25000 - 1}", vitl, 512)
        )
    cells += [
        ("hires", f"cbm_hires_res{train}", "model_step25000", vitl, res)
        for train in HIRES
        for res in (256, 384, 512, 768, 1024)
    ]
    return cells


def test_evals_cover_every_experiment_checkpoint(repo):
    out, jobs = _launch(repo, "evals")
    assert not jobs

    cells = _expected_eval_cells()
    assert len(cells) == 71
    for _, run, ckpt, _, _ in cells:
        _old_file(repo / "nbs" / "model_checkpoints" / run / ckpt)
    _, jobs = _launch(repo, "evals")
    submitted = [job[job.index("slurm/eval.sbatch") + 1 :] for job in jobs]
    assert submitted == [
        [
            f"nbs/model_checkpoints/{run}/{ckpt}",
            model,
            str(res),
            f"eval_out/{family}/{run}/{ckpt}_res{res}",
        ]
        for family, run, ckpt, model, res in cells
    ]
    assert len({_opt(job, "job-name") for job in jobs}) == len(jobs)

    done = repo / "eval_out" / "lossw" / "cbm_lossw_pix1.0_img0.0" / "model_step100000_res512"
    done.mkdir(parents=True)
    (done / ".done").touch()
    out, jobs = _launch(repo, "evals", "lossw")
    assert len(jobs) == len(LOSSW) - 1
    assert "done: 1" in out
