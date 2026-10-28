"""Engaging ViT-size and loss-weight jobs stay off CSAIL and off S3."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ENGAGING = _REPO / "slurm" / "engaging"

_TRAIN = (
    "vitsize_train.sbatch",
    "vitsize_train_hplus.sbatch",
    "sweep_loss_weights_train.sbatch",
)
_EVAL = (
    "vitsize_eval.sbatch",
    "sweep_loss_weights_eval.sbatch",
)
_SBATCH = _TRAIN + _EVAL


def _text(name: str) -> str:
    return (_ENGAGING / name).read_text()


def test_engaging_scripts_are_not_csail_jobs():
    for name in _SBATCH:
        text = _text(name)
        assert "#SBATCH -p mit_preemptable" in text
        assert "#SBATCH --requeue" in text
        assert "#SBATCH --time=2-00:00:00" in text
        assert "#SBATCH --open-mode=append" in text
        assert "#SBATCH -c 12" in text
        assert "#SBATCH --mem=96G" in text
        assert "#SBATCH -N 1" in text
        assert "#SBATCH --ntasks=1" in text
        assert "vision-beery" not in text
        assert "aws " not in text
        assert "AWS_ACCESS_KEY" not in text
        assert "mermaid-training.env" not in text
        assert "/data/vision/beery" not in text


def test_train_jobs_requeue_past_preemption_and_the_wall():
    body = _text("train_body.sh")
    assert "engaging_resubmit" in body
    assert "on_usr1" in body
    assert "--checkpoint" in body
    assert "--data-seed" not in body
    assert "WANDB_RUN_ID" in body
    assert "WANDB_RESUME=allow" in body
    resume = _text("resume.sh")
    assert 'dependency="afterany:${SLURM_JOB_ID}"' in resume
    assert "optimizer_state_dict" in resume
    assert "scheduler_state_dict" in resume
    assert "model_state_dict" in resume
    for name in _TRAIN:
        text = _text(name)
        assert "#SBATCH --signal=B:USR1@240" in text
        assert "--preempt-checkpoint-every-steps" in text
        assert "PREEMPT_EVERY=1000" in text
        assert "train_body.sh" in text


def test_offline_env_strips_aws_and_keeps_the_cache_override():
    env = _text("env.sh")
    assert "MERMAIDSEG_S3_OFFLINE=1" in env
    assert "HF_HUB_OFFLINE=1" in env
    assert "HF_DATASETS_OFFLINE=1" in env
    assert "TRANSFORMERS_OFFLINE=1" in env
    assert "AWS_EC2_METADATA_DISABLED=true" in env
    assert "unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY" in env
    assert "/archive/vision/beery/red_sea_coral/cache_MERMAID" in env
    assert "MERMAIDSEG_NUM_WORKERS" in env
    assert "export HOME=" not in env


def test_vitsize_grid_matches_csail_and_hplus_is_an_h200():
    csail = (_REPO / "slurm" / "vitsize_train.sbatch").read_text()
    assert "SIZES=(vits vitb vitl vithplus)" in csail
    small = _text("vitsize_train.sbatch")
    assert "SIZES=(vits vitb vitl)" in small
    assert "#SBATCH --array=0-2" in small
    assert "#SBATCH -G h100:1" in small
    assert "training_config_cbm_vitsize.yaml" in small
    assert "--batch-size 64" not in small
    assert "cbm_vitsize_engaging" in small

    hplus = _text("vitsize_train_hplus.sbatch")
    assert "SIZE=vithplus" in hplus
    assert "#SBATCH -G h200:1" in hplus
    assert "#SBATCH --array" not in hplus
    assert "--batch-size 64" in hplus
    assert "--grad-accum-steps" not in hplus
    assert "model_config_cbm_dpt_lora_${SIZE}_256.yaml" in hplus
    assert "cbm_vitsize_engaging" in hplus


def test_loss_weight_grid_matches_csail():
    csail = (_REPO / "slurm" / "sweep_loss_weights_train.sbatch").read_text()
    engaging = _text("sweep_loss_weights_train.sbatch")
    for line in (
        "PIX=(1.0 0.9 0.75 0.5 0.25 0.1 0.0)",
        "IMG=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)",
    ):
        assert line in csail
        assert line in engaging
    assert "#SBATCH --array=0-6" in engaging
    assert "#SBATCH -G h100:1" in engaging
    assert "training_config_cbm_sweep.yaml" in engaging
    assert "EPOCH_ITERS=15000" in engaging
    assert "TOTAL_STEPS=150000" in engaging
    assert "cbm_lossw_engaging" in engaging
    assert "--model-config" not in engaging


def test_evals_keep_the_interval_checkpoints_and_skip_preempt_files():
    vitsize = _text("vitsize_launch_evals.sh")
    assert 'CKPT_GLOB="nbs/model_checkpoints/cbm_vitsize_*/model_step*"' in vitsize
    assert "model_preempt*" in vitsize
    assert "RESOLUTIONS=(256 512)" in vitsize
    assert "cbm_vitsize_vithplus" in vitsize
    assert "gpu=h200" in vitsize
    eval_sbatch = _text("vitsize_eval.sbatch")
    assert 'EVAL_ARGS="${4:---evals all}"' in eval_sbatch
    assert "EVAL_BATCH=4" in eval_sbatch
    assert "vithplus" in eval_sbatch
    assert "#SBATCH -G h100:1" in eval_sbatch
    assert "--num-workers 8" in _text("eval_body.sh")
    assert "aws sts" not in eval_sbatch
    assert "env.sh" in eval_sbatch

    loss = _text("sweep_loss_weights_launch_evals.sh")
    assert 'CKPT_GLOB="nbs/model_checkpoints/cbm_lossw_*/model_epoch*"' in loss
    assert "model_preempt*" in loss
    assert "RESOLUTIONS=(256 512)" in loss
    assert "model_step*" not in loss
    loss_eval = _text("sweep_loss_weights_eval.sbatch")
    assert "configs/model_config_cbm_dpt_lora_vitl_256.yaml" in loss_eval
    assert "configs/model_config_cbm_dpt_lora_vitl.yaml" in loss_eval
    assert "EVAL_BATCH=8" in loss_eval


def test_pipeline_preempt_flag_defaults_off_and_keeps_interval_saves():
    pipeline = (_REPO / "nbs" / "Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py").read_text()
    assert "--preempt-checkpoint-every-steps" in pipeline
    assert "default=0" in pipeline
    assert 'checkpoint_name=f"model_step{step}"' in pipeline
    assert 'name = f"model_preempt{step}"' in pipeline
    assert "atomic=True" in pipeline
    assert "prune_rolling_checkpoints" in pipeline
    assert "epoch_complete=step >= total_updates" in pipeline
    # Epoch checkpoints still run when step checkpoints are off (the loss-weight sweep).
    assert "checkpoint_every <= 0" in pipeline


def test_latest_checkpoint_picks_the_highest_step(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    for name in (
        "model_epoch0",
        "model_preempt16000",
        "model_step25000",
        "model_preempt24000",
        "model_preempt24000.partial",
        "model_step25000.partial",
    ):
        (run / name).write_text("x")
    script = _ENGAGING / "resume.sh"
    result = subprocess.run(
        ["bash", "-c", f'source "{script}"; latest_checkpoint "{run}" 15000'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("model_step25000")

    (run / "model_preempt26000").write_text("x")
    result = subprocess.run(
        ["bash", "-c", f'source "{script}"; latest_checkpoint "{run}" 15000'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("model_preempt26000")

    (run / "model_epoch9").write_text("x")
    result = subprocess.run(
        ["bash", "-c", f'source "{script}"; latest_checkpoint "{run}" 15000'],
        check=True,
        capture_output=True,
        text=True,
    )
    # model_epoch9 is (9 + 1) * 15000 = 150000 updates.
    assert result.stdout.strip().endswith("model_epoch9")


def test_launchers_dry_run_and_reject_a_bad_gpu():
    vitsize = _ENGAGING / "vitsize_launch_train.sh"
    hplus = subprocess.run(
        ["bash", str(vitsize), "--hplus", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert "submitted" not in hplus.stdout
    assert "sbatch -G h200:1" in hplus.stdout or "skip" in hplus.stdout

    refused = subprocess.run(
        ["bash", str(vitsize), "--hplus", "--gpu", "a100"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert refused.returncode == 2

    small = subprocess.run(
        ["bash", str(vitsize), "--dry-run", "--gpu", "a100"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert "h200:1 slurm/engaging/vitsize_train_hplus.sbatch" not in small.stdout
    if "would submit" in small.stdout:
        assert "-G a100:1" in small.stdout
        assert "vitsize_train.sbatch" in small.stdout

    loss = subprocess.run(
        ["bash", str(_ENGAGING / "sweep_loss_weights_launch_train.sh"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    if "would submit" in loss.stdout:
        assert "-G h100:1" in loss.stdout
        assert "sweep_loss_weights_train.sbatch" in loss.stdout
