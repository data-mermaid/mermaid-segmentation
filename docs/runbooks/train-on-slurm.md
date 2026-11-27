# Train on Slurm

This runbook trains the final model on the Slurm cluster: `cbm_final` at 256 px, then its 768 px fine-tune `cbm_final_res768`. Slurm is the default training lane. [SageMaker](train-on-sagemaker.md) is the backup, and runs the same `scripts/train.py` command.

- [Before you start](#before-you-start)
- [Fill the data cache](#fill-the-data-cache)
- [Launch the runs](#launch-the-runs)
- [Follow a run](#follow-a-run)
- [Continue a run after a time limit or preemption](#continue-a-run-after-a-time-limit-or-preemption)
- [Start a run again from scratch](#start-a-run-again-from-scratch)
- [When something fails](#when-something-fails)

## Before you start

- Complete [Set up the Slurm cluster](set-up-slurm.md).
- Run every command from the repository root on the cluster.
- Load the env file into your shell: `set -a; source .env; set +a`.
- Run the launcher on the login node, where AWS credentials and the Hugging Face hub are reachable.

## Fill the data cache

Training reads only the local cache and `$MERMAIDSEG_SCRATCH/hf-cache`, so a node without AWS or Hugging Face access still trains. The cache at `MERMAIDSEG_CACHE_DIR` holds the dataset versions that `configs/data_config.yaml` pins; `hf-cache` holds the encoder.

`slurm/launch_final.sh` fills both on the login node before it submits a job:

1. `scripts/sync_data.py --data-config configs/data_config.yaml --cache-dir "$MERMAIDSEG_CACHE_DIR"` downloads and verifies the pinned versions. It resizes CoralNet images to 2048 px at sync time.
2. `scripts/warm_hf_cache.py --model-config configs/model_config_cbm_dpt_lora_vitl.yaml`, run with `HF_HOME=$MERMAIDSEG_SCRATCH/hf-cache`, downloads the DINOv3 encoder.

If either step fails, the launcher exits and submits nothing. A rerun skips the cached files that pass verification. The launcher needs `MERMAIDSEG_CACHE_DIR` and `MERMAIDSEG_SCRATCH` in `.env`, AWS credentials that read `mermaid-ic-datasets`, and a Hugging Face token (`HF_TOKEN`) with access to DINOv3. `--dry-run` skips both steps.

To fill the cache without launching, run step 1 by hand. `--workers N` sets the number of download threads:

```bash
uv run --locked python scripts/sync_data.py --data-config configs/data_config.yaml \
    --cache-dir "$MERMAIDSEG_CACHE_DIR"
```

Each launch syncs again, so a changed pin in the data config is picked up without extra steps.

## Launch the runs

`slurm/launch_final.sh` submits both runs through `slurm/train.sbatch`. It is idempotent: it skips a run that has finished or has a job in the queue, so running it again submits only what is missing.

1. Print the `sbatch` commands without submitting them:

   ```bash
   bash slurm/launch_final.sh --dry-run
   ```

2. Submit the runs:

   ```bash
   bash slurm/launch_final.sh
   ```

To submit only some runs, name them: `bash slurm/launch_final.sh cbm_final`.

| Run | What it trains | Resources |
|---|---|---|
| `cbm_final` | 300k iterations at 256 px, batch 32 | 1 GPU, 32 CPUs, 96 GB (the `train.sbatch` defaults) |
| `cbm_final_res768` | 50k iterations at 768 px from `cbm_final`'s `model_final` weights, batch 8 × accumulation 4 | 4 GPUs (DDP under torchrun), 192 GB |

The 768 px run waits for the 256 px run:

- If you submit both together, `cbm_final_res768` depends on `cbm_final` ending successfully (`afterok`).
- If `cbm_final` is not in the same submission, the launcher submits `cbm_final_res768` only when `outputs/model_checkpoints/cbm_final/model_final` exists and has not changed for 10 minutes. Otherwise it prints `waiting : cbm_final_res768` and submits nothing for it. Run the launcher again later.

The run settings are in `configs/training_config_cbm_final.yaml`, `configs/training_config_cbm_final_hires.yaml` and the header comment of `slurm/launch_final.sh`.

## Follow a run

| What | Where |
|---|---|
| Queue state | `squeue --me`. The job name is the run name |
| Job output | `$MERMAIDSEG_LOG_DIR/slurm-<run>-<job id>.out` and `.err` |
| Metrics | wandb. The run ID is `<MERMAIDSEG_WANDB_ID_PREFIX>-<run>` and the group is `cbm_final` |
| Checkpoints | `outputs/model_checkpoints/<run>/` in the checkout |
| Finished | `outputs/model_checkpoints/<run>/.done` exists |

The first lines of the job output say whether the job resumes, starts from an init checkpoint or starts from scratch. They also show the node, the GPUs and the workers for each rank.

## Continue a run after a time limit or preemption

A job has a time limit of 71 h 59 min (`slurm/train.sbatch`). A run that needs more time continues in a new job. Every job of a run continues from the newest checkpoint in `outputs/model_checkpoints/<run>/` (`model_preempt*`, `model_step*`, `model_final` or `model_epoch*`). A `model_final` checkpoint marks a finished run, so a resume from it exits without training. The trainer writes a rolling `model_preempt<step>` checkpoint every 2000 optimizer steps, so a stopped job loses at most 2000 steps. A resume refuses a data config whose dataset or taxonomy pins differ from the checkpoint's.

To continue a run that stopped, run the launcher again:

```bash
bash slurm/launch_final.sh
```

All jobs of a run append to the same wandb run.

## Start a run again from scratch

The launcher does not submit a run that has `.done`, and a new job resumes from the newest checkpoint. To train a run again from the start, move its directory out of the way first:

```bash
mv outputs/model_checkpoints/cbm_final outputs/model_checkpoints/cbm_final.old
```

To keep the new wandb run separate from the old one, change `MERMAIDSEG_WANDB_ID_PREFIX` in the env file.

## When something fails

| Failure | What to do |
|---|---|
| The launcher says a Slurm variable is not set, or `MERMAIDSEG_LOG_DIR` does not exist | Fix the env file. See [Write the env file](set-up-slurm.md#write-the-env-file). |
| The job stops with `set WANDB_API_KEY in the secured env file` | Add `WANDB_API_KEY` to the env file that the job reads. |
| The launcher prints `cache sync failed; nothing submitted` | Read the error above it. Check the AWS credentials, and that the pins in `configs/data_config.yaml` exist in `mermaid-ic-datasets`. Run the launcher again. |
| The launcher prints `encoder download failed; nothing submitted` | Check `HF_TOKEN` and access to the DINOv3 repo, then run the launcher again. |
| The job fails because a dataset is not synced | The cache lacks a version, group or resize that the data config selects. Run the launcher again to sync it. |
| `cbm_final_res768` runs out of GPU memory | Set `FINAL_768_BATCH` and `FINAL_768_ACCUM` in the shell before you launch. Their product must stay 32, or the launcher refuses. |
| A job is cancelled about a second after it starts, with no logs | Do not pass parameters with `sbatch --export=ALL,VAR=...`. A site prolog cancels such jobs. Pass them as arguments to `train.sbatch`. |
| A multi-GPU job times out in a collective | The NCCL flight recorder writes each rank's last collectives to `logs/nccl_trace_<run>_<job id>_rank_*`. |

To submit one job by hand, read the header of `slurm/train.sbatch`. A direct `sbatch` must pass `--account`, `--qos`, `--partition`, `--output` and `--error` itself, and must run from the repository root.
