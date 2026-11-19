# Set up the Slurm cluster

This runbook prepares a checkout on a Slurm cluster so that `slurm/launch_final.sh` can submit training jobs. Do it once per cluster account. To submit a run after setup, follow [Train on Slurm](train-on-slurm.md).

- [Before you start](#before-you-start)
- [Choose the filesystems](#choose-the-filesystems)
- [Clone and install](#clone-and-install)
- [Write the env file](#write-the-env-file)
- [Check the setup](#check-the-setup)

## Before you start

- Complete [Set up a development environment](set-up-dev-environment.md) on a machine with AWS credentials. Training jobs on the cluster do not use AWS, but you fill the data cache with AWS credentials.
- Get the Slurm account, QoS and partition that your jobs submit to.
- Get a wandb API key from [wandb.ai/authorize](https://wandb.ai/authorize). Slurm runs log to wandb.

## Choose the filesystems

A batch job writes `.venv`, `outputs/`, `logs/` and `wandb/` inside the checkout. Put the checkout on a filesystem that batch jobs can write to. AFS does not work, because batch jobs have no AFS token.

`slurm/common.sh` sets up every batch job. It moves `HOME`, the uv cache, the uv Python installs, the Hugging Face cache (`HF_HOME`) and the pip cache under `MERMAIDSEG_SCRATCH`, so a job does not write to your real home directory. It puts `TMPDIR` on node-local `/tmp`, because the DataLoader workers pass batches over Unix sockets that must not be on NFS.

Choose these directories before you write the env file:

| Variable | What it holds |
|---|---|
| `MERMAIDSEG_SCRATCH` | The scratch root for the relocated home directory and caches |
| `MERMAIDSEG_CACHE_DIR` | The local S3 data cache that training reads. Training passes it to `scripts/train.py --cache-dir` |
| `MERMAIDSEG_LOG_DIR` | The Slurm `slurm-%x-%j.out` and `.err` logs. It must exist before you launch |

## Clone and install

1. Clone the repository on the writable filesystem:

   ```bash
   git clone https://github.com/data-mermaid/mermaid-segmentation.git
   cd mermaid-segmentation
   ```

2. If uv is not on the cluster, install it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

You do not need to run `uv sync`. Each batch job runs `uv sync --locked --extra wandb` before it trains.

## Write the env file

The launcher and every batch job read the same env file. By default it is `.env` in the checkout.

1. Create `.env` from the example:

   ```bash
   cp .env.example .env
   ```

2. Set the Slurm launcher values. They are in the "Slurm launcher" block of `.env.example`:

   ```bash
   MERMAIDSEG_SCRATCH=/path/to/scratch
   MERMAIDSEG_CACHE_DIR=/path/to/scratch/cache
   MERMAIDSEG_LOG_DIR=/path/to/scratch/logs
   SLURM_ACCOUNT=your-account
   SLURM_QOS=your-qos
   SLURM_PARTITION=your-partition
   ```

3. Set the wandb key. Batch jobs stop at start if it is missing:

   ```bash
   WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

4. Make the log directory:

   ```bash
   mkdir -p /path/to/scratch/logs
   ```

Keep site values in the git-ignored env file, never in the repository.

If the env file must be outside the checkout, for example an NFS copy that a batch job can read, export `MERMAIDSEG_ENV_FILE` in the shell before you run the launcher. Setting `MERMAIDSEG_ENV_FILE` inside `.env` has no effect.

`MERMAIDSEG_WANDB_ID_PREFIX` (default `r1`) is the prefix of the wandb run ID. Every job of a run appends to one wandb run. To start new wandb runs under the same run names, change the prefix in the env file.

## Check the setup

Print the jobs that the launcher would submit:

```bash
bash slurm/launch_final.sh --dry-run
```

If a Slurm variable is missing, or `MERMAIDSEG_LOG_DIR` does not exist, the launcher stops and names the problem. A dry run prints `would submit: sbatch ...` for each job.

Before the first real launch, fill the data cache. See [Fill the offline data cache](train-on-slurm.md#fill-the-offline-data-cache).
