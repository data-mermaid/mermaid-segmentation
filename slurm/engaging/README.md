# Engaging: ViT-size and loss-weight sweeps

These jobs run on the ORCD Engaging cluster (`mit_preemptable`). They read the
image cache copied from CSAIL and never contact S3. The CSAIL scripts under
`slurm/` are unchanged.

Engaging accounts use your MIT Kerberos id. If SSH says the account does not
exist, sign in once at <https://orcd-ood.mit.edu> to activate it, then SSH to
`orcd-login.mit.edu` (Kerberos password + Duo).

Storage, from the [ORCD filesystem docs](https://orcd-docs.mit.edu/filesystems-file-transfer/filesystems/):

| Path | Quota | Use here |
| --- | --- | --- |
| `/home/$USER` | 200 GB, snapshotted | repo, uv venv, wandb key |
| `~/orcd/scratch` | 1 TB flash, not backed up, removed after 6 months with no login | image cache, HF cache, checkpoints, evals, wandb files, uv cache |
| `~/orcd/pool` | 1 TB disk, not backed up | image cache only if scratch cannot hold it |

Each of those has a 1 million file cap. `~/orcd/.quota` refreshes about every
30 minutes. Do not put the dataset or checkpoints in home.

GPU jobs use partition `mit_preemptable` (48 hour wall, preempted when the
node owner needs it, base cap of 4 GPUs). `--requeue` restarts a preempted
job and resets the 48 hour clock. It does **not** restart a job that runs out
the clock, so the training scripts submit a continuation a few minutes before
the wall. Do not set `--account=vision-beery` or `--qos=vision-beery-*`; those
accounts exist only on CSAIL. Leaving account and qos unset uses the default
base allocation.

An unspecified GPU type is an L40S (44 GB). Every script names a type. A100s
on this partition are 80 GB, H100 80 GB cards report about 79 GB, and H200s
are 140 GB. ViT-H+ asks for an H200. The other runs ask for an H100, and the
launchers can switch them to `a100` or `h200`. RTX PRO 6000 nodes are
Blackwell; this repo pins `torch<2.8`, so those nodes are not requested.
Jobs ask for 12 CPUs so Slurm does not charge a second GPU on the 8-GPU H100
(14 CPUs/GPU) and H200 (15 CPUs/GPU) nodes.

## 1. See how big the copy is (on CSAIL)

Run this on a CSAIL host where both trees are mounted.

```bash
du -sh /archive/vision/beery/red_sea_coral/cache_MERMAID \
      /data/vision/beery/scratch/sauder/hf-cache
# inodes, roughly (scratch allows 1e6 files)
find /archive/vision/beery/red_sea_coral/cache_MERMAID \
     /data/vision/beery/scratch/sauder/hf-cache -printf '.' | wc -c
```

Scratch should still have about 200 GB free after the copy so checkpoints
have room. If the image cache is too big, put it on pool instead (step 3)
and leave the HF cache and checkpoints on scratch.

## 2. Make the directories (on Engaging)

```bash
ssh "$USER@orcd-login.mit.edu"
cat ~/orcd/.quota
mkdir -p ~/orcd/scratch/mermaidseg/{cache,hf,checkpoints,eval_out,wandb,logs,uv-cache,tmp}
```

## 3. Copy the cache from CSAIL (not from S3)

Run the `rsync` on CSAIL, where the files are local. Duo prompts once per
connection. `--partial` lets you re-run the same command after a drop.
Do not copy `~/.aws`, `.env`, or any `model_checkpoints` directory.

A control master keeps the Duo session open for the rest of the copy. On the
CSAIL side, in `~/.ssh/config`:

```
Host orcd-login
    Hostname orcd-login.mit.edu
    User YOUR_KERBEROS
    ControlMaster auto
    ControlPath ~/.ssh/%r@%h:%p
    ControlPersist 300s
```

Then:

```bash
rsync -aH --partial --info=progress2 \
  /archive/vision/beery/red_sea_coral/cache_MERMAID/ \
  orcd-login:~/orcd/scratch/mermaidseg/cache/
rsync -aH --partial --info=progress2 \
  /data/vision/beery/scratch/sauder/hf-cache/ \
  orcd-login:~/orcd/scratch/mermaidseg/hf/
```

Pool fallback, only if scratch cannot hold the image cache. Checkpoints stay
on scratch either way. On Engaging:

```bash
mkdir -p ~/orcd/pool/mermaidseg/cache
```

Re-run the first `rsync` with the destination
`orcd-login:~/orcd/pool/mermaidseg/cache/`, and on Engaging add this to
`~/.bashrc` so later `sbatch` jobs inherit it:

```bash
export MERMAIDSEG_LOCAL_CACHE_DIR="$HOME/orcd/pool/mermaidseg/cache"
```

## 4. uv and the repo (on Engaging)

No AWS CLI and no `~/.aws`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# The installer usually edits ~/.bashrc. If `uv` is not on PATH in a new shell:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# Skip ssh-keygen if ~/.ssh/id_ed25519 already exists.
# Add the printed key at https://github.com/settings/keys
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:data-mermaid/mermaid-segmentation.git ~/mermaid-segmentation
cd ~/mermaid-segmentation
# If slurm/engaging is not on the default branch, check that branch out first.

ln -sfn ~/orcd/scratch/mermaidseg/checkpoints nbs/model_checkpoints
ln -sfn ~/orcd/scratch/mermaidseg/eval_out eval_out
export UV_CACHE_DIR="$HOME/orcd/scratch/mermaidseg/uv-cache"
export UV_LINK_MODE=copy
uv sync --locked
```

## 5. wandb

Create a key at <https://wandb.ai/authorize>. This file is gitignored. It
must not contain `AWS_*` or `MLFLOW_*` variables.

```bash
umask 077
cat > ~/mermaid-segmentation/.env << 'EOF'
WANDB_API_KEY=paste-key-here
WANDB_ENTITY=your-wandb-entity
WANDB_PROJECT=mermaidseg
EOF
chmod 600 ~/mermaid-segmentation/.env
set -a; source ~/mermaid-segmentation/.env; set +a
uv run wandb login --relogin "$WANDB_API_KEY"
uv run wandb status
```

Training sets `WANDB_RESUME=allow` and a stable `WANDB_RUN_ID` per run, so a
requeue appends to the same wandb run. Local wandb files go to
`~/orcd/scratch/mermaidseg/wandb`.

## 6. Offline check (login node)

This must not try to build an S3 client. `ls` should show
`models--facebook--dinov3-vits16-pretrain-lvd1689m`, `vitb16`, `vitl16`, and
`vith16plus`.

```bash
export MERMAIDSEG_LOCAL_CACHE_DIR="${MERMAIDSEG_LOCAL_CACHE_DIR:-$HOME/orcd/scratch/mermaidseg/cache}"
export MERMAIDSEG_S3_OFFLINE=1
export HF_HOME="$HOME/orcd/scratch/mermaidseg/hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AWS_EC2_METADATA_DISABLED=true
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION
ls "$HF_HOME/hub" | grep dinov3
```

The benthic-attribute hierarchy is fetched from `api.datamermaid.org` (not
S3) the first time a job builds its label map. From the login node:

```bash
curl -fsI https://api.datamermaid.org/v1/benthicattributes/ | head -n 1
```

## 7. Confirm GPU type names, then launch

The scripts request `h100`, `h200`, and (via `--gpu a100`) `a100`. Check the
live GRES strings before the first submit. If a name differs, pass the name
`sinfo` prints.

```bash
sinfo -p mit_preemptable -o "%G" | sort | uniq -c
```

From `~/mermaid-segmentation` on a login node. The base allocation runs 4
GPUs at a time; the other train jobs stay pending.

```bash
mkdir -p ~/orcd/scratch/mermaidseg/logs
# ViT-S / ViT-B / ViT-L on H100. --gpu a100|h200 overrides the type.
slurm/engaging/vitsize_launch_train.sh
# ViT-H+ on H200.
slurm/engaging/vitsize_launch_train.sh --hplus
# Seven loss-weight runs on H100.
slurm/engaging/sweep_loss_weights_launch_train.sh
```

Add `--dry-run` to print the `sbatch` lines without submitting. A run that
already has a checkpoint is left alone. Preemption and the 48 hour handoff
resume it. To resume a run that crashed (the job is gone, a checkpoint
exists), submit the sbatch again; it loads the newest checkpoint:

```bash
sbatch slurm/engaging/vitsize_train.sbatch                  # S/B/L, or --array=1 for one
sbatch slurm/engaging/vitsize_train_hplus.sbatch            # H+, H200
sbatch slurm/engaging/sweep_loss_weights_train.sbatch
```

Evals stay on the original intervals: ViT-size `model_step` every 25k updates
through 150k, loss-weight `model_epoch` at the end of each of the 10 epochs.
`model_preempt*` files are only there for preemption and are not evaluated.
Launchers are idempotent. ViT-H+ evals request an H200; the rest use the
train GPU type (`--gpu`, default `h100`).

```bash
slurm/engaging/vitsize_launch_evals.sh
slurm/engaging/sweep_loss_weights_launch_evals.sh
```

`squeue --me` shows what is running. Logs are under
`~/orcd/scratch/mermaidseg/logs/`.
