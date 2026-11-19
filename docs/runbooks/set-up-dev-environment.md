# Set up a development environment

This runbook gives you a local checkout that can read the training data from S3, download the DINOv3 encoder and log a training run. For the Slurm cluster, follow [Set up the Slurm cluster](set-up-slurm.md) after this one.

- [Before you start](#before-you-start)
- [Install the project](#install-the-project)
- [Load the environment file](#load-the-environment-file)
- [Get access to the DINOv3 models](#get-access-to-the-dinov3-models)
- [Choose an experiment tracker](#choose-an-experiment-tracker)
- [Check the setup](#check-the-setup)

## Before you start

You need:

- [uv](https://docs.astral.sh/uv/).
- Python 3.11.5 or later (`requires-python` in `pyproject.toml`). `.python-version` pins 3.12, and uv uses it by default.
- AWS credentials for the `mermaid-core` profile, for S3 access.
- The AWS CLI, for `aws configure` and the `make sm-*` monitoring targets.
- Docker with buildx, only for `make sm-smoke`.
- A Hugging Face account, for the DINOv3 models.

## Install the project

1. Clone the repository:

   ```bash
   git clone https://github.com/data-mermaid/mermaid-segmentation.git
   cd mermaid-segmentation
   ```

2. If you do not have uv, install it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Configure the AWS profile:

   ```bash
   aws configure --profile mermaid-core
   ```

4. Create `.env` from the example, and set the profile in it:

   ```bash
   cp .env.example .env
   echo "AWS_PROFILE=mermaid-core" >> .env
   ```

5. Install the locked dependencies, with the extra for your tracker (see [Choose an experiment tracker](#choose-an-experiment-tracker)):

   ```bash
   uv sync --locked --extra wandb
   ```

`uv sync` installs the `dev` group by default. Add `--no-dev` to leave it out.

| Group | Contents |
|---|---|
| `tests` | pytest and pytest-cov |
| `lint` | ruff, pre-commit and mypy |
| `dev` | `tests` and `lint` |

The CoralNet scraper and dataset build are separate uv workspace members. A root `uv sync` does not install them. You need them only to rebuild the CoralNet data; see [Recreate and update the CoralNet data](coralnet-data.md).

## Load the environment file

Python code does not read `.env`. Load it into each shell before you run a command:

```bash
set -a; source .env; set +a
```

Or use direnv: add a `.envrc` that contains `dotenv`, run `direnv allow`, and restart your shell.

`.env.example` lists every variable, grouped by the tool that reads it.

## Get access to the DINOv3 models

The DINOv3 models, for example `facebook/dinov3-vitl16-pretrain-lvd1689m`, are gated on Hugging Face.

1. Open the [model page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), sign in, and accept the license.
2. Create a token at [hf.co/settings/tokens](https://hf.co/settings/tokens).
3. Log in with the token:

   ```bash
   uv run hf auth login
   ```

   As an alternative, add `HF_TOKEN=hf_xxxxxxxxxxxx` to `.env`.

## Choose an experiment tracker

`mermaidseg/logger.py` logs through one interface with two backends. The `tracker` key of the logger config (`wandb` or `mlflow`) selects the backend. Any other name raises `ValueError`. The core install has no tracker, so install the extra for your backend.

| Backend | Use | Logger config | Extra | Environment |
|---|---|---|---|---|
| wandb | Local and Slurm runs | `configs/logger_config.yaml` (the `--logger-config` default) | `wandb` | Local: `WANDB_API_KEY`, or an earlier `wandb login`. Slurm: `WANDB_API_KEY` in the env file, required |
| MLflow | SageMaker runs | `configs/logger_config_sagemaker.yaml` (set by the SageMaker run YAMLs) | `mlflow` | `MLFLOW_TRACKING_URI`. The run raises if it is not set. The SageMaker launcher sets it from `SM_MLFLOW_TRACKING_ARN` |

If the extra for the tracker is not installed, the run stops with an `ImportError` that names the extra. This happens when the tracker starts, after dataset and model setup.

A run that crashes after its tracker starts is recorded as `FAILED` in MLflow, and with `exit_code` 1 in wandb.

The wandb project comes from `wandb_project` in `configs/logger_config.yaml`. `WANDB_PROJECT` has no effect. `WANDB_ENTITY` applies only when `wandb_entity` in the config is `null`.

## Check the setup

Run a short local training. `--num-workers` is required:

```bash
uv run python scripts/train.py --num-workers 4 --cache-dir /path/to/cache
```

Checkpoints go to `outputs/model_checkpoints/<run>/`. `--output-dir` changes `outputs`. The run writes two files there:

- `config.yaml`: the resolved config, as a record of the settings the run used. Do not pass it back in as a run config.
- `model.json`, written when the run starts: the architecture, encoder config, label and concept maps, preprocessing and lineage (git SHA, dirty flag, `uv.lock` hash, seed, data version, resolved config, and the torch, transformers and peft versions). A resumed or warm-started run also records the source checkpoint's git SHA and dirty flag under `parent`. `parent` is absent on a fresh run and `null` when the source checkpoint had no lineage.

The config flags (`--training-config`, `--model-config`, `--data-config`, `--concept-schema-config`, `--logger-config`) take a path relative to the repo root, or an absolute path. The defaults are in `configs/`. `--resolution` sets the train crop size and the model input size. Run `scripts/train.py --help` for resume (`--checkpoint`), init (`--init-weights`) and the other flags.

To check the code without a training run, run the CI test selection. See [Running Tests](../../README.md#running-tests).
