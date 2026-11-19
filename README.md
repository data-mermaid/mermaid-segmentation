# MermaidSeg

Codebase for the training, evaluation and usage of image segmentation models of coral data.
The codebase contains data loaders, augmentations, preprocessors, models, training & evaluation scripts.

- [Usage](#usage)
  - [Quick start](#quick-start)
  - [Architecture overview](#architecture-overview)
  - [Running Tests](#running-tests)
  - [Common Commands](#common-commands)
- [Model Serialization Strategy](#model-serialization-strategy)
- [Concept Bottleneck Models](#concept-bottleneck-models)
- [Documentation](#documentation)

## Usage

The step-by-step procedures are in [docs/runbooks/](docs/runbooks/README.md): setup, data, training on Slurm and SageMaker, export and evaluation.

### Quick start

```bash
git clone https://github.com/data-mermaid/mermaid-segmentation.git
cd mermaid-segmentation
uv sync --locked --extra wandb
```

You also need AWS credentials for S3 and a Hugging Face token for the gated DINOv3 models. [Set up a development environment](docs/runbooks/set-up-dev-environment.md) has the full steps.

Python code does not read `.env`. Load it into the shell with `set -a; source .env; set +a`, or use direnv.

### Architecture overview

```text
scripts/train.py
  -> mermaidseg/training.py
       configs validated by mermaidseg/config.py (strict pydantic models)
       labels and concepts from mermaidseg/dataset_reconciliation/
       MetaModel (mermaidseg/model/meta.py), Logger over a wandb or MLflow tracker (mermaidseg/logger.py)
  -> outputs/model_checkpoints/<run>/
       model_step* / model_epoch* checkpoints (with lineage), config.yaml, model.json
  -> scripts/export.py (mermaidseg/export.py)
  -> bundle: weights.safetensors + model.json
  -> mermaidseg/inference/
       load_bundle (bundle.py), preprocess (preprocess.py), CBMPredictor (predictor.py)
       used by scripts/evaluate.py
```

`scripts/train.py` is the only training entrypoint. Each compute substrate has a thin launcher that runs it with the same config: your shell (local), `slurm/launch_final.sh` (Slurm) and `scripts/sagemaker/` (SageMaker). `AGENTS.md` section 1 has the directory map.

### Running Tests

```bash
uv sync --locked --group tests --extra wandb --extra mlflow
```

CI installs both tracker extras; without the `mlflow` extra the MLflow tracker tests skip. It runs this selection (`.github/workflows/tests.yml`). Use it before you open a PR:

```bash
uv run --group tests --extra wandb --extra mlflow pytest -m "not slow and not integration and not live"
```

`pyproject.toml` defines these markers:

| Marker | Meaning |
|---|---|
| `live` | Needs S3 or CoralNet access. Deselected by default through `addopts`. |
| `slow` | Slow tests. |
| `integration` | Integration tests. |
| `smoke` | Fast offline smoke tests of the training pipeline wiring. |

Tests that need a GPU skip when none is available.

A test for `<path>/<name>.py` lives at `tests/<path>/test_<name>.py`. For example, the tests for `mermaidseg/logger.py` are in `tests/mermaidseg/test_logger.py`:

```bash
uv run pytest tests/mermaidseg/test_logger.py -v
```

To run the `live` tests, select them with `-m live`.

CI also runs the coralnet-scraper and coralnet-dataset suites. See the Tests sections of [packages/coralnet-scraper/README.md](packages/coralnet-scraper/README.md#tests) and [packages/coralnet-dataset/README.md](packages/coralnet-dataset/README.md#tests).

### Common Commands

```bash
uv lock --check                        # lockfile matches pyproject.toml
uv run --group lint ruff check         # lint
uv run --group lint ruff format --check
uv run --group lint mypy               # type check
uv run --group lint pre-commit run --all-files
```

CI runs the equivalent of the first four commands (`.github/workflows/lint.yml`) when Python files, `pyproject.toml` or `uv.lock` change.

The commands that fill the data cache, launch training and follow SageMaker jobs (`make sm-*`) are in the [runbooks](docs/runbooks/README.md).

## Model Serialization Strategy

The project uses two formats.

**Checkpoints.** Training and resume checkpoints are `torch.save` pickles. They hold the full state (model, optimizer and scheduler) that a resume needs. The code loads them with `weights_only=True` (`read_checkpoint` in `mermaidseg/run_setup.py`). `AGENTS.md` 9.6 allows `weights_only=False` only for a resume checkpoint that this project's own training wrote, with a comment at the call site. No production call site uses it. Load every other checkpoint with `weights_only=True`.

**Bundle.** The deployable bundle is SafeTensors plus `model.json` (`export_bundle` in `mermaidseg/export.py`, `load_bundle` in `mermaidseg/inference/bundle.py`). `model.json` carries the architecture, label and concept maps, preprocessing, lineage and the sha256 of the weights, so a bundle describes itself.

## Concept Bottleneck Models

Non-taxonomic concept metrics in `mermaidseg/model/eval.py` are multiclass `Accuracy` with a hard-coded `num_classes=3` and `ignore_index=0`: class 0 is excluded from the metric, and the other two classes are the concept's two states. Taxonomic concepts use `Accuracy` with a per-concept class count derived from `concept_value2id`.

## Documentation

| Document | Covers |
|---|---|
| [docs/runbooks/README.md](docs/runbooks/README.md) | Index of the runbooks |
| [docs/runbooks/set-up-dev-environment.md](docs/runbooks/set-up-dev-environment.md) | Setting up a development environment |
| [docs/runbooks/set-up-slurm.md](docs/runbooks/set-up-slurm.md) | Setting up the Slurm cluster |
| [docs/runbooks/training-datasets.md](docs/runbooks/training-datasets.md) | Where each dataset comes from, and adding a dataset |
| [docs/runbooks/coralnet-data.md](docs/runbooks/coralnet-data.md) | Recreating or updating the CoralNet data end to end |
| [docs/runbooks/derived-configs.md](docs/runbooks/derived-configs.md) | Regenerating the train + val data config, concept removal list and concept source mask |
| [docs/runbooks/train-on-slurm.md](docs/runbooks/train-on-slurm.md) | Training on Slurm |
| [docs/runbooks/train-on-sagemaker.md](docs/runbooks/train-on-sagemaker.md) | Training on SageMaker |
| [docs/runbooks/export-a-model.md](docs/runbooks/export-a-model.md) | Exporting a checkpoint to a bundle |
| [docs/runbooks/evaluate-a-model.md](docs/runbooks/evaluate-a-model.md) | Evaluating a bundle |
| [docs/agents-md-sources.md](docs/agents-md-sources.md) | The source of each `AGENTS.md` rule |
| [packages/coralnet-scraper/README.md](packages/coralnet-scraper/README.md) | The CoralNet scraper |
| [packages/coralnet-dataset/README.md](packages/coralnet-dataset/README.md) | The CoralNet dataset build |
| [docker/jobs/README.md](docker/jobs/README.md) | The SageMaker job image, its build, IAM and quota |
| [AGENTS.md](AGENTS.md) | Rules for coding agents |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Ways to contribute |
