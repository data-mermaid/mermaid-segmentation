# Using Notebooks

Jupyter notebooks are the primary way to run training and evaluation experiments in this project. This page explains how to set them up and use them correctly.

---

## The main notebooks

| Notebook | Purpose |
|----------|---------|
| `nbs/Base_Pipeline.ipynb` | Train a standard (non-concept) segmentation model end-to-end |
| `nbs/Concept_Bottleneck_Pipeline.ipynb` | Train a concept bottleneck model end-to-end |
| `nbs/Model_Evaluation.ipynb` | Quantitative and qualitative evaluation of trained models |
| `nbs/datasets/Dataset_Exploration.ipynb` | Explore dataset classes and concept mappings |

---

## Kernel setup

Training notebooks need both the `notebooks` extra (Jupyter, plotting) and the
`training` extra (MLflow). On SageMaker, `make sync` installs everything via
`--all-extras`.

```bash
uv sync --extra notebooks --extra training   # local training setup
# or
uv sync --all-extras                         # everything
```

After sync, the kernel appears in JupyterLab as **Python (mermaid-seg)**.

If the kernel isn't available:

```bash
make kernel
# or manually:
uv run --all-extras python -m ipykernel install --user --name mermaid-seg \
  --display-name "Python (mermaid-seg)"
```

Then restart JupyterLab and select it from the kernel menu.

Validate the environment (env vars, AWS, MLflow):

```bash
make check
```

---

## Running a training notebook

1. Pick a config file from `configs/` (or create a new one — see [Reproducibility](Reproducibility))
2. Open `nbs/Base_Pipeline.ipynb` (or the CBM variant)
3. Set the config path in the first cell
4. Run all cells in order (`Run → Run All Cells`)

The notebook will load data from S3, train the model, and log the run to MLflow automatically. You don't need to add logging calls — it's handled by the pipeline.

---

## S3 access from notebooks

Notebooks read data from AWS S3. Before the first run after opening a terminal or restarting your machine:

1. Ensure `AWS_PROFILE=mermaid-core` is in your `.env`
2. Run `direnv allow` in the repo root (if using direnv)
3. Restart the Jupyter kernel so it picks up the environment variables
4. Verify: `aws sts get-caller-identity` should return your account info

If you hit S3 permission errors mid-run, your session has likely expired. Re-run `aws sso login --profile mermaid-core` and restart the kernel.

---

## Notebooks vs. scripts

| Use notebooks for | Use scripts for |
|-------------------|----------------|
| Exploration and EDA | Repeatable, production-quality runs |
| One-off experiments | Anything you'd want to rerun identically |
| Visualization and analysis | CI or automated workflows |
| Explaining results to others | Runs triggered from the command line |

The script equivalent of the training notebook is `scripts/train.py`. Use it when you need reproducible runs outside of JupyterLab.

---

## Cautions for reproducible science

Notebooks make it easy to get results that are hard to reproduce. These habits prevent the most common pitfalls:

**Always run cells top to bottom.** Notebooks maintain hidden state — variables, imports, and data from previous runs persist in memory. Running cells out of order produces results that can't be reproduced by re-running the notebook cleanly. Before sharing or committing results, do a fresh `Kernel → Restart and Run All` to confirm the notebook runs correctly from scratch.

**Never hardcode experiment parameters in notebook cells.** Hyperparameters, dataset paths, class subsets, and model choices belong in a config file under `configs/`. A parameter buried in cell 14 of a notebook is invisible to teammates and won't be captured in MLflow. If it affects the result, it belongs in the config.

**Log every run to MLflow — including failures.** A run that wasn't logged is a run that can't be referenced, compared, or learned from. Log exploratory runs too: knowing what didn't work is as valuable as knowing what did.

**Don't modify the config mid-run.** If you change a config file and re-run only part of a notebook, your results are a mix of two configurations. Start a new run with a new config instead.

**Record the MLflow run ID.** When you report results (in a ticket, Slack, or the Monday meeting), include the MLflow run ID so teammates can find the full run details — config, metrics, and model checkpoint.

---

## Before committing a notebook

Clear all output cells before committing. Large outputs (images, DataFrames, training logs) bloat the git history and create noisy diffs.

In JupyterLab: **Edit → Clear All Outputs**

Then save and commit the cleaned notebook.
