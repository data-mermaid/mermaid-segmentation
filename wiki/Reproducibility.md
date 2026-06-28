# Reproducibility

Reproducibility is what separates research from guesswork. If a result can't be reproduced — by you six months from now, or by a teammate next week — it might as well not exist.

This project has two paths towards training a model `train.py` and Jupyter Notebooks.

---

## 1. Config YAMLs for everything

All hyperparameters, dataset splits, model architecture choices, and training settings belong in a config file under `configs/`.

If it affects the result, it belongs in the config.

```yaml
# configs/run1_segformer_baseline.yaml
run_name: run1_segformer_baseline
model:
  type: Segformer
  encoder: nvidia/mit-b2
data:
  train_csv: configs/run1_train_sources.csv
  val_csv: configs/run1_val_sources.csv
  class_subset: [Acropora, Porites, Rubble]
  padding: 32
training:
  epochs: 50
  learning_rate: 0.0001
  batch_size: 8
```

The config file is committed to the repo. Your results are then tied to a specific, reviewable set of decisions — not scattered across notebook cells. Config set in notebooks should be at the very top for easy review.

---

## 2. Every experiment logs to MLflow

Every run — including exploratory ones, failed ones, and quick tests — gets logged to MLflow. If it didn't get logged, it didn't happen (for the team). MLFlow logging is set up to track local experiments (on your PC) as well as through Sagemaker.

MLflow is the shared record of what's been tried, what worked, and what didn't. Teammates can't see your local results, and memory is unreliable.

---

## The loop in practice

There are two entry points for training, notebooks or train.py. Both can be used to experiment locally, or on a small subset.
Once you have verified your approach with a small subset, moving your training to SageMaker on the full dataset.

**Notebooks — for early exploration:**
Use `nbs/Base_Pipeline.ipynb` (or the CBM variant) when you're testing an initial idea on a small sample. Notebooks are well-suited for quick iteration: checking that a config loads correctly, validating that a model runs on a small data subset, or inspecting intermediate outputs before committing to a full run.

```
configs/my_experiment.yaml
        ↓
nbs/Base_Pipeline.ipynb  (small subset, exploratory)
        ↓
MLflow run logged automatically
```

**`scripts/train.py` — for full runs, sharing, and deployment:**
Once an idea has been validated in a notebook, graduate to `scripts/train.py` for the full experiment. Scripts produce clean, repeatable runs that aren't affected by notebook state, are straightforward to re-run identically, and produce the model checkpoints suitable for sharing or deployment via MLflow.

```bash
uv run --extra training python scripts/train.py --config configs/my_experiment.yaml
```

If you're preparing a model to share with the team or register in MLflow for model registration, it should ideally been trained through `train.py`, not a notebook.

Both entry points read the same config file and log to the same MLflow experiment — the difference is reliability and reproducibility at scale.

---

## Finding and comparing runs in MLflow

1. Open the MLflow UI (the URL is in your `.env` as `MLFLOW_TRACKING_URI`, or ask the team lead)
2. Select the experiment corresponding to your `run_name`
3. Use the comparison view to plot metrics across runs

Before starting a new experiment, check MLflow to see if something similar has already been tried.

---

## Why this matters

**Reproducibility:** Another team member should be able to re-run your experiment from the config file and get the same result.

**Coordination:** MLflow is the team's shared view of what's been tried. You shouldn't have to interrupt someone to find out if an idea has been tested.

**Debugging:** When something breaks or a result looks wrong, the config + MLflow run together tell the full story. Without them, debugging means guessing.
