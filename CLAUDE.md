# mermaid-segmentation: Claude Code Guide

This document describes the codebase, conventions, and how to work with Claude Code in this project.

## Project Overview

Coral reef semantic segmentation using DINOv2/v3 and concept bottleneck models for interpretable coral classification.

**Key domains:**
- Computer vision: Coral reef benthic segmentation, dataset handling, preprocessing
- Deep learning: DINOv3 fine-tuning, ConceptBottleneckModel architecture, loss functions
- MLOps: Experiment tracking with MLflow, SageMaker training, reproducibility
- Data engineering: S3 pipelines, Parquet storage, dataset class design

## Using Claude Code

### Quick Start

1. **Route tasks to specialists:** Use `/using-mermaid-ds` skill to find the right agent or skill
2. **Expert agents:**
   - `mermaid-cv-expert` — Coral reef CV, datasets, preprocessing
   - `mermaid-ml-expert` — PyTorch, training, loss functions
   - `mermaid-mlops-expert` — MLflow, checkpoints, SageMaker
   - `mermaid-data-expert` — S3, data loading, dataset design

3. **Utility skills:**
   - `/run-experiment-project` — Validate and launch experiments
   - `/explore-dataset-project` — Dataset diagnostics
   - `/pytest-models` — Model testing patterns
   - `/training-reproducibility-runbook` — Reproducibility checklist
   - `/deep-learning-python` — DINOv3 and PyTorch patterns
   - `/data-analysis-jupyter` — EDA in notebooks
   - More: `/requesting-code-review`, `/deslop`, `/dispatching-parallel-agents`, `/ml-pipeline-integration-tests`

### Example Workflows

**Debug NaN loss:**
> "I'm getting NaN loss mid-training. I've tried lowering LR, but still failing."

Claude dispatches to `mermaid-ml-expert` who debugs gradient flow, loss scaling, etc.

**Before launching an experiment:**
> "I have a new DINOv3 baseline config ready. Can you validate it?"

Claude dispatches to `/run-experiment-project` which checks YAML structure, hyperparams, validates splits, then offers to launch.

**Understand dataset quality:**
> "Should I use benthos_yuval or pacific_labeled_corals for this baseline?"

Claude dispatches to `mermaid-cv-expert` who discusses dataset provenance, class coverage, annotation quality.

---

## Codebase Conventions

### DINOv3 Patterns (CRITICAL)

These apply to all model code:

1. **Prefix tokens:** DINOv3 uses special prefix tokens for spatial awareness. Always include prefix token encoding in forward pass.
2. **Output contracts:** Dense prediction models return `logits[batch, height, width, classes]`. Always test shapes before training.
3. **Serialization:** Save model state with frozen backbone checksums. Never resume training from frozen states without explicit thaw.
4. **Patch tokens:** DINOv3 processes 14×14 patches (196 tokens). Output resolution is `(H//14, W//14)`. Account for this in decoder upsampling.

**Always validate:**

```python
def test_forward_shape():
    model = SegmentationModel(num_classes=128)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 16, 16, 128), f"Got {logits.shape}"
```

### YAML Config Guard

Experiment configs must follow structure:

```yaml
name: {issue_number}-{description}  # e.g., issue-129-baseline
model:
  backbone: dinov3-vits14
  num_classes: 128
  freeze_backbone: true/false
data:
  dataset_path: s3://mermaid-data/{dataset}/v{version}/
  split_config: configs/splits/{name}.yaml
  batch_size: 32  # Recommended 16-64
training:
  optimizer: adamw
  learning_rate: 1e-4  # Reasonable range: 1e-6 to 1e-2
  num_epochs: 100
  warmup_epochs: 5
  seed: 42  # REQUIRED
logging:
  log_dir: ./logs/{issue_number}/
  checkpoint_interval: 10
  save_best: true
```

**Before launching:** Validate with `/run-experiment-project`.

### Notebook Best Practices

When editing `.ipynb` files:

- ✗ Never store secrets (API keys, credentials)
- ✗ Never hardcode absolute paths (`/Users/...`)
- ✓ Use relative paths: `data/splits/train.json`
- ✓ Use environment variables for creds: `os.getenv('CORALNET_API_KEY')`
- ✓ Document data source and preprocessing steps
- ✓ Include reproducibility info: seeds, data versions, git commit

**Violations are flagged by Claude.**

---

## Model Output Contracts

All dense prediction models must follow:

```
Input:  [batch, channels, height, width]
Output: [batch, height, width, num_classes]
```

Example:

```python
class SegmentationModel(nn.Module):
    def forward(self, x):
        # x: [B, 3, 224, 224]
        features = self.backbone(x)  # [B, 384, 16, 16]
        logits = self.decoder(features)  # [B, 128, 16, 16]
        return logits.permute(0, 2, 3, 1)  # [B, 16, 16, 128]
```

Always test before training:

```python
assert model(x).shape == (batch, height, width, num_classes)
```

---

## Training Reproducibility

Before any training run:

1. **Seed everything** (random, np.random, torch, torch.cuda)
2. **Config is versioned** — store with data and code
3. **Checkpoint naming is deterministic** — `checkpoint_epoch_050.pt`, not timestamps
4. **What to log:**
   - Per-epoch: train_loss, val_loss, val_miou, learning_rate
   - Final: config, seed, git commit, dataset version, checkpoint
5. **Resume vs fork:**
   - Resume: same config, interrupted run
   - Fork: changed hyperparams, new config

**Use `/training-reproducibility-runbook` for full checklist.**

---

## Checkpoint Management

**Naming:** `{model_name}_{split}_{epoch:03d}.pt`

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'config': config,
    'seed': SEED,
}
torch.save(checkpoint, f'logs/issue-129/checkpoint_epoch_{epoch:03d}.pt')
```

**Resume:**

```python
checkpoint = torch.load('logs/issue-129/checkpoint_epoch_050.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## Common Gotchas

| Issue | Fix |
|-------|-----|
| NaN loss | Lower LR, check for uninitialized weights, reduce label smoothing |
| Frozen backbone unfreezes after resume | Always set `param.requires_grad = False` after loading |
| Shape mismatch | Print intermediate shapes: `print(x.shape)` before/after each layer |
| Dataset shifts mid-training | Document split versions: `dataset_version: coralnet-v1.0` |
| Irreproducible results | Check seed, config, data, code haven't changed; use deterministic=True |

---

## File Structure

```
mermaid-segmentation/
├── README.md
├── CLAUDE.md  (this file)
├── Makefile
├── scripts/
│   ├── train.py           (training loop)
│   ├── validate_config.py (YAML validation)
│   ├── analyze_dataset.py (dataset diagnostics)
│   └── launch_sagemaker.py (SageMaker jobs)
├── configs/
│   ├── splits/            (train/val/test splits)
│   └── experiments/       (experiment configs)
├── models/
│   ├── segmentation.py    (base segmentation model)
│   └── cbm.py             (concept bottleneck model)
├── datasets/
│   ├── base.py            (BaseDataset)
│   └── coral_reef.py      (coral reef dataset)
├── tests/
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_integration_pipeline.py
├── logs/                  (training logs, not in git)
└── data/                  (local cache, not in git)
```

---

## Code Review Checklist

Before merging, Claude reviews for:

- ✓ **Correctness** — Logic flows, edge cases handled
- ✓ **Performance** — No unnecessary allocations, efficient data structures
- ✓ **Code quality** — Clear naming, no dead code, reasonable complexity
- ✓ **Project consistency** — Follows conventions, uses project patterns
- ✓ **Testing** — Unit + integration tests, edge case coverage

Use `/requesting-code-review` before merging.

---

## Asking for Help

- **Route tasks:** `/using-mermaid-ds` skill
- **Code review:** `/requesting-code-review` skill
- **Config validation:** `/run-experiment-project` skill
- **Dataset analysis:** `/explore-dataset-project` skill
- **Reproducibility checklist:** `/training-reproducibility-runbook` skill
- **Model testing:** `/pytest-models` skill
- **Clean AI artifacts:** `/deslop` skill
- **Parallel tasks:** `/dispatching-parallel-agents` skill

---

## Useful Commands

```bash
# Validate experiment config
python scripts/validate_config.py configs/experiments/issue-129-baseline.yaml

# Analyze dataset
python scripts/analyze_dataset.py --split train --class-dist --padding-check

# Run training
python scripts/train.py --config configs/experiments/issue-129-baseline.yaml

# Resume training
python scripts/train.py --config configs/experiments/issue-129-baseline.yaml --resume logs/issue-129/checkpoint_epoch_050.pt

# Run tests
pytest tests/ -v

# Smoke test (quick integration test)
pytest tests/test_integration_pipeline.py -k "smoke" --timeout=30

# View TensorBoard
tensorboard --logdir logs/

# View MLflow
mlflow ui
```

---

## Contact

For questions about conventions or collaboration with Claude Code, see `/using-mermaid-ds` or reach out to Lauren.
