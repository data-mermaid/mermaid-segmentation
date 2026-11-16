# Architecture

This page describes the code on main as of this writing; where it disagrees with the code, the code wins.

## Project Overview

MermaidSeg is a PyTorch semantic segmentation framework for coral reef imagery. It trains pixel-level benthic segmentation models on sparse point annotations pooled from nine source datasets — each with its own label vocabulary — with images and annotation parquets in S3. This is a research/exploration codebase, not yet in production.

## Architecture

| Path | Role |
|---|---|
| `mermaidseg/datasets/` | Nine per-source PyTorch datasets; each emits labels in its own source space |
| `mermaidseg/dataset_reconciliation/` | Joint label + concept space across datasets — the central mechanism |
| `mermaidseg/model/` | Architectures, losses, `MetaModel`, training loop, `Evaluator` |
| `mermaidseg/io.py` | `ConfigDict` + YAML merge + CLI override plumbing |
| `mermaidseg/logger.py` | MLflow runs, checkpoints, model export |
| `mermaidseg/sagemaker/` | Pydantic schema for the launcher YAML (host-side only) |

### Configuration System

Four independent YAML files, each overridable alone via `--config-data`, `--config-model`, `--config-training`, `--config-logger`; `setup_config()` (`mermaidseg/io.py`) merges them under their section keys and CLI args override the result. `ConfigDict` gives dot-notation access. `preprocess_data_config` expands the per-dataset `default:` block into every dataset that doesn't override it, then compiles each split's `transform:` spec into an `albumentations.Compose`. `class_subset`, `whitelist_sources` and `blacklist_sources` take inline lists of source ids or class names — there is no CSV coercion anywhere in `io.py`, so a filename passed here is treated as a literal label.

Switching between standard and concept-bottleneck training means swapping `--config-model` and `--config-training`, not editing one large config.

### Label-space reconciliation (`mermaidseg/dataset_reconciliation/`)

Each dataset emits `source_labels` in its native space (CoralNet provider IDs, MERMAID benthic attribute IDs, Coralscapes 1..39, Coralscapes V2 1..95, …). `SourceLabelRegistry` is built once per run from the split datasets and produces:

- a global source label space (0 = background, positive ints unique across datasets — `__getitem__` masks arrive already offset into it),
- `source_to_target`, a `(N+1,)` LongTensor into the **MERMAID benthic attribute space** (the canonical target space; MERMAID's own mapping is the identity),
- `source_to_concepts`, an `(N+1, C)` FloatTensor of concept rows.

`MetaModel` therefore only does two on-device lookups per batch. `label_mapping.py` holds the `fetch_*_to_mermaid` helpers (several call `https://api.datamermaid.org`; Coralscapes uses static dicts). Adding a dataset means adding a mapping here, not touching the trainer.

### Model Architectures (`mermaidseg/model/models.py`)

Every architecture is a frozen DINOv3 encoder (ViT-B/ViT-L, 512×512 input) plus two orthogonal choices — head type and whether the classifier routes through a concept bottleneck. The six public classes are that matrix:

| | linear 1×1 conv head | DPT multi-scale head | DPT + LoRA |
|---|---|---|---|
| **classes** | `LinearDINOv3` | `LinearDPTDINOv3` | `LinearDPTLoRADINOv3` |
| **concepts → classes** | `ConceptBottleneckDINOv3` | `ConceptBottleneckDPTDINOv3` | `ConceptBottleneckDPTLoRADINOv3` |

All return HuggingFace `SemanticSegmenterOutput` (CBM variants return `ConceptBottleneckOutput`). LoRA adapters attach to the encoder's `q_proj`/`k_proj`/`v_proj`/`o_proj`.

### MetaModel (`mermaidseg/model/meta.py`)

Central training wrapper. Resolves the model by name via `getattr(mermaidseg.model.models, model_name)`, and losses and optimizers the same way. Three training modes:

- `standard` — direct label classification
- `concept` — concept head predicts concepts, post-processed to labels via `postprocess_predicted_concepts`
- `concept-bottleneck` — encoder → concept logits → label logits, with combined loss

`freeze_encoder` defaults to `True` when the mode is `concept-bottleneck`, `False` otherwise. Mixed precision, LR warmup and gradient clipping are configured under `training:`.

### Data Pipeline (`mermaidseg/datasets/`)

- `BaseCoralDataset` fetches images from S3, loads point annotations from parquet, and builds masks by padding each point (radius from `training.padding`, default 3px). Load failures are recorded per-dataset rather than raising; `train_model` aborts if the per-epoch failure rate exceeds 5%.
- Masks use `ignore_index=0` for unlabeled pixels — the vast majority, since annotations are sparse points.
- Nine dataset classes are exported from `mermaidseg/datasets/__init__.py`; **eight** are wired into `configs/data_config.yaml` and `scripts/train.py`. `UCSDMosaicsDataset` exists but is not yet wired into a run.
- `datasets/coralnet/` is a subsystem, not a file: `scraper/` (download, refresh, remediate) → `etl/` (versioned audit/annotations/images parquets) → `preprocessing/` (resize + manifest) → `CoralNetDataset`. Pin a build with `MERMAID_CORALNET_ANNOTATIONS_VERSION`. See `wiki/CoralNet-ETL.md`.
- Concept definitions come from `configs/class_to_concepts.csv` via `ConceptSchema.from_csv`; the benthic hierarchy is fetched from the MERMAID API (`dataset_reconciliation/concepts.py`).

### Experiment Tracking (`mermaidseg/logger.py`)

`Logger` wraps MLflow. It logs params, per-epoch metrics, dataloader params, reconciliation metadata (`metadata/id2label.json`, `metadata/concept_id2name.json`, `metadata/target_id2label.json` — the demo consumes these **by name**, so adding or renaming one is a breaking change), and checkpoints.

Serialization is deliberately three-way (see the `logger.py` module docstring):

- **Training checkpoints** — `torch.save` (pickle), full state for resume.
- **MLflow best model** — `torch.save` to a temp dir, then `mlflow.log_artifact`; load with `torch.load('model.pt')`. **Not** `mlflow.pytorch.log_model`, whose internal session flush triggers a `UniqueViolation` on SageMaker's managed MLflow PostgreSQL backend.
- **Published models** — SafeTensors (`save_safetensors_for_publish`), zero-copy and no arbitrary code execution.

### Demo (`demo/`)

A Gradio app for the concept-bottleneck model, deployed to Hugging Face Spaces at
`https://huggingface.co/spaces/datamermaid/mermaid-segmentation`. `app.py` +
`inference.py` + `rendering.py`, with static test images.

Two things make it a drift risk rather than just another directory:

- **Its runtime is not this repo's.** `demo/README.md` front-matter pins
  `sdk_version: 6.17.3` and `python_version: "3.12"`, while the project floor is
  3.11 and the `demo` extra only asks for `gradio>=5.0`. Reproducing a Spaces bug
  locally means matching the front-matter, not the lockfile.
- **It carries committed copies of training outputs** — `id2label.json`,
  `concept_id2name.json`, `class_to_concepts.csv` — plus its own
  `model_config_cbm_dpt_lora_vitl.yaml`. Retraining does not update them; they
  will silently disagree with a newer run's `metadata/` artifacts.

## Key Conventions & Gotchas

- **Resolution by name is pervasive** — model, loss, optimizer, scheduler and every augmentation are looked up by their config string. Renaming a class breaks configs silently, with no import-time error. The one exception is the metric name: `SUPPORTED_METRIC_NAMES` (`mermaidseg/model/metric_policy.py`) backs argparse `choices=`, so a bad `--metric-of-interest` fails immediately.
- **Checkpoint selection and early stopping both key off `--metric-of-interest`** (default `accuracy`; `METRIC_POLICY` fixes each metric's min/max direction). `--early-stopping` is opt-in, with `--early-stopping-patience` (10) and `--early-stopping-min-delta`.
- **`concept_mapping_path` is CWD-relative.** `configs/training_config_cbm.yaml` sets `"../configs/class_to_concepts.csv"` and `ConceptSchema.from_csv` passes it straight to `pd.read_csv`, so it resolves only when the working directory is `nbs/`. `scripts/train.py` defaults to that config — run it from the repo root and it raises `FileNotFoundError`. Override the path, or run from `nbs/`. Omitting the key entirely falls back to the correct package-relative default.
- ImageNet normalization lives in the data config as an Albumentations transform, not in code.
