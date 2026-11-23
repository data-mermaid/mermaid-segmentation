# AGENTS.md

Rules for every coding agent working in this repository, for planning, designing and implementing
each change. Lookup material (setup, commands, architecture) is linked in section 10.

## 0. How to use this file

1. Apply every rule to every change. Cite a rule by section and item, for example 6.1.
2. A rule that says must, never or always is hard. If a request conflicts with a hard rule, do not
   make that change. Stop, name the rule, explain the risk, and ask the human to approve an
   exception or to amend this file.
3. An exception the human approves in the conversation covers that change only. The closest
   `AGENTS.md` to the edited file wins.
4. Every other rule is a judgement rule. Follow it; if you deviate, name the rule and the reason in
   the change summary.

## 1. Map

- `mermaidseg/`: the package. All training (`training.py`), evaluation and inference logic. `mermaidseg/inference/` is the serving surface: preprocessing, predictor and model bundle. `mermaidseg/data/` holds the readers (`PointDataset`, `DenseDataset`) over dataset versions that `segdata_spec` syncs, plus the data-config parser and sync; training does not use it yet.
- `configs/`: committed YAML configs and the concept, mask and label-mapping inputs they reference. `configs/splits/` holds the committed CoralNet train/val/test split (`coralnet.csv`) and its provenance (`coralnet.meta.yaml`); it is reviewed in git and never regenerated at train time.
- `scripts/`: entrypoints for training (`scripts/train.py`), bundle export, evaluation, cache prefetch and the
  data-config, concept-removal, concept-source-mask and CoralscapesV2 probe-point builders, the train label count, the dataset cache sync (`scripts/sync_data.py`, which fills the local cache from pinned dataset versions) and the one-off old-vs-new point parity check (`scripts/check_dataset_parity.py`).
- `slurm/`: Slurm launchers and `train.sbatch`.
- `scripts/sagemaker/`: the SageMaker launcher (`python -m scripts.sagemaker.launch`) and the in-container entrypoint.
- `sagemaker/runs/`: SageMaker run YAMLs.
- `docker/`: the SageMaker job image; deployable images are built only by the CI workflow.
- `packages/`: uv workspace members outside the model path. `packages/coralnet-scraper/` downloads public CoralNet sources to S3; `packages/segdata-etl/` builds the CoralNet training manifest from that scrape and loads the Catlin, Moorea, Pacific and Benthos raw releases into S3 (`segdata ingest`), builds them and the MERMAID benthic taxonomy as versioned datasets (`segdata build`, `segdata taxonomy`); its `segdata_etl/splits/` builds and leakage-checks the split on demand (`segdata split build`, `segdata split leakage`); `packages/segdata-spec/` defines the versioned dataset manifest, S3 layout and sync that an ETL publisher and a trainer share, plus the split-file format and coverage check. `mermaidseg` may import `segdata_spec` and none of the other packages.
- `tests/`: the pytest suite, laid out as 7.4 sets.
- `pyproject.toml`: dependencies, extras, dependency groups, and ruff, mypy and pytest settings.

## 2. Working agreement

1. For a non-trivial change, explore, then plan, then implement. State in the plan which rules it
   touches.
2. Keep diffs minimal: no reformatting, style changes, or refactors of code unrelated to the change.
3. One PR per root cause. Check related code for the same mistake and include any affected sites in
   the same PR.
4. Fix the root cause in the existing structure rather than adding special cases. If the better fix
   carries risk, explain the trade-off in the PR. Write no backward-compatibility code. The one
   exception is a model artifact that has already been delivered: convert it to the current format
   where possible, and keep a loader for it only if conversion is impossible.
5. Don't commit unless the human approved the exact commit message. Never push, open a PR, or post
   to GitHub (comments, reviews, replies) without the human's approval.
6. Never hand-edit a generated file (`uv.lock`, exported artifacts). Change its source and
   regenerate it.
7. For a change that affects model outputs, run the evaluation and put the metrics, with the
   baseline they compare against, in the PR. Don't wait for reviewers to ask.

## 3. Repository structure

1. `mermaidseg/` and `scripts/` hold only the path that trains, evaluates and serves the model. They
   must carry no experiment flags and must never import from `experiments/`. Experiments and
   ablations, code and results, live in `experiments/` and may import the package. To promote
   experiment logic, move it into the package with tests. Delete abandoned paths.
2. Training, evaluation and inference logic must live in `mermaidseg/`. Scripts must be thin
   entrypoints: parse arguments, then call the package. Notebooks must only call package code.
3. There must be one training entrypoint. Each compute substrate (local, Slurm, SageMaker) has a
   thin launcher that runs it with the same config. The package must hold no substrate logic.
4. Evaluation, the demo and the API inference lane must share one inference interface and must not
   re-implement loading, preprocessing or postprocessing.
5. A new data source adds a dataset class and a label mapping into the canonical MERMAID
   benthic-attribute space. The trainer does not change.
6. Committed code holds no absolute, user-specific or cluster paths. Resolve repository files
   relative to the package. Scripts run from the repository root.

## 4. Model and PyTorch code

1. All layer calls should be visible directly in `forward` — avoid helper functions that hide
   `nn.Module` calls.
2. Pass `forward` inputs by name and return a dataclass with named fields rather than a dict of
   tensors, and keep `__init__` arguments flat (plain values, not nested dicts).
3. The DINOv3 encoder stays frozen. Only LoRA adapters (when configured) and heads train. Keep a
   test that asserts the set of trainable parameters.
4. Never hardcode a device: no `.cuda()`, no `"cuda"` defaults, no `autocast(device_type="cuda")`.
   Take the device as an argument and derive the autocast device type from it.
5. Don't hardcode `torch.float32` or `torch.bfloat16` in model `forward`, and don't cast activations
   by reading a weight's dtype. Always derive the cast target from the input tensor's dtype or
   `self.dtype`.
6. The AMP dtype comes from config. Evaluation and inference don't hardcode a precision. Losses
   compute in fp32.
7. Every forward pass outside training runs under `model.eval()` and `torch.inference_mode()`.
8. No NumPy or Python-side computation inside `forward` or the per-step training loop, and no
   `.item()` or `.cpu()` host syncs there except at logging intervals.
9. Seed through one module, `mermaidseg/seeding.py`: `seed_everything` seeds Python, NumPy and
   torch, and every DataLoader takes `make_worker_seeder` as `worker_init_fn` and a seeded
   `generator`.
10. Don't promise bitwise reproducibility across PyTorch releases, platforms, or CPU and GPU.
11. Load state dicts with `strict=True`, unless the mismatch is intended and the call site says why.

## 5. Configs and name resolution

1. A run's settings come only from committed YAML configs and logged CLI overrides, credentials
   excepted (9.5). The fully resolved config is saved with every run.
2. Models, losses, optimizers, schedulers and augmentations resolve by class name from config.
   Renaming one updates every config in `configs/` that names it, in the same change.
3. An unknown class name or config key must raise an error. Never fall back to a default.
4. Support only what committed configs use: the configs of the training and serving path, not those
   under `experiments/`. Ship only the config options, branches, and classes that committed configs
   need. Every config option must appear in a committed config. Don't add options for a config you
   plan to add later; add them with that config. Same bar for runtime arguments: if no committed
   config or entrypoint passes a `forward` or `__call__` parameter, don't add it. Trim those
   surfaces, then delete the unreachable paths.

## 6. Train/serve parity and deployment

1. Deterministic preprocessing (resize, normalization, tiling) and its normalization constants must
   be defined once in package code and imported by the evaluation path and by inference.
2. Augmentation is separate from preprocessing. Serving must run only deterministic preprocessing.
3. A deployable artifact must describe itself and be versioned as one unit: SafeTensors weights,
   architecture config, `id2label`, `concept_id2name`, preprocessing parameters and run lineage.
4. Inference must read label maps and preprocessing parameters only from the artifact, never from
   repository defaults, committed copies or API constants. Never assume class or concept indices
   are stable across checkpoints.
5. The inference path must import only what serving needs: no trainer, experiment-logging, ETL or
   data-loading code.
6. A change to preprocessing, the model forward, export or label maps must come with a parity test:
   the same fixture through the evaluation path and the inference path gives matching outputs
   within tolerance.
7. Every run must log the git SHA, a dirty-tree flag, the resolved config, the data version, the
   lockfile hash and the seed. Artifacts for deployment must come from a clean tree.
8. The file names and formats of exported artifacts are a contract with their consumers (the demo
   and `mermaid-inference`). Never change one without coordinating with those consumers.
9. The request/response contract and API-boundary validation belong to `mermaid-inference`. Never
   duplicate them here.

## 7. Tests

1. For a bug fix, write the test first and confirm it fails. Then fix the bug and confirm the test
   passes.
2. Build tests from tiny configs and random weights, not real checkpoints, S3 or the network. Use
   those only when the behavior can't be reproduced at tiny scale, and mark that test `live`.
3. A test that needs a GPU skips when none is available.
4. The test for `<path>/<name>.py` lives at `tests/<path>/test_<name>.py`: `scripts/train.py` →
   `tests/scripts/test_train.py`, `mermaidseg/model/meta.py` → `tests/mermaidseg/model/test_meta.py`.
   Extend that file if it exists. If you change tests whose file is elsewhere, move that file to
   the mirrored path in the same change. Reuse existing `conftest.py` fixtures and helpers. One
   behavior per test, smallest setup that triggers it. No flaky tests: they are worse than none.
5. Run the tests relevant to the change. Before proposing a PR, run what CI runs:
   `uv run --group tests --extra wandb --extra mlflow pytest -m "not slow and not integration and not live"`,
   and each changed package's suite (see its README).

## 8. Style and formatting

1. All CI checks must pass. They are defined in `.github/workflows/`.
2. Always add type annotations when modifying code; CI runs mypy with `disallow_untyped_defs`
   on `mermaidseg/`, `scripts/` and the package sources.
3. Docstrings must be Google style (do not duplicate type annotations and defaults). CI checks
   them with ruff outside the test directories.
4. Comments and docstrings that only make sense to the current PR's author or reviewer (for example
   `# per reviewer comment on PR #NNNN`, `# as discussed in review`) state the *reason* so they
   stand alone, or are dropped. No debug printouts.
5. Don't make trivial (1-2 LOC) helper functions that are only used once unless it significantly
   improves code readability.
6. Avoid overly defensive code, e.g. checking that an argument is not `None` when `None` was not a
   valid argument type to begin with.
7. Keep state explicit: a class definition declares all its members. Don't dynamically `setattr` a
   field on an object and then dynamically `getattr` the field on the object.
8. Do not guess user intent and silently correct behavior. Make the expected inputs clear in the
   docstring, and raise a concise error for unsupported cases rather than adding complex fallback
   logic.
9. If uncertain, choose the simpler, more concise implementation.

## 9. Dependencies and security (public repository)

1. Never use system `python3` or bare `pip`/`pip install`. All Python commands go through `uv`
   (`uv run`, `uv sync --locked`, `uv add`).
2. Never add a mandatory dependency without the human's approval.
3. Import an extra's dependencies lazily, at the point of use, so the core install works without
   them.
4. Keep dependencies current. Don't pin an old version to avoid a breaking change without a tracked
   follow-up. Security advisories take priority over feature work.
5. Never put credentials, tokens or account-specific identifiers in code, configs, notebook outputs
   or logs. Secrets always come from the environment or the git-ignored `.env`.
6. Always load checkpoints with `torch.load(weights_only=True)`, third-party weights included. Never
   use `weights_only=False` except for a resume checkpoint this project's own training wrote, with
   a comment at the call site that says so.
7. Always use SafeTensors for an artifact that is published or crosses a trust boundary. Never set
   `trust_remote_code=True`. Always parse YAML with `yaml.safe_load`.
8. GitHub Actions dependencies must be locked with a workflow lockfile. If the repository cannot
   adopt one, each action must be pinned to a full-length commit SHA verified to come from the
   action's repository, not a fork.
9. The default `GITHUB_TOKEN` permission must be read-only for repository contents; raise it per
   job only as required.
10. Use `pull_request_target` and `workflow_run` only when the workflow needs the privileged
    context. Workflows they trigger must not check out untrusted code, including code from pull
    request forks.
11. The workflow security scan (actionlint and zizmor) must stay in place.
12. Container images must build from the locked environment and must carry no secrets.
13. Serving-side and API-boundary security belong to `mermaid-inference`; defer to it. The public
    demo must load only artifacts this project publishes, never user-supplied weights.
14. Never commit data, weights, large binaries, generated preview images or OS files (`.DS_Store`).

## 10. Where to look things up

Where a doc disagrees with this file or the code, this file and the code win.

| Need | Where |
|---|---|
| Test and lint commands, architecture overview | `README.md`: Running Tests, Common Commands, Architecture overview |
| Setup, data, training (Slurm, SageMaker), export and evaluation procedures | `docs/runbooks/` |
| The CoralNet scraper: S3 layout, environment variables, running and testing it | `packages/coralnet-scraper/README.md` |
| The CoralNet dataset build and the external datasets (Catlin, Moorea, Pacific, Benthos): outputs, completeness checks, environment variables, output contract, ingesting, building, publishing and testing them; the taxonomy snapshot; the split build and leakage commands | `packages/segdata-etl/README.md` |
| The versioned dataset contract (manifest, S3 and cache layout, publish, sync, split-file format, errors), and its tests | `packages/segdata-spec/README.md` |
| Recreate or update the CoralNet data end to end | `docs/runbooks/coralnet-data.md` |
| Lint, type-check, test and dependency settings; CI checks; pre-commit hooks | `pyproject.toml`; `.github/workflows/`; `.pre-commit-config.yaml` |
| Source and history of each rule, for editing this file | `docs/agents-md-sources.md` |

<!-- Adapted from huggingface/{diffusers,transformers,peft,pytorch-image-models,lerobot} and
vllm-project/vllm (Apache-2.0), pytorch/pytorch (BSD-style), agentsmd/agents.md (MIT), Google's Rules
of ML and GitHub Docs (CC BY 4.0), the PyTorch docs, and "The ML Test Score" and "Hidden Technical
Debt in Machine Learning Systems" (title only). Sources and deltas: docs/agents-md-sources.md -->
