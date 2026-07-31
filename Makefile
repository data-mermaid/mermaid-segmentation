.PHONY: sync logs lcc-log check kernel \
	train-local train-local-taxonomical train-local-dual compare-local compare-local-coralnet \
	sm-sync sm-check sm-dry-run sm-launch sm-smoke sm-require-env

# Re-sync the uv environment and Jupyter kernel from the current branch.
# Equivalent to re-running the LCC without restarting the space.
sync:
	bash scripts/sync_env.sh

# Register the Jupyter kernel only (no git pull, no uv sync).
# Useful after a full uv sync already ran.
kernel:
	uv run python -m ipykernel install \
		--user \
		--name=mermaid-seg \
		--display-name "Python (mermaid-seg)"

# Tail training logs. Requires at least one logs/train_*.log file to exist.
logs:
	tail -f logs/train_*.log

# Tail the LCC startup log on EFS.
# Shows progress of the background uv sync + kernel registration.
lcc-log:
	tail -f ~/lcc-setup.log

# Validate the notebook environment: env vars, AWS session, MLflow version.
check:
	uv run python -c "\
from nbs.nb_setup import check_env, check_aws_session, check_mlflow_version; \
check_env(); \
check_aws_session(); \
check_mlflow_version()"

# --- Local loss smoke test (see docs/local-loss-testing.md) ---
# Fully OFFLINE: builds the real model (cached DINOv3 backbone) + the real taxonomical/dual loss +
# hierarchy eval over synthetic batches, and runs a few train steps. No S3, no creds, no network —
# exercises exactly the wiring unit tests miss (model-output -> loss shapes, buffer/device
# placement, the new loss components + hierarchy metrics). HF_HUB_OFFLINE avoids any hub lookup.
LOCAL_MODEL_CONFIG    ?= configs/model_config_dinov3_lora_qv_r8.yaml
LOCAL_TRAINING_CONFIG ?= configs/training_config_dinov3_lora_taxonomical.yaml

# Generic: override LOCAL_MODEL_CONFIG / LOCAL_TRAINING_CONFIG to smoke any model + loss combo.
train-local:
	HF_HUB_OFFLINE=1 uv run python scripts/diagnostics/local_loss_smoke.py \
		--model-config $(LOCAL_MODEL_CONFIG) \
		--training-config $(LOCAL_TRAINING_CONFIG)

# TaxonomicalLoss (CE + tree-distance + level CE) on the LoRA q/v model.
train-local-taxonomical: LOCAL_MODEL_CONFIG    = configs/model_config_dinov3_lora_qv_r8.yaml
train-local-taxonomical: LOCAL_TRAINING_CONFIG = configs/training_config_dinov3_lora_taxonomical.yaml
train-local-taxonomical: train-local

# DualTaxonomicalLoss (adds masked morphology BCE) on the dual head.
train-local-dual: LOCAL_MODEL_CONFIG    = configs/model_config_dinov3_dual_lora_qv_r8.yaml
train-local-dual: LOCAL_TRAINING_CONFIG = configs/training_config_dinov3_dual_taxonomical.yaml
train-local-dual: train-local

# Controlled offline comparison: CE baseline vs Taxonomical vs Dual on a structured synthetic task
# where the hierarchy is real (siblings look alike). A mechanism demo, NOT a coral-quality verdict.
COMPARE_ARGS ?= --steps 80 --noise 1.2
compare-local:
	HF_HUB_OFFLINE=1 uv run python scripts/diagnostics/local_loss_compare.py $(COMPARE_ARGS)

# Same CE-vs-Taxonomical-vs-Dual comparison, but on the REAL downloaded CoralNet subset
# (data/coralnet_local_subset) instead of synthetic patches: real reef photos, real benthic
# hierarchy, real coral-genus confusion. Loss weights default to the taxonomical training config.
# Needs the subset on disk; fetches + caches the public benthic hierarchy on first run (no creds).
COMPARE_CORALNET_ARGS ?= --n-train 20 --n-val 10 --steps 60
compare-local-coralnet:
	HF_HUB_OFFLINE=1 uv run python scripts/diagnostics/local_loss_compare_coralnet.py $(COMPARE_CORALNET_ARGS)

# --- SageMaker TrainingJob (see wiki/SageMaker-Jobs.md) ---
# Account ARNs in .env (gitignored), loaded by direnv. Login: aws sso login --profile wcs-sso
SM_CONFIG_DIR ?= sagemaker/configs/example
SM_RUN_CONFIG ?= sagemaker/runs/example-training.yaml

# Read from environment (.env + direnv). Profile name is not a secret.
SM_AWS_ENV = AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) AWS_PROFILE=$(SM_AWS_PROFILE)

sm-require-env:
	@test -n "$$SM_ROLE_ARN" || (echo "SM_ROLE_ARN not set — add to .env (see .env.example)" && exit 1)
	@test -n "$$MLFLOW_TRACKING_URI" || (echo "MLFLOW_TRACKING_URI not set — add to .env (see .env.example)" && exit 1)
	@test -n "$$SM_AWS_PROFILE" || (echo "SM_AWS_PROFILE not set — add to .env (e.g. wcs-launcher)" && exit 1)

sm-sync:
	uv sync --extra sagemaker

sm-check: sm-require-env
	$(SM_AWS_ENV) uv run --extra sagemaker python scripts/check_sagemaker_env.py \
		--role-arn $(SM_ROLE_ARN) \
		--check-hf-token

sm-dry-run: sm-require-env
	$(SM_AWS_ENV) uv run --extra sagemaker python scripts/launch_training.py \
		--run-config $(SM_RUN_CONFIG) \
		--config-dir $(SM_CONFIG_DIR)/ \
		--mlflow-tracking-uri $(MLFLOW_TRACKING_URI) \
		--role-arn $(SM_ROLE_ARN) \
		--dry-run

sm-launch: sm-require-env
	$(SM_AWS_ENV) uv run --extra sagemaker python scripts/launch_training.py \
		--run-config $(SM_RUN_CONFIG) \
		--config-dir $(SM_CONFIG_DIR)/ \
		--mlflow-tracking-uri $(MLFLOW_TRACKING_URI) \
		--role-arn $(SM_ROLE_ARN) \
		$(if $(HF_TOKEN),--hf-token $(HF_TOKEN),)

sm-smoke:
	bash docker/jobs/local_smoke.sh training
