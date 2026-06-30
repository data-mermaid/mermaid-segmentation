.PHONY: sync logs lcc-log check kernel \
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
