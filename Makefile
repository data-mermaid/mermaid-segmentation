.PHONY: help sync logs lcc-log check kernel

help:
	@echo "MermaidSeg Makefile targets:"
	@echo "  make sync     — git pull + uv sync --all-extras + register Jupyter kernel (SageMaker LCC)"
	@echo "  make kernel   — register Jupyter kernel only (uses --all-extras)"
	@echo "  make check    — validate env vars, AWS session, MLflow version (uses --all-extras)"
	@echo "  make logs     — tail logs/train_*.log"
	@echo "  make lcc-log  — tail ~/lcc-setup.log (LCC background sync progress)"

# Re-sync the uv environment and Jupyter kernel from the current branch.
# Equivalent to re-running the LCC without restarting the space.
sync:
	bash scripts/sync_env.sh

# Register the Jupyter kernel only (no git pull, no uv sync).
# --all-extras so ipykernel (now in the `notebooks` extra) is present and not
# pruned from the env by `uv run`.
kernel:
	uv run --all-extras python -m ipykernel install \
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
# --all-extras: nb_setup imports mlflow (now in the `training` extra).
check:
	uv run --all-extras python -c "\
from nbs.nb_setup import check_env, check_aws_session, check_mlflow_version; \
check_env(); \
check_aws_session(); \
check_mlflow_version()"
