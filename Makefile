.PHONY: sync logs lcc-log check kernel

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
