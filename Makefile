.PHONY: sync logs lcc-log check kernel \
	sm-sync sm-check sm-dry-run sm-launch sm-smoke sm-require-env \
	sm-jobs sm-status sm-logs sm-errors sm-metrics sm-require-job

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

# --- SageMaker job debugging / monitoring (see wiki/SageMaker-Jobs.md) ---
# Read-only. Pass JOB=<training-job-name> (the run-id printed at launch, or `make sm-jobs`).
# Uses SM_AWS_PROFILE when set (as the launch targets do); otherwise inherits ambient AWS creds.
SM_REGION ?= us-east-1
SM_LOG_GROUP ?= /aws/sagemaker/TrainingJobs
SM_DEBUG_ENV = AWS_DEFAULT_REGION=$(SM_REGION) $(if $(SM_AWS_PROFILE),AWS_PROFILE=$(SM_AWS_PROFILE),)

sm-require-job:
	@test -n "$(JOB)" || (echo "JOB not set — pass JOB=<training-job-name> (see: make sm-jobs)" && exit 1)

# List the 10 most recent training jobs and their status.
sm-jobs:
	$(SM_DEBUG_ENV) aws sagemaker list-training-jobs \
		--sort-by CreationTime --sort-order Descending --max-results 10 \
		--query 'TrainingJobSummaries[].{Name:TrainingJobName,Status:TrainingJobStatus,Created:CreationTime}' \
		--output table

# Status + failure reason for one job. FailureReason is where SageMaker records the
# container exit ("AlgorithmError: exit code 1"), which the CloudWatch logs may not show.
sm-status: sm-require-job
	$(SM_DEBUG_ENV) aws sagemaker describe-training-job \
		--training-job-name $(JOB) \
		--query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Failure:FailureReason,Created:CreationTime,End:TrainingEndTime}' \
		--output table

# Tail (follow) a job's CloudWatch logs.
sm-logs: sm-require-job
	$(SM_DEBUG_ENV) aws logs tail $(SM_LOG_GROUP) \
		--log-stream-name-prefix $(JOB)/ --follow

# Grep a job's logs for errors / tracebacks / OOM / disk-full / silent-kill signatures.
sm-errors: sm-require-job
	$(SM_DEBUG_ENV) aws logs filter-log-events \
		--log-group-name $(SM_LOG_GROUP) \
		--log-stream-name-prefix $(JOB) \
		--filter-pattern '?Error ?Traceback ?Exception ?"exit code" ?"killed by signal" ?"No space" ?"out of memory" ?MemoryError' \
		--query 'events[].message' --output text

# Resource utilization over a job's lifetime — the fastest way to tell a silent death apart
# (disk-full vs RAM OOM vs GPU OOM). Window is derived from the job's own start/end time.
sm-metrics: sm-require-job
	@START=$$($(SM_DEBUG_ENV) aws sagemaker describe-training-job --training-job-name $(JOB) --query 'CreationTime' --output text); \
	END=$$($(SM_DEBUG_ENV) aws sagemaker describe-training-job --training-job-name $(JOB) --query 'TrainingEndTime' --output text); \
	if [ "$$END" = "None" ] || [ -z "$$END" ]; then END=$$(date -u +%Y-%m-%dT%H:%M:%SZ); fi; \
	for M in DiskUtilization MemoryUtilization GPUMemoryUtilization CPUUtilization; do \
		echo "=== $$M (max, 5-min buckets) ==="; \
		$(SM_DEBUG_ENV) aws cloudwatch get-metric-statistics \
			--namespace /aws/sagemaker/TrainingJobs --metric-name $$M \
			--dimensions Name=Host,Value=$(JOB)/algo-1 \
			--start-time "$$START" --end-time "$$END" \
			--period 300 --statistics Maximum \
			--query 'Datapoints | sort_by(@,&Timestamp)[].[Timestamp,Maximum]' --output text; \
	done
