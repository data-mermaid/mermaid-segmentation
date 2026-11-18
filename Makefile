.PHONY: sm-require-env sm-validate sm-dry-run sm-launch sm-smoke \
	sm-require-job sm-jobs sm-status sm-logs sm-errors sm-metrics

# --- SageMaker TrainingJob: backup to Slurm; same trainer command ---
# Images come only from the sagemaker-image CI workflow. Account values (role, bucket,
# registry, secret ids, profile) come from the git-ignored .env, never from this file.
SM_RUN_CONFIG ?= sagemaker/runs/cbm_final.yaml
SM_ENV_VARS = SM_ROLE_ARN SM_STAGING_BUCKET SM_ECR_REGISTRY HF_TOKEN_SECRET_ID \
	WANDB_API_KEY_SECRET_ID SM_AWS_PROFILE
SM_AWS_ENV = AWS_PROFILE=$(SM_AWS_PROFILE) AWS_DEFAULT_REGION=us-east-1
# Pass INIT_FROM_JOB=<training-job-name> for a run YAML that sets train.init_weights.
SM_LAUNCH = uv run python -m scripts.sagemaker.launch --run-config $(SM_RUN_CONFIG) \
	$(if $(INIT_FROM_JOB),--init-from-job $(INIT_FROM_JOB),)

sm-require-env:
	@missing=""; for v in $(SM_ENV_VARS); do \
		test -n "$$(printenv $$v)" || missing="$$missing $$v"; \
	done; \
	test -z "$$missing" || (echo "not set:$$missing (add to .env, see .env.example)" && exit 1)

# Offline validation of a run YAML; needs no environment and makes no AWS call.
sm-validate:
	$(SM_LAUNCH) --check

sm-dry-run: sm-require-env
	$(SM_AWS_ENV) $(SM_LAUNCH) --dry-run

sm-launch: sm-require-env
	$(SM_AWS_ENV) $(SM_LAUNCH)

sm-smoke:
	bash docker/jobs/local_smoke.sh

# --- SageMaker job monitoring: read-only ---
# Pass JOB=<training-job-name> (printed at launch, or see `make sm-jobs`).
# Uses SM_AWS_PROFILE when set; otherwise inherits ambient AWS credentials.
SM_REGION ?= us-east-1
SM_LOG_GROUP ?= /aws/sagemaker/TrainingJobs
SM_DEBUG_ENV = AWS_DEFAULT_REGION=$(SM_REGION) $(if $(SM_AWS_PROFILE),AWS_PROFILE=$(SM_AWS_PROFILE),)

sm-require-job:
	@test -n "$(JOB)" || (echo "JOB not set: pass JOB=<training-job-name> (see make sm-jobs)" && exit 1)

# The 10 most recent training jobs and their status.
sm-jobs:
	$(SM_DEBUG_ENV) aws sagemaker list-training-jobs \
		--sort-by CreationTime --sort-order Descending --max-results 10 \
		--query 'TrainingJobSummaries[].{Name:TrainingJobName,Status:TrainingJobStatus,Created:CreationTime}' \
		--output table

# FailureReason records the container exit, which the CloudWatch logs may not show.
sm-status: sm-require-job
	$(SM_DEBUG_ENV) aws sagemaker describe-training-job \
		--training-job-name $(JOB) \
		--query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Failure:FailureReason,Created:CreationTime,End:TrainingEndTime}' \
		--output table

sm-logs: sm-require-job
	$(SM_DEBUG_ENV) aws logs tail $(SM_LOG_GROUP) \
		--log-stream-name-prefix $(JOB)/ --follow

# Error, traceback, OOM, disk-full and silent-kill lines of a job's logs.
sm-errors: sm-require-job
	$(SM_DEBUG_ENV) aws logs filter-log-events \
		--log-group-name $(SM_LOG_GROUP) \
		--log-stream-name-prefix $(JOB) \
		--filter-pattern '?Error ?Traceback ?Exception ?"exit code" ?"killed by signal" ?"No space" ?"out of memory" ?MemoryError' \
		--query 'events[].message' --output text

# Peak disk, RAM, GPU memory and CPU over the job's lifetime, in 5-minute buckets.
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
