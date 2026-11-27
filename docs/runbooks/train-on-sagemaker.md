# Train on SageMaker

This runbook trains the final model as SageMaker training jobs. SageMaker is the backup to [Slurm](train-on-slurm.md). It runs the same `scripts/train.py` with the same training arguments as `slurm/launch_final.sh`, except that the run YAMLs log to MLflow (`--logger-config configs/logger_config_sagemaker.yaml`).

SageMaker jobs are on-demand and do not resume. If a job fails, launch it again from the start.

Before it starts the trainer, the job entrypoint runs `scripts/sync_data.py` with the run's `--data-config` into `/opt/ml/cache`. The job downloads and verifies the pinned dataset versions from `mermaid-ic-datasets` on each start. Training then reads only that cache. The job is online, so it downloads the Hugging Face encoder at start.

- [Before you start](#before-you-start)
- [Build the training image](#build-the-training-image)
- [Check the run config](#check-the-run-config)
- [Launch cbm_final](#launch-cbm_final)
- [Launch the 768 px fine-tune](#launch-the-768-px-fine-tune)
- [Follow a job](#follow-a-job)
- [Get the checkpoints](#get-the-checkpoints)

## Before you start

- Complete [Set up a development environment](set-up-dev-environment.md), with the AWS CLI.
- Make sure the account has the IAM grants, the Hugging Face secret and the GitHub environment. They are listed under Prerequisites in [docker/jobs/README.md](../../docker/jobs/README.md#prerequisites).
- Make sure the job execution role can read `mermaid-ic-datasets`, the bucket that the data sync downloads from.
- Raise the account quota "Longest run time for a training job" to at least 480 h. `cbm_final` needs 480 h and `cbm_final_res768` needs 168 h. The default is 5 days.
- Set the SageMaker variables in `.env`. They are in the "SageMaker launcher" block of `.env.example`:

  | Variable | What it is |
  |---|---|
  | `SM_ROLE_ARN` | The IAM role that the training job assumes |
  | `SM_STAGING_BUCKET` | The S3 bucket for job output and checkpoints |
  | `SM_ECR_REGISTRY` | The ECR registry that holds the training image |
  | `SM_AWS_PROFILE` | The AWS profile that the launcher uses to submit jobs |
  | `HF_TOKEN_SECRET_ID` | The Secrets Manager secret that holds the Hugging Face token |
  | `SM_MLFLOW_TRACKING_ARN` | The ARN of the SageMaker MLflow App that the job logs to |

- Load `.env` into your shell. The Makefile reads these variables from the environment:

  ```bash
  set -a; source .env; set +a
  ```

## Build the training image

A job runs the code and `configs/` that are in its image. Only the run YAML, and for an init run the init checkpoint, travel with the job. Merge every code and config change to `main` before you build.

1. Open the repository on GitHub and go to **Actions**.
2. Select the `sagemaker-image` workflow.
3. Click **Run workflow** on `main`. The `sagemaker-image` environment allows only `main`, so a run from another branch fails.

Each workflow run pushes the tag `training-<sha>` and moves `training-latest`. The run YAMLs use `training-latest`. See [docker/jobs/README.md](../../docker/jobs/README.md#build) for how the image is built.

To check the image locally before you merge, run `make sm-smoke`. It builds a local image that is never pushed, and runs `launch --check` on each run YAML inside it.

## Check the run config

The run YAMLs are in `sagemaker/runs/`:

| Run YAML | Run |
|---|---|
| `cbm_final.yaml` | 256 px, the default `SM_RUN_CONFIG` |
| `cbm_final_res768.yaml` | 768 px fine-tune from a finished `cbm_final` job |

1. Check the run YAML offline:

   ```bash
   make sm-validate
   ```

   It checks only your local checkout, not the image.

2. Show the job request without submitting it:

   ```bash
   make sm-dry-run
   ```

   A dry run makes no AWS call and shows the image tag.

## Launch cbm_final

1. Submit the job:

   ```bash
   make sm-launch
   ```

2. Record the job name from the output. You need it to follow the job and to launch the fine-tune.

At launch, the launcher resolves `job.image` (`repo:tag`) to its ECR digest, submits `repo@sha256:...` and prints it. The job keeps that exact image even if `training-latest` moves later. A `job.image` that is already a digest is used as it is.

## Launch the 768 px fine-tune

The 768 px run YAML sets `train.init_weights`, so every command for it needs `SM_RUN_CONFIG` and `INIT_FROM_JOB`.

1. Wait for the `cbm_final` job to be `Completed`.
2. Check the run YAML:

   ```bash
   make sm-validate SM_RUN_CONFIG=sagemaker/runs/cbm_final_res768.yaml INIT_FROM_JOB=<cbm_final job name>
   ```

3. Submit the job:

   ```bash
   make sm-launch SM_RUN_CONFIG=sagemaker/runs/cbm_final_res768.yaml INIT_FROM_JOB=<cbm_final job name>
   ```

The launcher refuses to launch if the source job is not `Completed`, or if its init checkpoint is not readable in S3.

## Follow a job

These targets need the AWS CLI. All of them except `sm-jobs` need `JOB=<training-job-name>`.

| Target | What it shows |
|---|---|
| `make sm-jobs` | The 10 most recent training jobs and their status |
| `make sm-status JOB=<name>` | The status and failure reason of one job |
| `make sm-logs JOB=<name>` | The job logs, followed live |
| `make sm-errors JOB=<name>` | The error lines from the job logs |
| `make sm-metrics JOB=<name>` | The peak disk, RAM, GPU memory and CPU of the job |

Metrics go to the MLflow App named by `SM_MLFLOW_TRACKING_ARN`.

The job log shows the sync output, one line per dataset, before training starts. The first lines also show the free disk space for `/opt/ml/cache` and `/opt/ml/checkpoints`. Each run YAML sets `volume_gb: 1500` for the synced datasets and the checkpoints.

If you stop a multi-GPU job, the MLflow run can stay `RUNNING`. See [docker/jobs/README.md](../../docker/jobs/README.md) for why.

## Get the checkpoints

Checkpoints sync to:

```text
s3://$SM_STAGING_BUCKET/runs/<job>/checkpoints/model_checkpoints/<run>/
```

The directory holds `model_step*`, `model_final`, `model.json`, `config.yaml` and the `label_space/` copy. Download the checkpoint you want together with `model.json` and `label_space/`, then [export it](export-a-model.md). The `git_sha` in `model.json` and the saved `config.yaml` record what the job ran.
