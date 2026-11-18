"""Host-side SageMaker launcher: validation, request shape, dry-run and submit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

import scripts.sagemaker.launch as launch
from mermaidseg.io import load_section, resolve_repo_path
from scripts.sagemaker.run_config import RunConfig, load_run_config

_RUNS = Path(__file__).resolve().parents[3] / "sagemaker" / "runs"
_CBM = _RUNS / "cbm_final.yaml"
_RES768 = _RUNS / "cbm_final_res768.yaml"
_SOURCE_JOB = "mermaidseg-cbm-final-20260101T000000Z-abc123"

_ENV_VARS = (
    "SM_ROLE_ARN",
    "SM_STAGING_BUCKET",
    "SM_ECR_REGISTRY",
    "HF_TOKEN_SECRET_ID",
    "SM_MLFLOW_TRACKING_ARN",
)
_FAKE_ENV = {
    "SM_ROLE_ARN": "arn:fake:role",
    "SM_STAGING_BUCKET": "fake-bucket",
    "SM_ECR_REGISTRY": "123.dkr.ecr.fake.amazonaws.com",
    "HF_TOKEN_SECRET_ID": "fake/hf",
    "SM_MLFLOW_TRACKING_ARN": "arn:fake:mlflow-app/app-x",
}


@pytest.fixture
def no_sm_env(monkeypatch):
    for var in (*_ENV_VARS, "SM_AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_env(monkeypatch):
    for var, value in _FAKE_ENV.items():
        monkeypatch.setenv(var, value)


def _config(path: Path = _CBM, **train: Any) -> RunConfig:
    cfg = load_run_config(path)
    return cfg.model_copy(update={"train": cfg.train.model_copy(update=train)})


def _write(tmp_path: Path, cfg: RunConfig) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(json.dumps(cfg.model_dump()))
    return path


# ---------- --check ----------


@pytest.mark.parametrize(
    ("path", "extra"),
    [(_CBM, []), (_RES768, ["--init-from-job", _SOURCE_JOB])],
)
def test_committed_run_yamls_pass_check_without_env(no_sm_env, capsys, path, extra):
    assert launch.main(["--run-config", str(path), "--check", *extra]) == 0
    assert capsys.readouterr().out == f"OK: {path}\n"


@pytest.mark.parametrize("path", sorted(_RUNS.glob("*.yaml")), ids=lambda p: p.stem)
def test_committed_run_yamls_select_the_mlflow_tracker(path):
    namespace = launch._parse_trainer_args(load_run_config(path))
    section = load_section(resolve_repo_path(namespace.logger_config), "logger")
    assert section["tracker"] == "mlflow"


def test_check_failure_exits_with_error_message(no_sm_env):
    with pytest.raises(SystemExit, match="^ERROR: .*--init-from-job"):
        launch.main(["--run-config", str(_RES768), "--check"])


# ---------- validate_run ----------


def test_init_weights_without_init_from_job_raises():
    with pytest.raises(ValueError, match=r"init_weights.*--init-from-job"):
        launch.validate_run(load_run_config(_RES768), None)


def test_init_from_job_without_init_weights_raises():
    with pytest.raises(ValueError, match=r"init_weights.*--init-from-job"):
        launch.validate_run(load_run_config(_CBM), _SOURCE_JOB)


@pytest.mark.parametrize("job", ["-starts-with-dash", "has_underscore", "a" * 64])
def test_invalid_init_from_job_name_raises(job):
    with pytest.raises(ValueError, match="job name"):
        launch.validate_run(load_run_config(_RES768), job)


def test_owned_flag_in_train_args_fails_validation():
    cfg = _config(args=["--output-dir", "/x"])
    with pytest.raises(ValueError, match="--output-dir"):
        launch.validate_run(cfg, None)


def test_unknown_trainer_flag_fails_as_value_error():
    cfg = _config(args=["--no-such-flag", "1"])
    with pytest.raises(ValueError, match="--no-such-flag"):
        launch.validate_run(cfg, None)


def test_nonexistent_config_path_fails_naming_it():
    cfg = _config(args=["--data-config", "configs/missing.yaml", "--logger-config", "nope.yaml"])
    with pytest.raises(ValueError, match=r"configs/missing\.yaml.*nope\.yaml"):
        launch.validate_run(cfg, None)


# ---------- build_request ----------


def test_build_request_for_res768():
    request = launch.build_request(
        load_run_config(_RES768), "mermaidseg-cbm-final-res768-job", _FAKE_ENV, _SOURCE_JOB
    )
    assert request == {
        "TrainingJobName": "mermaidseg-cbm-final-res768-job",
        "RoleArn": "arn:fake:role",
        "AlgorithmSpecification": {
            "TrainingImage": (
                "123.dkr.ecr.fake.amazonaws.com/mermaid-segmentation-jobs:training-latest"
            ),
            "TrainingInputMode": "File",
        },
        "ResourceConfig": {
            "InstanceType": "ml.g6e.12xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 500,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 604800},
        "OutputDataConfig": {
            "S3OutputPath": "s3://fake-bucket/runs/mermaidseg-cbm-final-res768-job/output/"
        },
        "CheckpointConfig": {
            "S3Uri": "s3://fake-bucket/runs/mermaidseg-cbm-final-res768-job/checkpoints/",
            "LocalPath": "/opt/ml/checkpoints",
        },
        "InputDataConfig": [
            {
                "ChannelName": "config",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://fake-bucket/runs/mermaidseg-cbm-final-res768-job/config/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "init",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": (
                            "s3://fake-bucket/runs/mermaidseg-cbm-final-20260101T000000Z-abc123/"
                            "checkpoints/model_checkpoints/cbm_final/model_step300000"
                        ),
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        "Environment": {
            "HF_TOKEN_SECRET_ID": "fake/hf",
            "MLFLOW_TRACKING_URI": "arn:fake:mlflow-app/app-x",
        },
    }


def test_build_request_without_init_weights_has_only_the_config_channel():
    request = launch.build_request(load_run_config(_CBM), "job", _FAKE_ENV, None)
    assert [ch["ChannelName"] for ch in request["InputDataConfig"]] == ["config"]


def test_full_ecr_uri_in_job_image_raises():
    cfg = load_run_config(_CBM)
    job = cfg.job.model_copy(update={"image": "1.dkr.ecr.us-east-1.amazonaws.com/repo:tag"})
    with pytest.raises(ValueError, match="repository:tag|<repo>:<tag>"):
        launch.build_request(cfg.model_copy(update={"job": job}), "job", _FAKE_ENV, None)


# ---------- env, dry-run, submit ----------


@pytest.mark.parametrize("mode", [["--dry-run"], []])
@pytest.mark.parametrize("missing", _ENV_VARS)
def test_missing_env_var_fails_naming_it(fake_env, monkeypatch, mode, missing):
    monkeypatch.setenv(missing, "")
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("no AWS call"))
    with pytest.raises(SystemExit, match=missing):
        launch.main(["--run-config", str(_CBM), *mode])


def test_dry_run_prints_the_request_without_boto3(fake_env, monkeypatch, capsys):
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("no AWS call"))
    monkeypatch.setattr(launch, "make_run_id", lambda prefix: "fixed-job")

    assert launch.main(["--run-config", str(_CBM), "--dry-run"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["TrainingJobName"] == "fixed-job"
    assert printed["RoleArn"] == "arn:fake:role"


_DIGEST = "sha256:" + "a" * 64
_IMAGE_REPO = "mermaid-segmentation-jobs"


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "Operation")


class _FakeECR:
    def __init__(self, calls: list[str], error: Exception | None = None) -> None:
        self.calls = calls
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def describe_images(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("ecr.describe_images")
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"imageDetails": [{"imageDigest": _DIGEST}]}


class _FakeS3:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.uploads: list[tuple[str, str, str]] = []
        self.heads: list[dict[str, Any]] = []
        self.head_error: Exception | None = None
        self.upload_error: Exception | None = None

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("s3.head_object")
        self.heads.append(kwargs)
        if self.head_error is not None:
            raise self.head_error
        return {}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.calls.append("s3.upload_file")
        if self.upload_error is not None:
            raise self.upload_error
        self.uploads.append((filename, bucket, key))


class _FakeSageMaker:
    def __init__(self, calls: list[str], s3: _FakeS3) -> None:
        self.calls = calls
        self.s3 = s3
        self.source_status = "Completed"
        self.described: list[str] = []
        self.created: list[dict[str, Any]] = []

    def describe_training_job(self, TrainingJobName: str) -> dict[str, str]:
        self.calls.append("sagemaker.describe_training_job")
        self.described.append(TrainingJobName)
        return {"TrainingJobStatus": self.source_status}

    def create_training_job(self, **kwargs: Any) -> dict[str, str]:
        assert self.s3.uploads, "config must be uploaded before the job is created"
        self.calls.append("sagemaker.create_training_job")
        self.created.append(kwargs)
        return {"TrainingJobArn": "arn:fake:job"}


class _Aws:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.ecr = _FakeECR(self.calls)
        self.s3 = _FakeS3(self.calls)
        self.sm = _FakeSageMaker(self.calls, self.s3)
        self.regions: list[str] = []


@pytest.fixture
def aws(fake_env, monkeypatch):
    harness = _Aws()
    clients = {"ecr": harness.ecr, "s3": harness.s3, "sagemaker": harness.sm}

    def fake_client(service: str, region_name: str) -> object:
        harness.regions.append(region_name)
        return clients[service]

    monkeypatch.setattr("boto3.client", fake_client)
    monkeypatch.setattr(launch, "make_run_id", lambda prefix: "fixed-job")
    return harness


def _pinned_request(path: Path, init_from_job: str | None) -> dict[str, Any]:
    request = launch.build_request(load_run_config(path), "fixed-job", _FAKE_ENV, init_from_job)
    request["AlgorithmSpecification"]["TrainingImage"] = (
        f"{_FAKE_ENV['SM_ECR_REGISTRY']}/{_IMAGE_REPO}@{_DIGEST}"
    )
    return request


def _assert_nothing_written(aws: _Aws) -> None:
    assert "s3.upload_file" not in aws.calls
    assert "sagemaker.create_training_job" not in aws.calls


def test_submit_pins_the_image_by_digest_then_uploads_then_creates(aws, capsys):
    assert launch.main(["--run-config", str(_CBM)]) == 0

    assert aws.ecr.requests == [
        {"repositoryName": _IMAGE_REPO, "imageIds": [{"imageTag": "training-latest"}]}
    ]
    assert aws.s3.uploads == [(str(_CBM), "fake-bucket", "runs/fixed-job/config/cbm_final.yaml")]
    assert aws.sm.created == [_pinned_request(_CBM, None)]
    assert aws.calls == ["ecr.describe_images", "s3.upload_file", "sagemaker.create_training_job"]
    assert set(aws.regions) == {"us-east-1"}
    out = capsys.readouterr().out
    assert f"Image: {_IMAGE_REPO}:training-latest -> {_DIGEST}\n" in out
    assert "fixed-job" in out
    assert "console.aws.amazon.com/sagemaker" in out
    assert "console.aws.amazon.com/cloudwatch" in out


def test_digest_form_image_passes_through_without_an_ecr_call(aws, tmp_path):
    cfg = load_run_config(_CBM)
    job = cfg.job.model_copy(update={"image": f"{_IMAGE_REPO}@{_DIGEST}"})
    path = _write(tmp_path, cfg.model_copy(update={"job": job}))

    assert launch.main(["--run-config", str(path)]) == 0

    assert aws.ecr.requests == []
    image = aws.sm.created[0]["AlgorithmSpecification"]["TrainingImage"]
    assert image == f"{_FAKE_ENV['SM_ECR_REGISTRY']}/{_IMAGE_REPO}@{_DIGEST}"


def test_unknown_image_tag_exits_naming_the_image_before_any_write(aws):
    aws.ecr.error = _client_error("ImageNotFoundException")

    with pytest.raises(SystemExit, match=r"^ERROR: .*mermaid-segmentation-jobs:training-latest"):
        launch.main(["--run-config", str(_CBM)])

    _assert_nothing_written(aws)


def test_res768_with_a_completed_source_creates_the_job_with_the_init_channel(aws):
    args = ["--run-config", str(_RES768), "--init-from-job", _SOURCE_JOB]

    assert launch.main(args) == 0

    assert aws.sm.described == [_SOURCE_JOB]
    assert aws.s3.heads == [
        {
            "Bucket": "fake-bucket",
            "Key": f"runs/{_SOURCE_JOB}/checkpoints/model_checkpoints/cbm_final/model_step300000",
        }
    ]
    assert aws.calls == [
        "ecr.describe_images",
        "sagemaker.describe_training_job",
        "s3.head_object",
        "s3.upload_file",
        "sagemaker.create_training_job",
    ]
    assert aws.sm.created == [_pinned_request(_RES768, _SOURCE_JOB)]


@pytest.mark.parametrize("status", ["InProgress", "Failed"])
def test_source_job_not_completed_exits_naming_job_and_status(aws, status):
    aws.sm.source_status = status
    args = ["--run-config", str(_RES768), "--init-from-job", _SOURCE_JOB]

    with pytest.raises(SystemExit, match=rf"^ERROR: .*{_SOURCE_JOB}.*{status}"):
        launch.main(args)

    _assert_nothing_written(aws)
    assert "s3.head_object" not in aws.calls


@pytest.mark.parametrize("code", ["404", "403"])
def test_unreadable_init_checkpoint_exits_naming_uri_and_code(aws, code):
    aws.s3.head_error = _client_error(code)
    args = ["--run-config", str(_RES768), "--init-from-job", _SOURCE_JOB]
    uri = (
        f"s3://fake-bucket/runs/{_SOURCE_JOB}/checkpoints/"
        "model_checkpoints/cbm_final/model_step300000"
    )

    with pytest.raises(SystemExit, match=rf"^ERROR: .*{re.escape(uri)}.*{code}"):
        launch.main(args)

    _assert_nothing_written(aws)


def test_failed_upload_does_not_create_the_job(aws):
    aws.s3.upload_error = RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        launch.main(["--run-config", str(_CBM)])

    assert "sagemaker.create_training_job" not in aws.calls


# ---------- make_run_id ----------


@pytest.mark.parametrize(
    "prefix", ["mermaidseg-cbm-final", "mermaidseg-cbm-final-res768", "a" * 39]
)
def test_make_run_id_is_a_valid_job_name(prefix):
    run_id = launch.make_run_id(prefix)
    assert re.fullmatch(rf"{prefix}-\d{{8}}T\d{{6}}Z-[0-9a-f]{{6}}", run_id)
    assert len(run_id) <= 63


def test_make_run_id_with_too_long_prefix_raises():
    with pytest.raises(ValueError, match="39"):
        launch.make_run_id("a" * 40)
