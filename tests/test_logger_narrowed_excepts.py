"""Seam 2: Logger swallows only expected MLflow/IO failures — real bugs surface.

The logging try/excepts were `except Exception: warning(...)`, which silently swallowed
programming errors (the dropped/misnamed-metric class of bug). Metadata-logging sites
now catch only (MlflowException, OSError), so a genuine bug in a logging call propagates
instead of degrading observability unnoticed.

Artifact-upload sites (save_model_checkpoint / concept matrix) additionally tolerate
transient S3 failures: mlflow's S3ArtifactRepository calls boto3 upload_file unwrapped,
so a throttle/ timeout blip surfaces as a boto exception (not MlflowException/OSError).
Those must stay swallowed — losing a whole training run to a transient checkpoint-upload
hiccup is the regression we're guarding against — while a real bug in the upload path
still propagates.
"""

from __future__ import annotations

import boto3.exceptions
import botocore.exceptions
import mlflow
import pytest

from mermaidseg.logger import Logger


def _enabled_logger(make_config, fake_meta_model) -> Logger:
    lgr = Logger(config=make_config(), meta_model=fake_meta_model, enable_mlflow=True)
    assert lgr.enabled is True
    return lgr


def test_log_propagates_programming_error(
    tmp_mlflow_uri, make_config, fake_meta_model, monkeypatch
):
    """A non-MLflow error (e.g. ValueError) in a logging call must NOT be swallowed."""
    lgr = _enabled_logger(make_config, fake_meta_model)

    def boom(*_args, **_kwargs):
        raise ValueError("real bug, not a tracking failure")

    monkeypatch.setattr(mlflow, "log_metrics", boom)
    with pytest.raises(ValueError, match="real bug"):
        lgr.log({"train/loss": 1.0}, step=0)


def test_log_swallows_mlflow_exception(tmp_mlflow_uri, make_config, fake_meta_model, monkeypatch):
    """A transient MlflowException stays swallowed (warn + continue) — not fatal to
    training."""
    lgr = _enabled_logger(make_config, fake_meta_model)

    def boom(*_args, **_kwargs):
        raise mlflow.exceptions.MlflowException("transient tracking failure")

    monkeypatch.setattr(mlflow, "log_metrics", boom)
    lgr.log({"train/loss": 1.0}, step=0)  # must not raise


@pytest.mark.parametrize(
    "exc",
    [
        boto3.exceptions.S3UploadFailedError("throttled while uploading checkpoint"),
        botocore.exceptions.EndpointConnectionError(endpoint_url="s3://artifacts"),
    ],
    ids=["s3_upload_failed", "endpoint_connection"],
)
def test_save_checkpoint_swallows_transient_s3_error(
    tmp_mlflow_uri, make_config, fake_meta_model, monkeypatch, exc
):
    """A transient S3 failure during checkpoint artifact upload must NOT crash training.

    Under the SageMaker managed backend these surface as boto exceptions (not
    MlflowException/OSError); the artifact sites tolerate them so a blip doesn't kill a
    run.
    """
    lgr = Logger(
        config=make_config(logger={"save_local_checkpoints": False}),
        meta_model=fake_meta_model,
        enable_mlflow=True,
    )
    assert lgr.enabled is True

    def boom(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(mlflow, "log_artifact", boom)
    # is_best=False -> single checkpoints/ upload path; must warn-and-continue, not raise.
    lgr.save_model_checkpoint(
        fake_meta_model, epoch=0, metrics_dict={"accuracy": 0.5}, is_best=False
    )


def test_save_checkpoint_propagates_programming_error(
    tmp_mlflow_uri, make_config, fake_meta_model, monkeypatch
):
    """A real bug (e.g. ValueError) in the checkpoint upload path must still
    propagate."""
    lgr = Logger(
        config=make_config(logger={"save_local_checkpoints": False}),
        meta_model=fake_meta_model,
        enable_mlflow=True,
    )
    assert lgr.enabled is True

    def boom(*_args, **_kwargs):
        raise ValueError("real bug in upload path")

    monkeypatch.setattr(mlflow, "log_artifact", boom)
    with pytest.raises(ValueError, match="real bug"):
        lgr.save_model_checkpoint(
            fake_meta_model, epoch=0, metrics_dict={"accuracy": 0.5}, is_best=False
        )
