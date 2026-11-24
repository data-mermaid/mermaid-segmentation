"""Fixtures shared by every segdata-etl test: AWS-free credentials and the in-memory S3 fake.

The in-memory ``FakeS3`` comes from ``s3_fake``; it keeps the suite free of network and AWS deps.
"""

from __future__ import annotations

import pytest
from s3_fake import FakeS3


@pytest.fixture(autouse=True)
def _fake_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give boto3 static credentials so it never queries the EC2 metadata endpoint."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
