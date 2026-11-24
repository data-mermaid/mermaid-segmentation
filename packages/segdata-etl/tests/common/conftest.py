"""Fixtures for the common-helper tests: a recording in-memory S3 client."""

from __future__ import annotations

import pytest
from s3_fake import FakeS3

__all__ = ["FakeS3"]


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
