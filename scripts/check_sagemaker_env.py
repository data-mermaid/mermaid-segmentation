"""Preflight checks for SageMaker training job launch."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import NamedTuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class CheckResult(NamedTuple):
    name: str
    ok: bool
    detail: str
    fatal: bool = False


MIN_SAGEMAKER_VERSION = (2, 200, 0)
MIN_BOTO3_VERSION = (1, 26, 0)


def _parse_version(raw: str) -> tuple[int, ...]:
    """Parse version string into tuple of ints."""
    parts: list[int] = []
    for piece in raw.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def check_sagemaker_sdk() -> CheckResult:
    """Check SageMaker SDK is installed and recent enough."""
    try:
        installed = version("sagemaker")
    except PackageNotFoundError:
        return CheckResult(
            "sagemaker SDK",
            False,
            "not installed — run: uv sync --extra sagemaker",
            fatal=True,
        )
    if _parse_version(installed) < MIN_SAGEMAKER_VERSION:
        return CheckResult(
            "sagemaker SDK",
            False,
            f"found {installed}, need >= {'.'.join(map(str, MIN_SAGEMAKER_VERSION))}",
            fatal=True,
        )
    return CheckResult("sagemaker SDK", True, f"v{installed}")


def check_boto3() -> CheckResult:
    """Check boto3 is installed."""
    try:
        installed = version("boto3")
    except PackageNotFoundError:
        return CheckResult(
            "boto3",
            False,
            "not installed — run: pip install boto3",
            fatal=True,
        )
    if _parse_version(installed) < MIN_BOTO3_VERSION:
        return CheckResult(
            "boto3",
            False,
            f"found {installed}, need >= {'.'.join(map(str, MIN_BOTO3_VERSION))}",
            fatal=True,
        )
    return CheckResult("boto3", True, f"v{installed}")


def check_aws_credentials() -> CheckResult:
    """Check AWS credentials are available."""
    try:
        session = boto3.Session()
        creds = session.get_credentials()
        if not creds:
            return CheckResult(
                "AWS credentials",
                False,
                "not found — run: aws sso login --profile <profile>",
                fatal=True,
            )
        return CheckResult("AWS credentials", True, "available")
    except NoCredentialsError as e:
        return CheckResult(
            "AWS credentials",
            False,
            f"error: {e}",
            fatal=True,
        )


def check_sagemaker_role(role_arn: str) -> CheckResult:
    """Check SageMaker execution role ARN format (not verifying permissions)."""
    role_name = role_arn.split("/")[-1]
    if not role_arn.startswith("arn:aws:iam::"):
        return CheckResult(
            "SageMaker role",
            False,
            f"invalid ARN format: {role_arn}",
            fatal=True,
        )
    return CheckResult("SageMaker role", True, role_name)


def check_sagemaker_region() -> CheckResult:
    """Check SageMaker is available in the region."""
    try:
        session = boto3.Session()
        region = session.region_name
        if not region:
            return CheckResult(
                "AWS region",
                False,
                "not set — run: export AWS_DEFAULT_REGION=us-east-1",
                fatal=True,
            )
        sm = boto3.client("sagemaker", region_name=region)
        sm.list_training_jobs(MaxResults=1)
        return CheckResult("AWS region", True, region)
    except ClientError as e:
        return CheckResult(
            "AWS region",
            False,
            f"error: {e.response['Error']['Message']}",
            fatal=True,
        )


def check_hf_token(secret_arn: str | None) -> CheckResult:
    """Check HuggingFace token (if needed for model access)."""
    if not secret_arn:
        return CheckResult(
            "HuggingFace token",
            True,
            "optional (use --check-hf-token to validate)",
        )
    try:
        secrets = boto3.client("secretsmanager")
        secrets.get_secret_value(SecretId=secret_arn)
        return CheckResult("HuggingFace token", True, secret_arn)
    except ClientError as e:
        return CheckResult(
            "HuggingFace token",
            False,
            f"not found: {e.response['Error']['Message']}",
            fatal=False,
        )


def main():
    """Run all checks and report results."""
    parser = argparse.ArgumentParser(
        description="Preflight checks for SageMaker training job launch"
    )
    parser.add_argument(
        "--role-arn",
        required=True,
        help="SageMaker execution role ARN (from SM_ROLE_ARN env var)",
    )
    parser.add_argument(
        "--hf-token-secret-arn",
        help="Optional AWS Secrets Manager ARN for HuggingFace token",
    )
    parser.add_argument(
        "--check-hf-token",
        action="store_true",
        help="Require HuggingFace token to be present",
    )
    args = parser.parse_args()

    checks = [
        check_sagemaker_sdk(),
        check_boto3(),
        check_aws_credentials(),
        check_sagemaker_region(),
        check_sagemaker_role(args.role_arn),
    ]

    if args.check_hf_token or args.hf_token_secret_arn:
        checks.append(check_hf_token(args.hf_token_secret_arn))

    # Print results
    print("\n" + "=" * 60)
    print("SageMaker Preflight Checks")
    print("=" * 60 + "\n")

    failed = []
    for check in checks:
        status = "✓" if check.ok else "✗"
        print(f"{status} {check.name:30s} {check.detail}")
        if not check.ok:
            failed.append(check)

    print("\n" + "=" * 60)
    if not failed:
        print("✓ All checks passed! Ready to launch SageMaker training.")
        print("=" * 60 + "\n")
        return 0

    fatal_failed = [c for c in failed if c.fatal]
    if fatal_failed:
        print("✗ Failed checks (blocking):")
        for check in fatal_failed:
            print(f"  - {check.name}: {check.detail}")
        print("=" * 60 + "\n")
        return 1

    print("⚠ Failed checks (non-blocking):")
    for check in failed:
        print(f"  - {check.name}: {check.detail}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
