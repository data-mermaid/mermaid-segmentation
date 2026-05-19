"""Low-level I/O helpers for the CoralNet ETL.

Centralises boto3 client construction (with adaptive retries), S3 CSV reads via either boto3 or
ibis-on-duckdb-httpfs, deterministic parquet writes, and version-tag generation. Keeping these here
makes the higher-level audit / annotations / images modules pure data-flow code.
"""

from __future__ import annotations

import io as _io
import logging
import subprocess
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import ENV_VERSION_OVERRIDE
from .schemas import validate

logger = logging.getLogger(__name__)

_BOTO_CONFIG = Config(retries={"max_attempts": 10, "mode": "adaptive"})

# Type alias for the CSV reader used by audit/annotations stages.
# Returns None on 404, raises on other errors so the caller can mark them.
CsvReader = Callable[[str, str], "pd.DataFrame | None"]


def make_s3_client() -> BaseClient:
    """Build a boto3 S3 client with adaptive retries for long-running ETL jobs."""
    return boto3.client("s3", config=_BOTO_CONFIG)


def make_ibis_connection():
    """Open a fresh DuckDB-backed ibis connection with S3 httpfs configured.

    DuckDB connections are not thread-safe — call this once per worker thread. Uses the AWS
    credential chain so SSO / IAM-role / env-var creds are all discovered automatically.
    """
    import ibis

    con = ibis.duckdb.connect()
    con.raw_sql("INSTALL httpfs; LOAD httpfs;")
    con.raw_sql("CREATE OR REPLACE SECRET s3 (TYPE S3, PROVIDER CREDENTIAL_CHAIN)")
    return con


def refresh_ibis_s3_credentials(con) -> None:
    """Re-run the DuckDB ``CREATE OR REPLACE SECRET`` to pick up rotated creds.

    SageMaker IAM creds expire every ~60 min. DuckDB httpfs does not auto-refresh them, so the audit
    / annotations loops call this periodically.
    """
    con.raw_sql("CREATE OR REPLACE SECRET s3 (TYPE S3, PROVIDER CREDENTIAL_CHAIN)")


def read_csv_via_ibis(con, bucket: str, key: str) -> pd.DataFrame | None:
    """Read a CSV from S3 through ibis + DuckDB httpfs into a pandas DataFrame.

    Uses ``null_padding=True`` so ragged CoralNet CSVs (which sometimes carry fewer columns on the
    last row than the header declares) parse cleanly instead of raising. Returns ``None`` if DuckDB
    reports the file as missing, matching :func:`read_csv_s3`'s 404 contract.
    """
    uri = f"s3://{bucket}/{key}"
    try:
        table = con.read_csv(uri, null_padding=True)
        return table.execute()
    except Exception as e:  # noqa: BLE001 - DuckDB surfaces many exception types
        msg = str(e)
        if any(needle in msg for needle in ("HTTP 404", "Not Found", "NoSuchKey")):
            return None
        raise


def read_csv_s3(
    client: BaseClient, bucket: str, key: str, **read_csv_kwargs
) -> pd.DataFrame | None:
    """Read a CSV from S3 into a DataFrame via boto3 + pandas, or return None on 404.

    Used by tests (via :func:`make_csv_reader_s3`) and as a fallback when no ibis connection is
    available. Other ClientErrors propagate so the caller can mark them as ``s3_error`` in the audit
    row.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise
    body = response["Body"].read()
    return pd.read_csv(_io.BytesIO(body), **read_csv_kwargs)


def make_csv_reader_ibis(con) -> CsvReader:
    """Build a :data:`CsvReader` closure over an ibis connection."""

    def _reader(bucket: str, key: str) -> pd.DataFrame | None:
        return read_csv_via_ibis(con, bucket, key)

    return _reader


def make_csv_reader_s3(client: BaseClient, **read_csv_kwargs) -> CsvReader:
    """Build a :data:`CsvReader` closure over a boto3 client.

    Convenient for tests against ``FakeS3`` and for code paths that already have a live boto3 client
    and don't want to spin up DuckDB.
    """

    def _reader(bucket: str, key: str) -> pd.DataFrame | None:
        return read_csv_s3(client, bucket, key, **read_csv_kwargs)

    return _reader


def head_object_exists(client: BaseClient, bucket: str, key: str) -> bool:
    """True if the S3 object exists, False if it returns 404.

    Re-raises on other errors.
    """
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def write_parquet_deterministic(
    df: pd.DataFrame,
    output_path: str | Path,
    schema: pa.Schema,
    primary_key: tuple[str, ...],
) -> None:
    """Write ``df`` to a parquet file with deterministic, content-addressable bytes.

    Sorts rows by ``primary_key``, casts every column to the declared pyarrow type, and writes with
    stable parquet settings (zstd-3, no statistics, no ``created_by`` metadata).
    """
    validate(df, schema)
    df = df.copy()
    df = df.sort_values(list(primary_key), kind="mergesort").reset_index(drop=True)
    df = df[[field.name for field in schema]]
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    table = table.replace_schema_metadata({})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        output_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=False,
        data_page_size=1 << 20,
    )


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    bucket, _, key = s3_uri[len("s3://") :].partition("/")
    return bucket, key


def upload_to_s3(client: BaseClient, local_path: str | Path, s3_uri: str) -> None:
    """Upload a local file to an ``s3://bucket/key`` URI."""
    bucket, key = parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 URI missing key: {s3_uri}")
    client.upload_file(str(local_path), bucket, key)


def refresh_thread_ibis_credentials_if_present(thread_local: threading.local) -> None:
    """Re-run DuckDB ``CREATE OR REPLACE SECRET`` on a thread-local ibis connection.

    No-ops if the thread hasn't initialised one yet. Used by the audit and annotations loops to
    handle SageMaker IAM rotation mid-run without each module reinventing the same try/except.
    """
    con = getattr(thread_local, "ibis_con", None)
    if con is None:
        return
    try:
        refresh_ibis_s3_credentials(con)
    except Exception as e:  # noqa: BLE001 - credential refresh must never crash the loop
        logger.warning("Ibis credential refresh failed: %s", e)


def _git_short_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip()
    return sha or None


def compute_version_tag(override: str | None = None) -> str:
    """Return a version tag of the form ``YYYYMMDD_<sha>`` or ``YYYYMMDD_nogit``.

    Resolution order:
        1. Explicit ``override`` argument.
        2. ``MERMAID_CORALNET_VERSION_OVERRIDE`` environment variable.
        3. Computed from current UTC date + ``git rev-parse --short HEAD``.
        4. ``YYYYMMDD_nogit`` if git is unavailable or HEAD is detached without history.
    """
    import os

    if override:
        return override
    env_override = os.getenv(ENV_VERSION_OVERRIDE)
    if env_override:
        return env_override
    today = datetime.now(UTC).strftime("%Y%m%d")
    sha = _git_short_sha()
    return f"{today}_{sha}" if sha else f"{today}_nogit"
