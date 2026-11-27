"""Low-level I/O helpers for the CoralNet dataset build.

Centralises boto3 client construction (with adaptive retries), S3 CSV reads via either boto3 or
ibis-on-duckdb-httpfs, deterministic parquet writes, uploads, and version-tag generation, so the
annotations / images / resize / manifest modules stay pure data-flow code.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import BaseClient

from coralnet_scraper.s3_io import read_csv_s3
from segdata_spec.s3 import exists

# Returns None when the object is missing, and raises on other errors so the caller can record them.
CsvReader = Callable[[str, str], "pd.DataFrame | None"]


# Returns whether the object exists, and raises on errors other than a missing key.
MarkerCheck = Callable[[str, str], bool]


def make_ibis_connection() -> Any:
    """Open a fresh DuckDB-backed ibis connection with S3 httpfs configured.

    DuckDB connections are not thread-safe, so call this once per worker thread. The AWS credential
    chain discovers SSO, IAM-role and environment credentials.
    """
    import ibis

    con = ibis.duckdb.connect()
    con.raw_sql("INSTALL httpfs; LOAD httpfs;")
    con.raw_sql("CREATE OR REPLACE SECRET s3 (TYPE S3, PROVIDER CREDENTIAL_CHAIN)")
    return con


def read_csv_via_ibis(con: Any, client: Any, bucket: str, key: str) -> pd.DataFrame | None:
    """Read a CSV from S3 through ibis + DuckDB httpfs into a pandas DataFrame.

    CoralNet CSVs are messy, so several DuckDB defaults are overridden:
    - ``null_padding=True``: ragged rows (fewer columns than header) parse cleanly.
    - ``all_varchar=True``: prevents type-inference failures on columns with mixed types
      (e.g. "YS-D2" in a column DuckDB would otherwise infer as BIGINT).
    - ``quote='"'``: forces double-quote recognition so commas inside quoted free-text fields
      (e.g. ``"Sheltered fringing reef, steep"``) don't split into extra columns.

    Downstream stages select and cast only the columns they need.

    Args:
        con: An ibis DuckDB connection from :func:`make_ibis_connection`.
        client: A boto3 S3 client; a HEAD on the key decides whether the object is missing.
        bucket: S3 bucket name.
        key: Object key of the CSV.

    Returns:
        The CSV as a DataFrame of strings, or ``None`` if the object is missing.

    Raises:
        botocore.exceptions.ClientError: The HEAD failed for a reason other than a missing key.
    """
    if not exists(client, bucket, key):
        return None
    table = con.read_csv(f"s3://{bucket}/{key}", null_padding=True, all_varchar=True, quote='"')
    return table.execute()


def make_csv_reader_ibis(con: Any, client: Any) -> CsvReader:
    """Build a :data:`CsvReader` closure over an ibis connection and the client that checks keys."""

    def _reader(bucket: str, key: str) -> pd.DataFrame | None:
        return read_csv_via_ibis(con, client, bucket, key)

    return _reader


def make_csv_reader_s3(client: Any, **read_csv_kwargs: Any) -> CsvReader:
    """Build a :data:`CsvReader` closure over a boto3 client.

    Lets tests read from an in-memory fake S3 client without spinning up DuckDB.
    """

    def _reader(bucket: str, key: str) -> pd.DataFrame | None:
        return read_csv_s3(client, bucket, key, **read_csv_kwargs)

    return _reader


def make_marker_check(client: Any) -> MarkerCheck:
    """Build a :data:`MarkerCheck` that probes object existence with ``head_object``."""

    def _check(bucket: str, key: str) -> bool:
        return exists(client, bucket, key)

    return _check


def write_parquet_deterministic(
    df: pd.DataFrame,
    output_path: str | Path,
    schema: pa.Schema,
    primary_key: tuple[str, ...],
) -> None:
    """Write ``df`` to a parquet file with deterministic, content-addressable bytes.

    Sorts rows by ``primary_key``, casts every column to the declared pyarrow type, and writes with
    stable parquet settings (zstd-3, no statistics, no ``created_by`` metadata).

    Args:
        df: Frame holding at least the columns of ``schema``.
        output_path: Destination file; parent directories are created.
        schema: Declared output schema.
        primary_key: Columns the rows are sorted by.
    """
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
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Raises:
        ValueError: If ``s3_uri`` does not start with ``s3://``.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    bucket, _, key = s3_uri[len("s3://") :].partition("/")
    return bucket, key


def upload_to_s3(client: BaseClient, local_path: str | Path, s3_uri: str) -> None:
    """Upload a local file to an ``s3://bucket/key`` URI.

    Raises:
        ValueError: If ``s3_uri`` is not an ``s3://`` URI or has no key.
    """
    bucket, key = parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 URI missing key: {s3_uri}")
    client.upload_file(str(local_path), bucket, key)


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

    Args:
        override: Returned unchanged when given.

    Returns:
        ``override`` when set; otherwise the current UTC date plus ``git rev-parse --short HEAD``,
        or ``YYYYMMDD_nogit`` when git is unavailable.
    """
    if override:
        return override
    today = datetime.now(UTC).strftime("%Y%m%d")
    sha = _git_short_sha()
    return f"{today}_{sha}" if sha else f"{today}_nogit"
