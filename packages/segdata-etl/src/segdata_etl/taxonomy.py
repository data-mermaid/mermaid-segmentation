"""``segdata taxonomy``: fetch the MERMAID benthic taxonomy and publish it as a version.

``--tables-from`` republishes the tables of a staged version, e.g. a local one to S3, with its
original source revision and the exact table bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from segdata_etl.common.store import open_store
from segdata_etl.common.version import list_versions, next_version, utc_today
from segdata_spec.manifest import SourceInfo, TaxonomyManifest
from segdata_spec.publish import probe_git, publish_taxonomy
from segdata_spec.schema import (
    BENTHIC_ATTRIBUTES_SCHEMA,
    GROWTH_FORMS_SCHEMA,
    TAXONOMY_TABLES,
    conform,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.datamermaid.org/v1"
NAME = "mermaid-benthic"
TIMEOUT = (10, 60)
_BA = "benthic_attributes.parquet"
_GF = "growth_forms.parquet"


def _get_json(session: requests.Session, url: str) -> Any:
    response = session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def _fetch_attributes(session: requests.Session, base_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    count: int | None = None
    url: str | None = f"{base_url}/benthicattributes/"
    while url:
        if url in seen:
            raise ValueError(f"benthic attributes pagination repeats {url}")
        seen.add(url)
        page = _get_json(session, url)
        if count is None:
            count = page["count"]
        rows += [
            {"id": r["id"], "name": r["name"], "parent_id": r["parent"]} for r in page["results"]
        ]
        if len(rows) > count:
            raise ValueError(f"benthic attributes: {len(rows)} rows fetched, API count is {count}")
        url = page.get("next")
    if not rows or len(rows) != count:
        raise ValueError(f"benthic attributes: {len(rows)} rows fetched, API count is {count}")
    return rows


def _fetch_growth_forms(session: requests.Session, base_url: str) -> list[dict[str, Any]]:
    for choice in _get_json(session, f"{base_url}/choices/"):
        if choice.get("name") == "growthforms":
            return [{"id": r["id"], "name": r["name"]} for r in choice["data"]]
    raise ValueError(f"{base_url}/choices/ has no 'growthforms' entry")


def _table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return conform(pa.Table.from_pylist(rows, schema=schema), schema).sort_by("id")


def fetch_taxonomy(
    session: requests.Session, base_url: str = DEFAULT_BASE_URL
) -> dict[str, pa.Table]:
    """Fetch the benthic attributes and growth forms, sorted by id.

    Raises:
        requests.HTTPError: A response is not 2xx.
        ValueError: ``/choices/`` has no ``growthforms`` entry, the attribute pages repeat a URL,
            disagree with the API count or hold no rows, a table has duplicate ids, or an
            attribute's parent is not an attribute.
        KeyError: An attribute row lacks ``parent``.
    """
    attributes = _table(_fetch_attributes(session, base_url), BENTHIC_ATTRIBUTES_SCHEMA)
    growth_forms = _table(_fetch_growth_forms(session, base_url), GROWTH_FORMS_SCHEMA)
    _check_tables(attributes, growth_forms)
    return {_BA: attributes, _GF: growth_forms}


def _check_tables(attributes: pa.Table, growth_forms: pa.Table) -> None:
    for label, table in (("benthic attributes", attributes), ("growth forms", growth_forms)):
        ids = table["id"].to_pylist()
        if len(set(ids)) != len(ids):
            raise ValueError(f"{label} hold duplicate ids")
    known = set(attributes["id"].to_pylist())
    unknown = {p for p in attributes["parent_id"].to_pylist() if p is not None} - known
    if unknown:
        raise ValueError(f"benthic attributes name unknown parents: {sorted(unknown)}")


def find_repo_root(start: Path) -> Path:
    """Return the nearest directory at or above ``start`` holding both ``.git`` and ``uv.lock``.

    Raises:
        FileNotFoundError: No such directory exists.
    """
    for directory in (start, *start.parents):
        if (directory / ".git").exists() and (directory / "uv.lock").is_file():
            return directory
    raise FileNotFoundError(f"no git root holding uv.lock at or above {start}")


def _command_line(args: argparse.Namespace) -> str:
    parts = ["segdata", "taxonomy", "--store", str(args.store)]
    if args.tables_from is not None:
        parts += ["--tables-from", str(args.tables_from)]
    if args.base_url != DEFAULT_BASE_URL:
        parts += ["--base-url", args.base_url]
    return shlex.join(parts)


def _read_staged(directory: Path) -> tuple[dict[str, pa.Table], SourceInfo]:
    manifest = TaxonomyManifest.model_validate_json((directory / "manifest.json").read_bytes())
    if manifest.name != NAME:
        raise ValueError(f"{directory} stages {manifest.name!r}, not {NAME!r}")
    tables: dict[str, pa.Table] = {}
    for name in TAXONOMY_TABLES:
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest.files[name].sha256:
            raise ValueError(f"{directory / name} does not match the sha256 in its manifest")
        tables[name] = pq.read_table(io.BytesIO(data))
    return tables, manifest.source


def _fetch_source(
    session: requests.Session, base_url: str
) -> tuple[dict[str, pa.Table], SourceInfo]:
    fetched_at = datetime.now(UTC).isoformat()
    tables = fetch_taxonomy(session, base_url)
    return tables, SourceInfo(type="mermaid-api", uri=base_url, revision=fetched_at)


def run_taxonomy(
    args: argparse.Namespace, *, source_client: Any, session: requests.Session | None = None
) -> int:
    """Publish a new ``mermaid-benthic`` version and print its version and manifest sha256."""
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    probe = probe_git(repo_root)
    if args.tables_from is not None:
        tables, source = _read_staged(Path(args.tables_from))
    else:
        tables, source = _fetch_source(session or requests.Session(), args.base_url)
    client, bucket = open_store(args.store, source_client=source_client)
    version = next_version(
        list_versions(client, bucket, f"taxonomy/{NAME}/"),
        git_sha=probe.sha,
        today=utc_today(),
    )
    manifest, sha = publish_taxonomy(
        client,
        bucket=bucket,
        name=NAME,
        version=version,
        tables=tables,
        etl_package="segdata-etl",
        etl_command=_command_line(args),
        source=source,
        repo_dir=repo_root,
        git=probe,
    )
    logger.info("published taxonomy %s", manifest.version)
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--tables-from`` and ``--base-url``; ``--store`` comes from ``add_store_argument``."""
    parser.add_argument(
        "--tables-from",
        type=Path,
        default=None,
        help="Publish the tables of this staged taxonomy version folder; makes no HTTP call",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=argparse.SUPPRESS)
