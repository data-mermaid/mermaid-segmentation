"""``segdata taxonomy``: fetch the MERMAID benthic taxonomy and stage a version on disk.

The version is named against S3, so the staged name is the published name; ``segdata promote --taxonomy``
uploads the staged tables write-once.
"""

from __future__ import annotations

import argparse
import logging
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import requests

from segdata_etl.common.stage import publishable_repo, stage_taxonomy_version
from segdata_spec.manifest import SourceInfo
from segdata_spec.schema import (
    BENTHIC_ATTRIBUTES_SCHEMA,
    GROWTH_FORMS_SCHEMA,
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
            if not choice["data"]:
                raise ValueError(f"{base_url}/choices/ 'growthforms' entry is empty")
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
        ValueError: ``/choices/`` has no ``growthforms`` entry or it is empty, the attribute pages repeat a URL,
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


def _command_line(args: argparse.Namespace) -> str:
    parts = ["segdata", "taxonomy", "--stage-dir", str(args.stage_dir)]
    if args.base_url != DEFAULT_BASE_URL:
        parts += ["--base-url", args.base_url]
    return shlex.join(parts)


def _fetch_source(
    session: requests.Session, base_url: str
) -> tuple[dict[str, pa.Table], SourceInfo]:
    fetched_at = datetime.now(UTC).isoformat()
    tables = fetch_taxonomy(session, base_url)
    return tables, SourceInfo(type="mermaid-api", uri=base_url, revision=fetched_at)


def run_taxonomy(
    args: argparse.Namespace, *, source_client: Any, session: requests.Session | None = None
) -> int:
    """Fetch the taxonomy and stage it under ``--stage-dir``; print the version and manifest sha256.

    The version is the first free name against the staged folders and the versions in S3
    (``source_client``), so the staged name is the name ``segdata promote --taxonomy`` publishes.

    Raises:
        PublishRefused: The git tree is dirty or unpushed.
        ValueError: The API data fails its checks.
    """
    repo = publishable_repo()
    tables, source = _fetch_source(session or requests.Session(), args.base_url)
    stage_taxonomy_version(
        args.stage_dir,
        repo=repo,
        client=source_client,
        name=NAME,
        tables=tables,
        etl_command=_command_line(args),
        source=source,
    )
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--stage-dir`` and ``--base-url``."""
    parser.add_argument(
        "--stage-dir",
        type=Path,
        required=True,
        help="Fetch the taxonomy and stage a version in this local directory",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=argparse.SUPPRESS)
