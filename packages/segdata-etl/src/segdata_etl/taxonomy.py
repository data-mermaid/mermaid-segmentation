"""``segdata taxonomy``: stage the MERMAID benthic taxonomy locally, then publish the staged tables.

``--stage-dir`` fetches the API and stages a version on disk. ``--tables-from <staged folder> --publish``
re-checks those tables and publishes them to S3 as a new version, with the staged source revision and the
exact table bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import shlex
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from segdata_etl.common.stage import LocalDirS3, next_free_version, refuse_unpublishable
from segdata_spec import DATASET_BUCKET
from segdata_spec._s3 import taxonomy_prefix
from segdata_spec.manifest import SourceInfo, TaxonomyManifest
from segdata_spec.publish import GitState, probe_git, publish_taxonomy
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
    parts = ["segdata", "taxonomy"]
    if args.tables_from is not None:
        parts += ["--tables-from", str(args.tables_from), "--publish"]
    else:
        parts += ["--stage-dir", str(args.stage_dir)]
        if args.base_url != DEFAULT_BASE_URL:
            parts += ["--base-url", args.base_url]
    return shlex.join(parts)


def _check_mode(args: argparse.Namespace) -> None:
    """Refuse flag combinations other than stage (``--stage-dir``) or publish (``--tables-from --publish``)."""
    if args.tables_from is not None:
        if not args.publish:
            raise ValueError("--tables-from only publishes a staged version; add --publish")
        if args.stage_dir is not None:
            raise ValueError("--stage-dir stages a fetch; it cannot be combined with --tables-from")
    elif args.publish:
        raise ValueError("--publish needs --tables-from <staged folder>; stage first, then publish")
    elif args.stage_dir is None:
        raise ValueError(
            "give --stage-dir to stage, or --tables-from <folder> --publish to publish"
        )


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
    _check_tables(tables[_BA], tables[_GF])
    return tables, manifest.source


def _fetch_source(
    session: requests.Session, base_url: str
) -> tuple[dict[str, pa.Table], SourceInfo]:
    fetched_at = datetime.now(UTC).isoformat()
    tables = fetch_taxonomy(session, base_url)
    return tables, SourceInfo(type="mermaid-api", uri=base_url, revision=fetched_at)


def _write_version(
    args: argparse.Namespace,
    *,
    client: Any,
    version: str,
    tables: dict[str, pa.Table],
    source: SourceInfo,
    repo_root: Path,
    git: GitState,
) -> int:
    manifest, sha = publish_taxonomy(
        client,
        bucket=DATASET_BUCKET,
        name=NAME,
        version=version,
        tables=tables,
        etl_package="segdata-etl",
        etl_command=_command_line(args),
        source=source,
        repo_dir=repo_root,
        git=git,
    )
    logger.info("wrote taxonomy %s", manifest.version)
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0


def run_taxonomy(
    args: argparse.Namespace, *, source_client: Any, session: requests.Session | None = None
) -> int:
    """Stage a fetched taxonomy under ``--stage-dir``, or publish a staged one with ``--tables-from``.

    Prints the version and manifest sha256. A staged version is named against the staged folders and the
    versions in S3; a published one is named against S3 only.

    Raises:
        ValueError: The flags are not a stage or a publish, or a staged table fails its check.
        PublishRefused: The git tree is dirty or unpushed.
    """
    _check_mode(args)
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    git = probe_git(repo_root)
    refuse_unpublishable(git)
    staging = args.tables_from is None
    if staging:
        tables, source = _fetch_source(session or requests.Session(), args.base_url)
    else:
        tables, source = _read_staged(Path(args.tables_from))
    version = next_free_version(
        args.stage_dir if staging else None,
        DATASET_BUCKET,
        partial(taxonomy_prefix, NAME),
        git_sha=git.sha,
        client=source_client,
    )
    client = LocalDirS3(args.stage_dir) if staging else source_client
    return _write_version(
        args,
        client=client,
        version=version,
        tables=tables,
        source=source,
        repo_root=repo_root,
        git=git,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--stage-dir``, ``--tables-from``, ``--publish`` and ``--base-url``."""
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Fetch the taxonomy and stage a version in this local directory",
    )
    parser.add_argument(
        "--tables-from",
        type=Path,
        default=None,
        help="Staged taxonomy version folder to publish to S3; requires --publish; makes no HTTP call",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the --tables-from version to S3 after re-checking its tables",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=argparse.SUPPRESS)
