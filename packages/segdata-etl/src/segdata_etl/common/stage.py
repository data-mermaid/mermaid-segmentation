"""Local staging of uniform-format dataset versions: where they live and what they are named.

``segdata_spec.stage`` writes a version to disk; ``segdata promote`` uploads the staged tree write-once.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa

from segdata_etl.common.copy_plan import StagedFile, write_copies, write_uploads
from segdata_spec import DATASET_BUCKET
from segdata_spec.git import GitState, probe_git, refuse_unpublishable
from segdata_spec.layout import cache_path, dataset_prefix, taxonomy_prefix
from segdata_spec.manifest import (
    ImageStore,
    LabelMapping,
    Manifest,
    ParentRef,
    Pin,
    SourceInfo,
    TaxonomyManifest,
)
from segdata_spec.stage import stage_dataset, stage_taxonomy

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("outputs") / "segdata-cache"
ETL_PACKAGE = "segdata-etl"

_SHA_LENGTH = 7


def version_taken(client: Any, bucket: str, prefix: str) -> bool:
    """Return whether any object exists under the version-folder ``prefix`` in ``bucket``."""
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(response.get("Contents"))


def next_free_version(
    stage_root: Path | None,
    bucket: str,
    prefix_for: Callable[[str], str],
    *,
    git_sha: str,
    today: date | None = None,
    client: Any | None = None,
) -> str:
    """Return ``YYYYMMDD-<sha7>``, with ``-2``, ``-3``, ... appended until the version is free.

    ``prefix_for(version)`` is the version's folder prefix, e.g. ``datasets/<ds>/<version>/``. A version is
    taken when its staged dir exists under ``stage_root`` (skipped when ``None``) or, given ``client``, when
    any object exists under its prefix in ``bucket``, which includes a half-promoted version. ``today``
    defaults to the current UTC date.
    """
    today = today or datetime.now(UTC).date()
    base = f"{today:%Y%m%d}-{git_sha[:_SHA_LENGTH]}"

    def taken(candidate: str) -> bool:
        prefix = prefix_for(candidate)
        if stage_root is not None and cache_path(stage_root, bucket, prefix).exists():
            return True
        return client is not None and version_taken(client, bucket, prefix)

    candidate, counter = base, 1
    while taken(candidate):
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def next_version(
    stage_root: Path,
    bucket: str,
    dataset: str,
    *,
    git_sha: str,
    today: date | None = None,
    client: Any | None = None,
) -> str:
    """Return the first free version of ``datasets/<dataset>/``; see ``next_free_version``."""
    return next_free_version(
        stage_root,
        bucket,
        partial(dataset_prefix, dataset),
        git_sha=git_sha,
        today=today,
        client=client,
    )


def find_repo_root(start: Path) -> Path:
    """Return the nearest directory at or above ``start`` holding both ``.git`` and ``uv.lock``.

    Raises:
        FileNotFoundError: No such directory exists.
    """
    for directory in (start, *start.parents):
        if (directory / ".git").exists() and (directory / "uv.lock").is_file():
            return directory
    raise FileNotFoundError(f"no git root holding uv.lock at or above {start}")


@dataclass(frozen=True)
class Publishable:
    """The repo root and git state a version is stamped with; the tree is clean and pushed."""

    root: Path
    git: GitState


def publishable_repo() -> Publishable:
    """Locate this repo, probe its git state, and refuse a dirty, unpushed or lock-less tree.

    Raises:
        FileNotFoundError: No repo root holding ``uv.lock`` is found.
        PublishRefused: The git tree is dirty or unpushed, or ``uv.lock`` is missing.
    """
    root = find_repo_root(Path(__file__).resolve().parent)
    git = probe_git(root)
    refuse_unpublishable(git, root)
    return Publishable(root=root, git=git)


@dataclass(frozen=True)
class StagedVersion:
    """A version staged on disk: its dataset (or taxonomy) name, version, manifest and manifest sha256."""

    dataset: str
    version: str
    manifest: Manifest | TaxonomyManifest
    sha256: str


def _report(staged: StagedVersion, detail: str) -> None:
    logger.info("staged %s %s%s", staged.dataset, staged.version, detail)
    print(f"version: {staged.version}\nmanifest_sha256: {staged.sha256}")


def stage_version(
    stage_dir: Path,
    *,
    repo: Publishable,
    client: Any,
    dataset: str,
    kind: Literal["points", "dense"],
    tables: Mapping[str, pa.Table],
    image_store: ImageStore,
    etl_command: str,
    source: SourceInfo,
    parents: Sequence[ParentRef] = (),
    extra_parent_dirs: Sequence[Path] = (),
    label_mapping: LabelMapping | None = None,
    dropped: Mapping[str, int] | None = None,
    taxonomy: Pin | None = None,
    copies: Sequence[Mapping[str, Any]] = (),
    uploads: Sequence[StagedFile] = (),
) -> StagedVersion:
    """Name, stage and report a dataset version; write its copy and upload plans when given.

    The version is the first free name against ``stage_dir`` and the bucket (``client``). Parents are read
    from ``stage_dir`` first, then ``extra_parent_dirs``. Prints ``version:`` and ``manifest_sha256:``.

    Raises:
        PublishRefused: The version's manifest is already staged.
        SchemaValidationError: A table fails its schema or the cross-table rules.
        ObjectConflict: A staged table or plan file exists with other bytes.
    """
    version = next_version(stage_dir, DATASET_BUCKET, dataset, git_sha=repo.git.sha, client=client)
    manifest, sha = stage_dataset(
        stage_dir,
        parent_dirs=(stage_dir, *extra_parent_dirs),
        bucket=DATASET_BUCKET,
        dataset=dataset,
        version=version,
        kind=kind,
        tables=tables,
        image_store=image_store,
        etl_package=ETL_PACKAGE,
        etl_command=etl_command,
        source=source,
        parents=parents,
        label_mapping=label_mapping,
        dropped=dropped,
        taxonomy=taxonomy,
        repo_dir=repo.root,
        git=repo.git,
    )
    if copies:
        write_copies(stage_dir, dataset_prefix(dataset, version), copies)
    if uploads:
        write_uploads(stage_dir, dataset, version, uploads)
    staged = StagedVersion(dataset=dataset, version=version, manifest=manifest, sha256=sha)
    _report(
        staged,
        f": {manifest.stats.n_images} images, {len(copies)} copies, {len(uploads)} uploads"
        f", dropped {dict(dropped or {})}",
    )
    return staged


def stage_taxonomy_version(
    stage_dir: Path,
    *,
    repo: Publishable,
    client: Any,
    name: str,
    tables: Mapping[str, pa.Table],
    etl_command: str,
    source: SourceInfo,
) -> StagedVersion:
    """Name, stage and report a taxonomy version; ``StagedVersion.dataset`` holds the taxonomy name.

    The version is the first free name against ``stage_dir`` and the bucket (``client``).

    Raises:
        PublishRefused: The version's manifest is already staged.
        SchemaValidationError: A table is missing, unknown or fails its schema.
        ObjectExists: A staged table exists with different bytes.
    """
    version = next_free_version(
        stage_dir,
        DATASET_BUCKET,
        partial(taxonomy_prefix, name),
        git_sha=repo.git.sha,
        client=client,
    )
    manifest, sha = stage_taxonomy(
        stage_dir,
        bucket=DATASET_BUCKET,
        name=name,
        version=version,
        tables=tables,
        etl_package=ETL_PACKAGE,
        etl_command=etl_command,
        source=source,
        repo_dir=repo.root,
        git=repo.git,
    )
    staged = StagedVersion(dataset=name, version=version, manifest=manifest, sha256=sha)
    _report(staged, "")
    return staged
