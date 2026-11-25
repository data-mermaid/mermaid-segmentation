"""``segdata curate coralnet``: stage a curated ``coralnet`` version built from ``coralnet-raw``.

Curates raw's points with the committed blacklist and overrides and maps labels by the MERMAID
LabelMapping API, checked against the pinned taxonomy. Stages tables only, on raw's image store,
for ``segdata promote`` to publish. The published ``coralnet`` v1 was built with the frozen
cleaned_up_rerun mapping at git e6abfad.
"""

from __future__ import annotations

import argparse
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
from botocore.exceptions import ClientError

from segdata_etl.common.cli_flags import pin_command_parts, pin_values
from segdata_etl.common.stage import LocalDirS3, next_version, refuse_unpublishable
from segdata_etl.datasets.coralnet.curate.build import curate
from segdata_etl.label_maps.coralnet_api import (
    CoralNetLabelMappings,
    api_labels,
    fetch_coralnet_label_mappings,
)
from segdata_etl.splits.inputs import sync_tables
from segdata_etl.taxonomy import find_repo_root, require_published_taxonomy
from segdata_spec import DATASET_BUCKET
from segdata_spec._s3 import is_not_found
from segdata_spec.manifest import ImageStore, LabelMapping, Manifest, ParentRef, Pin, SourceInfo
from segdata_spec.publish import GitState, probe_git, publish_dataset
from segdata_spec.read import DatasetVersion, open_taxonomy

logger = logging.getLogger(__name__)

DATASET = "coralnet"
RAW_DATASET = "coralnet-raw"
TAXONOMY = "mermaid-benthic"
CONFIG_DIR = Path("configs") / "coralnet"


@dataclass(frozen=True)
class CuratedVersion:
    """Everything ``publish_dataset`` needs for one curated version besides the store and git."""

    tables: dict[str, pa.Table]
    parents: tuple[ParentRef, ...]
    label_mapping: LabelMapping
    dropped: dict[str, int]
    image_store: ImageStore
    source: SourceInfo


def id_to_name(labels: pa.Table) -> dict[int, str | None]:
    """``label_id -> name`` of a ``labels.parquet``, with the empty-string name read as None."""
    ids = labels.column("label_id").to_pylist()
    names = labels.column("name").to_pylist()
    return {int(i): (n or None) for i, n in zip(ids, names, strict=True)}


def _source(raw: Manifest) -> SourceInfo:
    return SourceInfo(type="coralnet-scrape", uri=raw.source.uri, revision=raw.source.revision)


def build_curated(
    raw: DatasetVersion,
    raw_ref: ParentRef,
    mappings: CoralNetLabelMappings,
    benthic_attributes: pa.Table,
    growth_forms: pa.Table,
    config_dir: Path,
) -> CuratedVersion:
    """Curate ``raw`` with the configs in ``config_dir`` and map its labels by the API rows.

    Raises:
        ValueError: ``raw`` has no points table, a mapped record has no
            ``benthic_attribute_id``, or a mapped ``benthic_attribute_id`` or ``growth_form_id`` is
            absent from ``benthic_attributes`` or ``growth_forms``.
    """
    curated = curate(
        raw,
        image_blacklist=config_dir / "image_blacklist.parquet",
        annotation_overrides=config_dir / "annotation_overrides.parquet",
    )
    labels, mapping = api_labels(
        id_to_name(raw.labels),
        mappings,
        set(benthic_attributes.column("id").to_pylist()),
        set(growth_forms.column("id").to_pylist()),
    )
    return CuratedVersion(
        tables={
            "images.parquet": curated.images,
            "points.parquet": curated.points,
            "labels.parquet": labels,
            "stats/label_counts.parquet": curated.label_counts,
        },
        parents=(raw_ref,),
        label_mapping=mapping,
        dropped=curated.dropped,
        image_store=raw.manifest.image_store,
        source=_source(raw.manifest),
    )


def _ref(pin: Pin) -> ParentRef:
    return ParentRef(dataset=pin.name, version=pin.version, manifest_sha256=pin.manifest_sha256)


def _pin(args: argparse.Namespace, prefix: str, name: str) -> Pin:
    version, manifest_sha256 = pin_values(args, f"{prefix}-")
    return Pin.model_validate(
        {"name": name, "version": version, "manifest_sha256": manifest_sha256}
    )


def _curate(args: argparse.Namespace, client: Any, bucket: str, repo_root: Path) -> CuratedVersion:
    raw_pin = _pin(args, "raw", RAW_DATASET)
    tax_pin = _pin(args, "taxonomy", TAXONOMY)
    require_published_taxonomy(client, bucket, tax_pin)
    synced = sync_tables(client, {RAW_DATASET: raw_pin}, args.cache_dir, bucket, tax_pin)
    tax = open_taxonomy(args.cache_dir, bucket, tax_pin)
    mappings = fetch_coralnet_label_mappings()
    return build_curated(
        synced[RAW_DATASET],
        _ref(raw_pin),
        mappings,
        tax.benthic_attributes,
        tax.growth_forms,
        repo_root / CONFIG_DIR,
    )


class _StageOverCache:
    """Writes to the stage; a read the stage cannot serve falls back to the cache.

    ``publish_dataset`` reads its first parent through its client, and a parent synced from S3
    sits in the cache; it verifies that parent's manifest and tables against the pin.
    """

    def __init__(self, stage_dir: Path, cache_dir: Path):
        """Write under ``stage_dir``; read from ``stage_dir``, then ``cache_dir``."""
        self._stage = LocalDirS3(stage_dir)
        self._cache = LocalDirS3(cache_dir)

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        """Write the object to the stage."""
        return self._stage.put_object(**kwargs)

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        """Head the object in the stage."""
        return self._stage.head_object(**kwargs)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        """Return the object from the stage, or from the cache when the stage lacks it."""
        try:
            return self._stage.get_object(**kwargs)
        except ClientError as exc:
            if not is_not_found(exc):
                raise
        return self._cache.get_object(**kwargs)


def _command_line(args: argparse.Namespace) -> str:
    parts = [
        "segdata",
        "curate",
        DATASET,
        "--stage-dir",
        str(args.stage_dir),
    ]
    for prefix in ("raw", "taxonomy"):
        parts += pin_command_parts(args, f"{prefix}-")
    return shlex.join(parts)


def _stage(
    args: argparse.Namespace,
    built: CuratedVersion,
    version: str,
    repo_root: Path,
    git: GitState,
) -> tuple[Manifest, str]:
    return publish_dataset(
        _StageOverCache(args.stage_dir, args.cache_dir),
        bucket=DATASET_BUCKET,
        dataset=DATASET,
        version=version,
        kind="points",
        tables=built.tables,
        image_store=built.image_store,
        etl_package="segdata-etl",
        etl_command=_command_line(args),
        source=built.source,
        parents=built.parents,
        label_mapping=built.label_mapping,
        dropped=built.dropped,
        repo_dir=repo_root,
        git=git,
    )


def run(args: argparse.Namespace, *, source_client: Any) -> int:
    """Stage the curated version under ``args.stage_dir``; print its version and sha256.

    Raw and the taxonomy are synced from the dataset bucket into ``args.cache_dir``; the sync
    reads the cache first, so ``--cache-dir`` set to the stage dir reads a raw staged there. The
    taxonomy pin must still be published in the dataset bucket, even when the cache holds it. The
    version reuses raw's image store and stages no copy or upload plan. Labels map by the live
    MERMAID LabelMapping API, whose digest and fetch time the manifest records. The published v1
    was built with the frozen cleaned_up_rerun mapping at git e6abfad.

    Raises:
        PublishRefused: The git tree is dirty or unpushed, or the taxonomy pin is not published
            in the bucket; nothing is staged.
        ValueError: A mapped ``benthic_attribute_id`` is absent from the pinned taxonomy;
            nothing is staged.
        requests.RequestException: The LabelMapping API request fails after its retries; the
            error propagates and nothing is staged.
    """
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    git = probe_git(repo_root)
    refuse_unpublishable(git)
    built = _curate(args, source_client, DATASET_BUCKET, repo_root)
    version = next_version(
        args.stage_dir, DATASET_BUCKET, DATASET, git_sha=git.sha, client=source_client
    )
    manifest, sha = _stage(args, built, version, repo_root, git)
    logger.info("staged %s %s under %s", DATASET, manifest.version, args.stage_dir)
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0
