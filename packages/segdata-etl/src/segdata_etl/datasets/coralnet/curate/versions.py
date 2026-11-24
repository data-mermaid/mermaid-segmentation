"""``segdata curate coralnet``: publish a curated ``coralnet`` version built from ``coralnet-raw``.

``--label-map cleaned_up_rerun`` (v1) curates raw's points with the committed blacklist and
overrides and maps labels by the cleaned_up_rerun JSON. ``--label-map mermaid_api`` (v2) keeps the
tables of a v1 version and maps labels by the MERMAID LabelMapping API.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from segdata_etl.common.cli_flags import pin_command_parts, pin_flag_names, pin_values
from segdata_etl.common.publishing import checked_git_state, pick_version
from segdata_etl.common.store import open_store
from segdata_etl.common.version import utc_today
from segdata_etl.datasets.coralnet.curate.build import curate
from segdata_etl.label_maps.coralnet_api import (
    CoralNetLabelMappings,
    api_labels,
    fetch_coralnet_label_mappings,
    snapshot_json_text,
)
from segdata_etl.label_maps.coralnet_cleaned_up_rerun import cleaned_up_rerun_labels
from segdata_etl.splits.builder import _require_raw_parent
from segdata_etl.splits.inputs import sync_tables
from segdata_etl.taxonomy import find_repo_root
from segdata_spec.manifest import ImageStore, LabelMapping, Manifest, ParentRef, Pin, SourceInfo
from segdata_spec.publish import GitState, publish_dataset, table_bytes
from segdata_spec.read import DatasetVersion, open_taxonomy
from segdata_spec.schema import DATASET_TABLES, conform

logger = logging.getLogger(__name__)

DATASET = "coralnet"
RAW_DATASET = "coralnet-raw"
TAXONOMY = "mermaid-benthic"
CLEANED_UP_RERUN = "cleaned_up_rerun"
MERMAID_API = "mermaid_api"
LABEL_MAPS = (CLEANED_UP_RERUN, MERMAID_API)
CONFIG_DIR = Path("configs") / "coralnet"
REUSED_TABLES = ("images.parquet", "points.parquet", "stats/label_counts.parquet")
SNAPSHOT_NAME = "coralnet_label_mappings.json"
# Optional pin flags (``--<prefix>-version``, ``--<prefix>-manifest-sha256``) each label map takes.
_PINS_BY_LABEL_MAP = {CLEANED_UP_RERUN: ("taxonomy",), MERMAID_API: ("parent",)}
_OPTIONAL_PINS = ("taxonomy", "parent")


@dataclass(frozen=True)
class CuratedVersion:
    """Everything ``publish_dataset`` needs for one curated version besides the store and git."""

    tables: dict[str, pa.Table]
    parents: tuple[ParentRef, ...]
    label_mapping: LabelMapping
    dropped: dict[str, int]
    image_store: ImageStore
    source: SourceInfo


def pin_flag_errors(args: argparse.Namespace) -> list[str]:
    """Problems with the optional pin flags for ``args.label_map``; empty when they are right.

    Each label map needs both flags of its own pins and takes neither flag of the other's.
    """
    needed = _PINS_BY_LABEL_MAP[args.label_map]
    errors: list[str] = []
    for prefix in _OPTIONAL_PINS:
        values = pin_values(args, f"{prefix}-")
        flags = pin_flag_names(f"{prefix}-")
        if prefix in needed and None in values:
            errors.append(f"--label-map {args.label_map} needs {flags}")
        elif prefix not in needed and values != (None, None):
            errors.append(f"--label-map {args.label_map} takes neither of {flags}")
    return errors


def id_to_name(labels: pa.Table) -> dict[int, str | None]:
    """``label_id -> name`` of a ``labels.parquet``, with the empty-string name read as None."""
    ids = labels.column("label_id").to_pylist()
    names = labels.column("name").to_pylist()
    return {int(i): (n or None) for i, n in zip(ids, names, strict=True)}


def _source(raw: Manifest) -> SourceInfo:
    return SourceInfo(type="coralnet-scrape", uri=raw.source.uri, revision=raw.source.revision)


def build_v1(
    raw: DatasetVersion, raw_ref: ParentRef, benthic_attributes: pa.Table, config_dir: Path
) -> CuratedVersion:
    """Curate ``raw`` with the configs in ``config_dir`` and map its labels by the JSON.

    Raises:
        ValueError: ``raw`` has no points table, or a mapped target name matches more than one
            benthic attribute ignoring case.
    """
    curated = curate(
        raw,
        image_blacklist=config_dir / "image_blacklist.parquet",
        annotation_overrides=config_dir / "annotation_overrides.parquet",
    )
    labels, mapping = cleaned_up_rerun_labels(id_to_name(raw.labels), benthic_attributes)
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


def build_v2(
    v1: DatasetVersion,
    raw: DatasetVersion,
    v1_ref: ParentRef,
    raw_ref: ParentRef,
    mappings: CoralNetLabelMappings,
) -> CuratedVersion:
    """Keep ``v1``'s images, points and label counts, and map raw's labels by ``mappings``.

    ``dropped`` repeats v1's, since the rows are v1's.

    Raises:
        ValueError: ``v1`` has no points table.
    """
    if v1.points is None:
        raise ValueError(f"{DATASET} {v1.manifest.version} has no points table")
    labels, mapping = api_labels(id_to_name(raw.labels), mappings)
    return CuratedVersion(
        tables={
            "images.parquet": v1.images,
            "points.parquet": v1.points,
            "labels.parquet": labels,
            "stats/label_counts.parquet": v1.label_counts,
        },
        parents=(v1_ref, raw_ref),
        label_mapping=mapping,
        dropped=dict(v1.manifest.stats.dropped),
        image_store=raw.manifest.image_store,
        source=_source(raw.manifest),
    )


def check_reused_tables(v1: Manifest, tables: dict[str, pa.Table]) -> None:
    """Refuse tables of ``REUSED_TABLES`` whose published bytes would differ from ``v1``'s.

    Each table is conformed and serialised exactly as ``publish_dataset`` does.

    Raises:
        RuntimeError: A reused table differs in sha256 or size from ``v1``'s file entry.
    """
    changed = []
    for name in REUSED_TABLES:
        blob = table_bytes(conform(tables[name], DATASET_TABLES[name]))
        entry = v1.files[name]
        if (hashlib.sha256(blob).hexdigest(), len(blob)) != (entry.sha256, entry.bytes):
            changed.append(name)
    if changed:
        raise RuntimeError(f"{DATASET} would not reuse the bytes of {v1.version} for {changed}")


def _ref(pin: Pin) -> ParentRef:
    return ParentRef(dataset=pin.name, version=pin.version, manifest_sha256=pin.manifest_sha256)


def _pin(args: argparse.Namespace, prefix: str, name: str) -> Pin:
    version, manifest_sha256 = pin_values(args, f"{prefix}-")
    return Pin.model_validate(
        {"name": name, "version": version, "manifest_sha256": manifest_sha256}
    )


def _curate_v1(
    args: argparse.Namespace, client: Any, bucket: str, repo_root: Path
) -> CuratedVersion:
    raw_pin = _pin(args, "raw", RAW_DATASET)
    tax_pin = _pin(args, "taxonomy", TAXONOMY)
    synced = sync_tables(client, {RAW_DATASET: raw_pin}, args.cache_dir, bucket, tax_pin)
    tax = open_taxonomy(args.cache_dir, bucket, tax_pin)
    return build_v1(
        synced[RAW_DATASET], _ref(raw_pin), tax.benthic_attributes, repo_root / CONFIG_DIR
    )


def _curate_v2(
    args: argparse.Namespace, client: Any, bucket: str
) -> tuple[CuratedVersion, DatasetVersion, CoralNetLabelMappings]:
    raw_pin = _pin(args, "raw", RAW_DATASET)
    v1_pin = _pin(args, "parent", DATASET)
    synced = sync_tables(client, {DATASET: v1_pin, RAW_DATASET: raw_pin}, args.cache_dir, bucket)
    v1 = synced[DATASET]
    _require_raw_parent(v1.manifest, raw_pin)
    mappings = fetch_coralnet_label_mappings()
    built = build_v2(v1, synced[RAW_DATASET], _ref(v1_pin), _ref(raw_pin), mappings)
    return built, v1, mappings


def _write_snapshot(out_dir: Path, mappings: CoralNetLabelMappings) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SNAPSHOT_NAME
    path.write_text(snapshot_json_text(mappings))
    return path


def _command_line(args: argparse.Namespace) -> str:
    parts = [
        "segdata",
        "curate",
        DATASET,
        "--label-map",
        args.label_map,
        "--store",
        str(args.store),
    ]
    for prefix in ("raw", *_PINS_BY_LABEL_MAP[args.label_map]):
        parts += pin_command_parts(args, f"{prefix}-")
    return shlex.join(parts)


def _publish(
    args: argparse.Namespace,
    client: Any,
    bucket: str,
    built: CuratedVersion,
    version: str,
    repo_root: Path,
    git: GitState,
) -> tuple[Manifest, str]:
    return publish_dataset(
        client,
        bucket=bucket,
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
    """Build the version ``args.label_map`` names, publish it and print its version and sha256.

    The mermaid_api build first writes the fetched API snapshot to
    ``<args.out_dir>/<version>/coralnet_label_mappings.json``, so a published v2 always has one.

    Raises:
        ValueError: A mapped target name matches more than one benthic attribute ignoring case,
            or the raw pin is not a raw parent of the ``--parent-*`` version.
        RuntimeError: The v2 tables reused from v1 would not keep v1's bytes; nothing is published.
        OSError: The v2 snapshot cannot be written; nothing is published.
    """
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    git = checked_git_state(repo_root)
    client, bucket = open_store(args.store, source_client=source_client)
    if args.label_map == CLEANED_UP_RERUN:
        built = _curate_v1(args, client, bucket, repo_root)
        version = pick_version(client, bucket, DATASET, git_sha=git.sha, today=utc_today())
    else:
        built, v1, mappings = _curate_v2(args, client, bucket)
        check_reused_tables(v1.manifest, built.tables)
        version = pick_version(client, bucket, DATASET, git_sha=git.sha, today=utc_today())
        snapshot = _write_snapshot(args.out_dir / version, mappings)
        logger.info("wrote %s", snapshot)
    manifest, sha = _publish(args, client, bucket, built, version, repo_root, git)
    logger.info("published %s %s", DATASET, manifest.version)
    print(f"version: {manifest.version}\nmanifest_sha256: {sha}")
    return 0
