"""``segdata features attach``: stage existing feature vectors as a feature set of a published version.

``common.featurevector.gather_features`` adopts the files already in the family and verifies the
template sources for the rest; the set and its ``copies.jsonl`` are staged under ``--stage-dir``,
and ``segdata promote`` publishes them. Nothing here writes to S3.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from segdata_etl.common.cli_flags import pin_flags
from segdata_etl.common.featurevector import gather_features, read_extractor_spec, stage_features
from segdata_etl.common.stage import DEFAULT_CACHE_DIR, Publishable, publishable_repo
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.features import check_claimed_extractor
from segdata_spec.layout import MANIFEST_NAME, feature_set_prefix
from segdata_spec.manifest import FeatureSetPin, Pin
from segdata_spec.read import open_tables
from segdata_spec.s3 import exists
from segdata_spec.sync import sync_tables

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``features attach`` flags."""
    parser.add_argument("dataset", help="Published dataset name")
    pin_flags(parser, "", "the published dataset")
    parser.add_argument("--family", required=True, help="Feature family, a lowercase key segment")
    parser.add_argument(
        "--source-template",
        required=True,
        help="s3://bucket/key template; fields are image_id, key, group and the row's attrs keys",
    )
    parser.add_argument(
        "--extractor-spec",
        type=Path,
        required=True,
        help="JSON file: class_path, crop_size, feature_dim, weights_uri, weights_sha256",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Leave images whose source vector is absent out of the feature set",
    )
    parser.add_argument(
        "--stage-dir", type=Path, required=True, help="Local directory the feature set is staged in"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local cache for the synced version tables (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")


def attach(
    client: Any,
    pin: Pin,
    family: str,
    template: str,
    extractor: Mapping[str, Any],
    *,
    stage_dir: Path,
    cache_dir: Path,
    workers: int,
    allow_missing: bool,
    command: str,
    repo: Publishable,
) -> FeatureSetPin:
    """Stage the existing vectors of a published version as feature set ``family``; write nothing to S3.

    Returns:
        The pin of the staged set, which ``segdata promote`` publishes.

    Raises:
        PinMismatch: The published manifest differs from ``pin``.
        PublishRefused: The feature set is already published, or already staged.
        ExtractorMismatch: The family's stored extractor spec has another identity.
        ValueError: The version has no points, the image store or ``family`` is not a valid
            feature store, or the template does not resolve for a row that adopts no file;
            raised before any source call.
        FeatureVectorsMissing: Sources are absent and ``allow_missing`` is false.
        FeatureVectorsInvalid: A vector is unreadable, of another width than the extractor's
            ``feature_dim``, or lacks one of its image's points.
    """
    sync_tables(client, {pin.name: pin}, cache_dir, bucket=DATASET_BUCKET)
    if exists(
        client, DATASET_BUCKET, feature_set_prefix(pin.name, pin.version, family) + MANIFEST_NAME
    ):
        raise PublishRefused(
            f"feature set {family} of {pin.name} {pin.version} is already published in "
            f"{DATASET_BUCKET}"
        )
    tables = open_tables(cache_dir, DATASET_BUCKET, pin)
    if tables.points is None:
        raise ValueError(f"{pin.name} {pin.version} has no points.parquet to check coverage on")
    image_store = tables.manifest.image_store
    check_claimed_extractor(client, DATASET_BUCKET, image_store, family, extractor)
    gathered = gather_features(
        client,
        tables.images,
        tables.points,
        template,
        image_store=image_store,
        family=family,
        feature_dim=extractor["feature_dim"],
        workers=workers,
        allow_missing=allow_missing,
    )
    feature_pin = stage_features(
        stage_dir,
        gathered,
        dataset_dirs=(cache_dir,),
        dataset=pin,
        family=family,
        extractor=extractor,
        allow_missing=allow_missing,
        command=command,
        repo_dir=repo.root,
        git=repo.git,
    )
    logger.info(
        "staged %s of %s %s (set manifest sha256 %s): %d to copy, %d adopted, %d missing",
        family,
        pin.name,
        pin.version,
        feature_pin.manifest_sha256,
        len(gathered.copies),
        gathered.adopted,
        gathered.missing,
    )
    return feature_pin


def _command_line(args: argparse.Namespace) -> str:
    parts = [
        f"segdata features attach {args.dataset}",
        f"--version {args.version}",
        f"--manifest-sha256 {args.manifest_sha256}",
        f"--family {args.family}",
        f"--source-template {args.source_template}",
        f"--extractor-spec {args.extractor_spec}",
        f"--stage-dir {args.stage_dir}",
    ]
    if args.allow_missing:
        parts.append("--allow-missing")
    return " ".join(parts)


def run(args: argparse.Namespace, client: Any) -> int:
    """Run ``segdata features attach`` from parsed arguments.

    Raises:
        FileNotFoundError: No repo root holding ``uv.lock`` is found.
        PublishRefused: The git tree is dirty or unpushed, or has no ``uv.lock``; raised before
            any S3 call.
    """
    extractor = read_extractor_spec(args.extractor_spec)
    repo = publishable_repo()
    attach(
        client,
        Pin(name=args.dataset, version=args.version, manifest_sha256=args.manifest_sha256),
        args.family,
        args.source_template,
        extractor,
        stage_dir=args.stage_dir,
        cache_dir=args.cache_dir,
        workers=args.workers,
        allow_missing=args.allow_missing,
        command=_command_line(args),
        repo=repo,
    )
    return 0
