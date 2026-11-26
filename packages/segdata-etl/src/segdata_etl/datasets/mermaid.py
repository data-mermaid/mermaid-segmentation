"""``segdata build mermaid``: stage a uniform-format points dataset from MERMAID's confirmed annotations.

The build reads the nightly annotations parquet, the source image headers and each image's pyspacer
feature vector, never writes to S3, and leaves a version under ``--stage-dir`` plus ``copies.jsonl``,
the plan a later promote step runs to copy the images into the dataset bucket. Beside it,
``features.jsonl`` plans the copy of the feature vectors, and ``features/<family>/`` in the version
folder holds the staged feature index and extractor spec that promote publishes as a feature set.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_etl.common.copy_plan import (
    FEATURE_STAGE_NAME,
    FEATURES_NAME,
    ImageOutcome,
    ObjectFacts,
    copy_plan,
    stage_bytes,
    write_copies,
)
from segdata_etl.common.copy_plan import probe_image as probe_source_image
from segdata_etl.common.featurevector import (
    FeatureVectorsMissing,
    Point,
    VectorFacts,
    extractor_spec_bytes,
    points_by_image,
    read_extractor_spec,
    read_vector,
    refusal,
    refuse_invalid,
)
from segdata_etl.common.s3_objects import EXTENSIONS
from segdata_etl.common.stage import next_version
from segdata_spec import DATASET_BUCKET
from segdata_spec.features import feature_file_key, feature_store_name
from segdata_spec.git import probe_git, refuse_unpublishable
from segdata_spec.layout import (
    EXTRACTOR_SPEC_NAME,
    FEATURE_INDEX_NAME,
    cache_path,
    dataset_prefix,
    feature_store_prefix,
)
from segdata_spec.manifest import ImageStore, LabelMapping, ParentRef, SourceInfo
from segdata_spec.s3 import is_not_found, is_precondition_failed, norm_etag
from segdata_spec.schema import (
    FEATURE_INDEX_SCHEMA,
    IMAGES_SCHEMA,
    LABEL_COUNTS_SCHEMA,
    LABELS_SCHEMA,
    POINTS_SCHEMA,
)
from segdata_spec.stage import stage_dataset, table_bytes

logger = logging.getLogger(__name__)

DATASET = "mermaid"
SOURCE_BUCKET = "coral-reef-training"
SOURCE_IMAGE_PREFIX = "mermaid/"
DEFAULT_ANNOTATIONS_URI = f"s3://{SOURCE_BUCKET}/mermaid/mermaid_confirmed_annotations.parquet"
IMAGE_PREFIX = "images/mermaid/"
IMAGE_STORE = ImageStore(bucket=DATASET_BUCKET, prefix=IMAGE_PREFIX, owner="segdata")
FEATURE_SUFFIX = "_featurevector"
DEFAULT_FEATURE_FAMILY = "efficientnet-v1"
GROUP = "mermaid"
ANNOTATOR = "default"

# The repo root, which holds ``uv.lock`` and the git checkout that built the version.
REPO_ROOT = Path(__file__).resolve().parents[5]


class AnnotationsChanged(Exception):
    """The annotations parquet was rewritten between the HEAD and the GET of one build."""


@dataclass(frozen=True)
class AnnotationsFile:
    """The annotations table and the object version it was read from."""

    table: pd.DataFrame
    etag: str
    last_modified: str


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the MERMAID build flags; ``workdir`` is unused because the build stages under ``--stage-dir``."""
    del workdir
    parser.add_argument("--annotations-uri", default=DEFAULT_ANNOTATIONS_URI)
    parser.add_argument(
        "--annotation-date-cutoff",
        required=True,
        help="YYYY-MM-DD; keep annotations updated through the end of this day (UTC), inclusive",
    )
    parser.add_argument("--parent-version", default=None, help="Staged version to reuse rows from")
    parser.add_argument(
        "--parent-manifest-sha256", default=None, help="Manifest sha256 of --parent-version"
    )
    parser.add_argument(
        "--feature-family",
        default=DEFAULT_FEATURE_FAMILY,
        help="Family the staged feature set is published under (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Keep images that have no source feature vector, without a feature",
    )
    parser.add_argument(
        "--extractor-spec",
        type=Path,
        required=True,
        help="JSON file: class_path, crop_size, feature_dim, weights_uri, weights_sha256",
    )


def _split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"not an s3:// object URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_annotations(client: Any, uri: str) -> AnnotationsFile:
    """HEAD the parquet, then GET exactly that version.

    Raises:
        AnnotationsChanged: The object was replaced between the HEAD and the GET.
    """
    bucket, key = _split_uri(uri)
    head = client.head_object(Bucket=bucket, Key=key)
    etag = head["ETag"]
    try:
        body = client.get_object(Bucket=bucket, Key=key, IfMatch=etag)["Body"].read()
    except ClientError as exc:
        if is_precondition_failed(exc):
            raise AnnotationsChanged(
                f"{uri} changed while it was being read; rerun the build"
            ) from exc
        raise
    frame = pq.read_table(io.BytesIO(body)).to_pandas()
    return AnnotationsFile(frame, norm_etag(etag), head["LastModified"].isoformat())


def apply_cutoff(annotations: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, int]:
    """Keep rows updated through the end of the ``cutoff`` day (UTC), inclusive; return them and the drop count."""
    keep = annotations["updated_on"] < pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
    return annotations[keep], int((~keep).sum())


def _image_row(
    image_id: str,
    source_key: str,
    facts: ObjectFacts,
    header: tuple[str, int, int],
    region: tuple[Any, Any],
) -> dict[str, Any]:
    media_type, width, height = header
    return {
        "image_id": image_id,
        "key": f"{image_id}.{EXTENSIONS[media_type]}",
        "etag": facts.etag,
        "size": facts.size,
        "crc64nvme": facts.crc64nvme,
        "media_type": media_type,
        "width": width,
        "height": height,
        "group": GROUP,
        "mask_key": None,
        "mask_etag": None,
        "mask_size": None,
        "mask_crc64nvme": None,
        "source_fingerprint": facts.source_etag,
        "source_name": None,
        "attrs": json.dumps(
            {"source_key": source_key, "region_id": region[0], "region_name": region[1]},
            sort_keys=True,
        ),
    }


def _source_key(row: Mapping[str, Any]) -> str:
    source_key: str = json.loads(row["attrs"])["source_key"]
    return source_key


def _head_source(client: Any, image_id: str) -> tuple[str, Mapping[str, Any]] | None:
    """HEAD ``mermaid/<id>.png``, then ``.jpg``; return the key found and its HEAD, or None."""
    for extension in (".png", ".jpg"):
        key = f"{SOURCE_IMAGE_PREFIX}{image_id}{extension}"
        try:
            head: Mapping[str, Any] = client.head_object(
                Bucket=SOURCE_BUCKET, Key=key, ChecksumMode="ENABLED"
            )
        except ClientError as exc:
            if is_not_found(exc):
                continue
            raise
        return key, head
    return None


def _head_feature(client: Any, image_id: str) -> Mapping[str, Any] | None:
    """HEAD ``mermaid/<id>_featurevector``; return its HEAD, or None when it is absent."""
    try:
        head: Mapping[str, Any] = client.head_object(
            Bucket=SOURCE_BUCKET, Key=_feature_source_key(image_id)
        )
    except ClientError as exc:
        if is_not_found(exc):
            return None
        raise
    return head


def _feature_source_key(image_id: str) -> str:
    return f"{SOURCE_IMAGE_PREFIX}{image_id}{FEATURE_SUFFIX}"


def probe_feature(
    client: Any, image_id: str, points: Sequence[Point], feature_dim: int
) -> tuple[str, VectorFacts] | None:
    """HEAD then read the image's source vector; return its source ETag and facts, or None if absent.

    Raises:
        SourceChanged: The vector was replaced between its HEAD and its read.
    """
    head = _head_feature(client, image_id)
    if head is None:
        return None
    key = _feature_source_key(image_id)
    facts = read_vector(client, SOURCE_BUCKET, key, head["ETag"], points, feature_dim)
    return norm_etag(head["ETag"]), facts


def feature_entries(
    found: Mapping[str, tuple[str, VectorFacts]], family: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the ``features.jsonl`` lines and the feature index rows, in ``image_id`` order."""
    plan, index = [], []
    for image_id in sorted(found):
        src_etag, facts = found[image_id]
        key = feature_file_key(IMAGE_STORE, DATASET_BUCKET, family, image_id)
        plan.append(
            {
                "src_bucket": SOURCE_BUCKET,
                "src_key": _feature_source_key(image_id),
                "src_etag": src_etag,
                "etag": facts.etag,
                "dst_bucket": DATASET_BUCKET,
                "dst_key": key,
                "size": facts.size,
                "crc64nvme": facts.crc64nvme,
            }
        )
        index.append(
            {
                "image_id": image_id,
                "key": key,
                "etag": facts.etag,
                "size": facts.size,
                "crc64nvme": facts.crc64nvme,
                "n_points": facts.n_points,
            }
        )
    return plan, index


def check_features(
    image_ids: Sequence[str],
    found: Mapping[str, tuple[str, VectorFacts] | None],
    allow_missing: bool,
) -> dict[str, tuple[str, VectorFacts]]:
    """Return the images' found vectors, refusing missing ones unless allowed and any refused one.

    Raises:
        FeatureVectorsMissing: An image has no source vector and ``allow_missing`` is false.
        FeatureVectorsInvalid: A vector is unreadable, of the wrong width or lacks a point.
    """
    missing = [i for i in image_ids if found[i] is None]
    if missing and not allow_missing:
        raise FeatureVectorsMissing(refusal("have no source feature vector", missing))
    present = {i: f for i, f in found.items() if f is not None}
    refuse_invalid(
        {i: facts for i, (_, facts) in present.items()},
        {i: f"s3://{SOURCE_BUCKET}/{_feature_source_key(i)}" for i in present},
    )
    return present


def _probe_features(
    client: Any, points: Mapping[str, Sequence[Point]], feature_dim: int, workers: int
) -> dict[str, tuple[str, VectorFacts] | None]:
    image_ids = sorted(points)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = pool.map(lambda i: probe_feature(client, i, points[i], feature_dim), image_ids)
        return dict(zip(image_ids, found, strict=True))


def stage_features(
    stage_dir: Path,
    version: str,
    family: str,
    plan: Sequence[Mapping[str, Any]],
    index: Sequence[Mapping[str, Any]],
    extractor: Mapping[str, Any],
    allow_missing: bool,
) -> None:
    """Stage ``features.jsonl`` and, under ``features/<family>/``, the index, spec and ``stage.json``.

    ``features.jsonl`` sits beside ``copies.jsonl``; ``stage.json`` records ``allow_missing``.
    """
    write_copies(stage_dir, DATASET, version, plan, FEATURES_NAME)
    prefix = f"{dataset_prefix(DATASET, version)}features/{family}/"
    rows = pa.Table.from_pylist(list(index), schema=FEATURE_INDEX_SCHEMA)
    stage_bytes(stage_dir, prefix + FEATURE_INDEX_NAME, table_bytes(rows))
    stage_bytes(stage_dir, prefix + EXTRACTOR_SPEC_NAME, extractor_spec_bytes(extractor))
    stage_bytes(
        stage_dir,
        prefix + FEATURE_STAGE_NAME,
        (json.dumps({"allow_missing": allow_missing}, sort_keys=True) + "\n").encode(),
    )


def probe_image(
    client: Any,
    image_id: str,
    region: tuple[Any, Any],
    parent_rows: Mapping[str, dict[str, Any]],
) -> ImageOutcome:
    """Decide one image from its source HEAD and, unless a parent row matches, its bytes."""
    found = _head_source(client, image_id)
    if found is None:
        return ImageOutcome(image_id, dropped="image_deleted_at_source")
    source_key, head = found
    return probe_source_image(
        client,
        SOURCE_BUCKET,
        image_id,
        source_key,
        head,
        parent_rows,
        lambda facts, header: _image_row(image_id, source_key, facts, header, region),
        found_as_jpg=source_key.endswith(".jpg"),
    )


def first_regions(annotations: pd.DataFrame) -> dict[str, tuple[Any, Any]]:
    """Map each image to the region of its first point by ``point_id``."""
    first = annotations.sort_values(["image_id", "point_id"]).drop_duplicates("image_id")
    return {
        str(row.image_id): (row.region_id, row.region_name) for row in first.itertuples(index=False)
    }


def _gf_id(frame: pd.DataFrame) -> pd.Series:
    return frame["growth_form_id"].fillna("")


def _parent_label_ids(parent_labels: pa.Table) -> dict[tuple[str, str], int]:
    """Map the parent's ``(benthic_attribute_id, growth_form_name or '')`` to its ``label_id``."""
    return {
        (row["mermaid_benthic_attribute_id"], row["mermaid_growth_form"] or ""): row["label_id"]
        for row in parent_labels.to_pylist()
    }


def _assign_label_ids(pairs: pd.DataFrame, parent_labels: pa.Table | None) -> list[int]:
    """Reuse the parent's id for each matching pair; number the rest upward from its maximum."""
    known = {} if parent_labels is None else _parent_label_ids(parent_labels)
    next_id = max(known.values(), default=0) + 1
    ids = []
    for ba_id, gf_name in zip(
        pairs["benthic_attribute_id"], pairs["growth_form_name"].fillna(""), strict=True
    ):
        label_id = known.get((ba_id, gf_name))
        if label_id is None:
            label_id, next_id = next_id, next_id + 1
        ids.append(label_id)
    return ids


def with_label_ids(
    annotations: pd.DataFrame, parent_labels: pa.Table | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give each BA+GF pair a ``label_id``, in ``(benthic_attribute_id, growth_form_id or '')`` order.

    Without ``parent_labels`` the pairs are numbered 1..N. With the parent's ``labels.parquet``, a
    pair keeps the parent's id and a new pair is numbered upward from the parent's maximum id.

    Returns the annotations with a ``label_id`` column and the labels frame.
    """
    frame = annotations.assign(_gf=_gf_id(annotations))
    pairs = (
        frame[["benthic_attribute_id", "_gf", "benthic_attribute_name", "growth_form_name"]]
        .drop_duplicates(["benthic_attribute_id", "_gf"])
        .sort_values(["benthic_attribute_id", "_gf"])
        .reset_index(drop=True)
    )
    pairs["label_id"] = _assign_label_ids(pairs, parent_labels)
    ids = pairs.set_index(["benthic_attribute_id", "_gf"])["label_id"]
    keyed = frame.set_index(["benthic_attribute_id", "_gf"]).index
    labelled = annotations.assign(label_id=ids.reindex(keyed).to_numpy())
    return labelled, pairs


def labels_table(pairs: pd.DataFrame) -> pa.Table:
    """Build ``labels.parquet`` from the numbered pairs; the name keeps a trailing ``::`` with no growth form."""
    gf_names = (
        pairs["growth_form_name"].astype(object).where(pairs["growth_form_name"].notna(), None)
    )
    return pa.table(
        {
            "label_id": pairs["label_id"].tolist(),
            "name": [
                f"{ba}::{gf or ''}"
                for ba, gf in zip(pairs["benthic_attribute_name"], gf_names, strict=True)
            ],
            "mermaid_benthic_attribute_id": pairs["benthic_attribute_id"].tolist(),
            "mermaid_benthic_attribute": pairs["benthic_attribute_name"].tolist(),
            "mermaid_growth_form": [gf or "" for gf in gf_names],
            "mapping_source": ["identity"] * len(pairs),
        },
        schema=LABELS_SCHEMA,
    )


def labels_sha256(pairs: pd.DataFrame) -> str:
    """Hash the sorted pair list as compact JSON of ``[ba_id, gf_id or '', ba_name, gf_name or '']``."""
    entries = [
        [ba_id, gf_id, ba_name, gf_name or ""]
        for ba_id, gf_id, ba_name, gf_name in zip(
            pairs["benthic_attribute_id"],
            pairs["_gf"],
            pairs["benthic_attribute_name"],
            pairs["growth_form_name"],
            strict=True,
        )
    ]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def points_table(labelled: pd.DataFrame) -> pa.Table:
    """Build ``points.parquet`` rows from annotations carrying ``label_id``."""
    return pa.table(
        {
            "image_id": labelled["image_id"].astype(str).tolist(),
            "row": labelled["row"].tolist(),
            "col": labelled["col"].tolist(),
            "label_id": labelled["label_id"].tolist(),
            "annotator": [ANNOTATOR] * len(labelled),
        },
        schema=POINTS_SCHEMA,
    )


def label_counts_table(labelled: pd.DataFrame) -> pa.Table:
    """Count points per ``(image_id, label_id)``."""
    counts = labelled.groupby(["image_id", "label_id"]).size().reset_index(name="n")
    return pa.table(
        {
            "image_id": counts["image_id"].astype(str).tolist(),
            "label_id": counts["label_id"].tolist(),
            "n": counts["n"].tolist(),
        },
        schema=LABEL_COUNTS_SCHEMA,
    )


def load_parent_rows(stage_dir: Path, version: str | None) -> dict[str, dict[str, Any]]:
    """Read the staged parent's image rows by ``image_id``; empty without a parent."""
    if version is None:
        return {}
    path = cache_path(
        stage_dir, DATASET_BUCKET, dataset_prefix(DATASET, version) + "images.parquet"
    )
    return {row["image_id"]: row for row in pq.read_table(path).to_pylist()}


def load_parent_labels(stage_dir: Path, version: str | None) -> pa.Table | None:
    """Read the staged parent's ``labels.parquet``; None without a parent."""
    if version is None:
        return None
    path = cache_path(
        stage_dir, DATASET_BUCKET, dataset_prefix(DATASET, version) + "labels.parquet"
    )
    return pq.read_table(path)


def _parent_refs(args: argparse.Namespace) -> list[ParentRef]:
    if (args.parent_version is None) != (args.parent_manifest_sha256 is None):
        raise ValueError("--parent-version and --parent-manifest-sha256 go together")
    if args.parent_version is None:
        return []
    return [
        ParentRef(
            dataset=DATASET,
            version=args.parent_version,
            manifest_sha256=args.parent_manifest_sha256,
        )
    ]


def _command_line(args: argparse.Namespace) -> str:
    parts = [
        "segdata build mermaid",
        f"--stage-dir {args.stage_dir}",
        f"--annotations-uri {args.annotations_uri}",
        f"--annotation-date-cutoff {args.annotation_date_cutoff}",
    ]
    if args.parent_version is not None:
        parts.append(f"--parent-version {args.parent_version}")
        parts.append(f"--parent-manifest-sha256 {args.parent_manifest_sha256}")
    parts.append(f"--feature-family {args.feature_family}")
    parts.append(f"--extractor-spec {args.extractor_spec}")
    if args.allow_missing_features:
        parts.append("--allow-missing-features")
    return " ".join(parts)


def _probe_all(
    client: Any,
    regions: Mapping[str, tuple[Any, Any]],
    parent_rows: Mapping[str, dict[str, Any]],
    workers: int,
) -> list[ImageOutcome]:
    image_ids = sorted(regions)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda i: probe_image(client, i, regions[i], parent_rows), image_ids))


def _dropped_counts(
    n_after_cutoff: int, outcomes: Sequence[ImageOutcome], points_per_image: Mapping[str, int]
) -> dict[str, int]:
    deleted = [o.image_id for o in outcomes if o.dropped == "image_deleted_at_source"]
    return {
        "annotation_after_cutoff": n_after_cutoff,
        "image_deleted_at_source": len(deleted),
        "points_of_deleted_images": sum(points_per_image[i] for i in deleted),
        "image_unreadable_header": sum(o.dropped == "image_unreadable_header" for o in outcomes),
    }


def _revision(file: AnnotationsFile, cutoff: str) -> str:
    return json.dumps(
        {"etag": file.etag, "last_modified": file.last_modified, "cutoff": cutoff},
        sort_keys=True,
        separators=(",", ":"),
    )


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a MERMAID points dataset version and its copy plan under ``args.stage_dir``.

    Raises:
        ValueError: Only one of the parent flags is given, or a URI is not ``s3://``.
        AnnotationsChanged: The annotations parquet was replaced during the read.
        SourceChanged: A source image or vector was replaced between its HEAD and its read.
        PublishRefused: The git tree is dirty or unpushed.
        FeatureVectorsMissing: An image has no feature vector and missing ones are not allowed.
        FeatureVectorsInvalid: A feature vector is unreadable, of another width than the
            extractor's ``feature_dim``, or lacks one of its image's points.
    """
    parents = _parent_refs(args)
    extractor = read_extractor_spec(args.extractor_spec)
    feature_store_prefix(feature_store_name(IMAGE_STORE, DATASET_BUCKET), args.feature_family)
    git = probe_git(REPO_ROOT)
    refuse_unpublishable(git, REPO_ROOT)
    source = read_annotations(client, args.annotations_uri)
    annotations, n_after_cutoff = apply_cutoff(source.table, args.annotation_date_cutoff)
    parent_rows = load_parent_rows(args.stage_dir, args.parent_version)
    outcomes = _probe_all(client, first_regions(annotations), parent_rows, args.workers)
    kept = {o.image_id for o in outcomes if o.row is not None}
    points_per_image = annotations["image_id"].astype(str).value_counts().to_dict()
    dropped = _dropped_counts(n_after_cutoff, outcomes, points_per_image)
    labelled, pairs = with_label_ids(
        annotations[annotations["image_id"].astype(str).isin(kept)],
        load_parent_labels(args.stage_dir, args.parent_version),
    )
    points = points_table(labelled)
    found = _probe_features(client, points_by_image(points), extractor["feature_dim"], args.workers)
    features = check_features(sorted(kept), found, args.allow_missing_features)
    rows = [o.row for o in outcomes if o.row is not None]
    version = next_version(args.stage_dir, DATASET_BUCKET, DATASET, git_sha=git.sha, client=client)
    manifest, _ = stage_dataset(
        args.stage_dir,
        parent_dirs=(args.stage_dir,),
        bucket=DATASET_BUCKET,
        dataset=DATASET,
        version=version,
        kind="points",
        tables={
            "images.parquet": pa.Table.from_pylist(rows, schema=IMAGES_SCHEMA),
            "points.parquet": points,
            "labels.parquet": labels_table(pairs),
            "stats/label_counts.parquet": label_counts_table(labelled),
        },
        image_store=IMAGE_STORE,
        etl_package="segdata-etl",
        etl_command=_command_line(args),
        source=SourceInfo(
            type="mermaid-confirmed-annotations",
            uri=args.annotations_uri,
            revision=_revision(source, args.annotation_date_cutoff),
            citation=None,
            license=None,
        ),
        parents=parents,
        label_mapping=LabelMapping(
            source="identity",
            sha256=labels_sha256(pairs),
            n_labels=len(pairs),
            n_mapped=len(pairs),
        ),
        dropped=dropped,
        repo_dir=REPO_ROOT,
        git=git,
    )
    plan = copy_plan(rows, src_bucket=SOURCE_BUCKET, src_key=_source_key, dst_prefix=IMAGE_PREFIX)
    write_copies(args.stage_dir, DATASET, version, plan)
    feature_plan, feature_index = feature_entries(features, args.feature_family)
    stage_features(
        args.stage_dir,
        version,
        args.feature_family,
        feature_plan,
        feature_index,
        extractor,
        args.allow_missing_features,
    )
    jpg_ids = sorted(o.image_id for o in outcomes if o.found_as_jpg and o.row is not None)
    if jpg_ids:
        logger.info("images found only as .jpg keys: %s", jpg_ids)
    logger.info(
        "staged mermaid %s: %d images, %d points, %d copies, %d feature copies, "
        "%d found only as .jpg, dropped %s",
        version,
        manifest.stats.n_images,
        manifest.stats.n_points,
        len(plan),
        len(feature_plan),
        len(jpg_ids),
        dropped,
    )
    return 0
