"""Write tiny synced dataset versions into a cache directory, for ``open_version``."""

from __future__ import annotations

import hashlib
import io
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from numpy.typing import NDArray
from PIL import Image

from segdata_spec import DATASET_BUCKET
from segdata_spec._marker import SyncMarker, dump_marker
from segdata_spec.git import GitState, Publishable
from segdata_spec.layout import resize_id, synced_marker_dir
from segdata_spec.manifest import (
    EtlInfo,
    FileEntry,
    ImageStore,
    Manifest,
    Pin,
    SourceInfo,
    Stats,
    dump_manifest,
    manifest_sha256,
)
from segdata_spec.read import DatasetVersion, open_version
from segdata_spec.resize import ResizeSpec
from segdata_spec.schema import (
    IMAGES_SCHEMA,
    LABEL_COUNTS_SCHEMA,
    LABELS_SCHEMA,
    POINTS_SCHEMA,
    conform,
    validate_dataset_tables,
)
from segdata_spec.stage import stage_taxonomy

BUCKET = DATASET_BUCKET
IMAGE_BUCKET = "img-bucket"
IMAGE_PREFIX = "imgs/"
VERSION = "20260101-abc1234"


@dataclass(frozen=True)
class ImageSpec:
    """One image of a fixture version; ``mask`` (a dense label map) sets its size when given.

    ``pixels`` (an ``(h, w, 3)`` uint8 array) is stored losslessly under the ``.jpg`` key; ``attrs``
    is the JSON string of the ``images.parquet`` ``attrs`` column.
    """

    image_id: str
    group: str = "g1"
    height: int = 8
    width: int = 8
    mask: NDArray[np.integer[Any]] | None = None
    pixels: NDArray[np.uint8] | None = None
    source_name: str | None = None
    attrs: str | None = None


PointRow = tuple[str, int, int, int] | tuple[str, int, int, int, str]


def _size(spec: ImageSpec) -> tuple[int, int]:
    if spec.pixels is not None:
        return spec.pixels.shape[0], spec.pixels.shape[1]
    if spec.mask is not None:
        return spec.mask.shape[0], spec.mask.shape[1]
    return spec.height, spec.width


def _write_files(cache_dir: Path, images: Sequence[ImageSpec]) -> dict[str, int]:
    """Write each image as JPEG (and mask as PNG); return the file sizes by key."""
    sizes: dict[str, int] = {}
    root = cache_dir / IMAGE_BUCKET / IMAGE_PREFIX
    root.mkdir(parents=True, exist_ok=True)
    for spec in images:
        height, width = _size(spec)
        jpg = root / f"{spec.image_id}.jpg"
        if spec.pixels is None:
            Image.fromarray(np.full((height, width, 3), 128, np.uint8)).save(jpg, "JPEG")
        else:
            Image.fromarray(spec.pixels).save(jpg, "PNG")
        sizes[jpg.name] = jpg.stat().st_size
        if spec.mask is not None:
            png = root / f"{spec.image_id}.png"
            Image.fromarray(spec.mask).save(png, "PNG")
            sizes[png.name] = png.stat().st_size
    return sizes


def _images_table(
    images: Sequence[ImageSpec], sizes: Mapping[str, int], etags: Mapping[str, str], dense: bool
) -> pa.Table:
    n = len(images)
    nulls = pa.nulls(n, pa.string())
    return conform(
        pa.table(
            {
                "image_id": [s.image_id for s in images],
                "key": [f"{s.image_id}.jpg" for s in images],
                "etag": [etags[f"{s.image_id}.jpg"] for s in images],
                "size": [sizes[f"{s.image_id}.jpg"] for s in images],
                "crc64nvme": nulls,
                "media_type": ["image/jpeg"] * n,
                "width": [_size(s)[1] for s in images],
                "height": [_size(s)[0] for s in images],
                "group": [s.group for s in images],
                "mask_key": [f"{s.image_id}.png" for s in images] if dense else nulls,
                "mask_etag": [etags[f"{s.image_id}.png"] for s in images] if dense else nulls,
                "mask_size": (
                    [sizes[f"{s.image_id}.png"] for s in images]
                    if dense
                    else pa.nulls(n, pa.int64())
                ),
                "mask_crc64nvme": nulls,
                "source_fingerprint": ["fp"] * n,
                "source_name": pa.array([s.source_name for s in images], pa.string()),
                "attrs": pa.array([s.attrs for s in images], pa.string()),
            }
        ),
        IMAGES_SCHEMA,
    )


def _labels_table(labels: Mapping[int, str] | Sequence[Mapping[str, Any]]) -> pa.Table:
    rows = (
        [{"label_id": i, "name": name} for i, name in labels.items()]
        if isinstance(labels, Mapping)
        else [dict(row) for row in labels]
    )
    defaults: dict[str, Any] = {"mapping_source": "curated"}
    columns = {
        name: [row.get(name, defaults.get(name)) for row in rows] for name in LABELS_SCHEMA.names
    }
    return conform(pa.table(columns), LABELS_SCHEMA)


def _points_table(points: Sequence[PointRow]) -> pa.Table:
    rows = [(*p, "default") if len(p) == 4 else p for p in points]
    columns = {name: [r[i] for r in rows] for i, name in enumerate(POINTS_SCHEMA.names)}
    return conform(pa.table(columns), POINTS_SCHEMA)


def _derived_counts(
    images: Sequence[ImageSpec], points: Sequence[PointRow]
) -> list[tuple[str, int, int]]:
    """Count labels per image from the points, or from the non-zero mask pixels."""
    counts: Counter[tuple[str, int]] = Counter()
    if points:
        counts.update((p[0], p[3]) for p in points)
    else:
        for spec in images:
            if spec.mask is None:
                continue
            channel = spec.mask if spec.mask.ndim == 2 else spec.mask[..., 0]
            for value in np.unique(channel[channel > 0]):
                counts[(spec.image_id, int(value))] = int((channel == value).sum())
    return [(image_id, label_id, n) for (image_id, label_id), n in counts.items()]


def _counts_table(rows: Sequence[tuple[str, int, int]]) -> pa.Table:
    columns = {name: [r[i] for r in rows] for i, name in enumerate(LABEL_COUNTS_SCHEMA.names)}
    return conform(pa.table(columns), LABEL_COUNTS_SCHEMA)


def write_version(
    tmp_path: Path,
    *,
    kind: Literal["points", "dense"],
    images: Sequence[ImageSpec],
    labels: Mapping[int, str] | Sequence[Mapping[str, Any]],
    points: Sequence[PointRow] = (),
    label_counts: Sequence[tuple[str, int, int]] | None = None,
    name: str = "ds",
    version: str = VERSION,
) -> tuple[Path, str, Pin]:
    """Write a dataset version into ``<tmp_path>/cache`` and return what ``open_version`` takes.

    Args:
        tmp_path: Directory that receives the ``cache`` folder.
        kind: ``"points"`` writes ``points.parquet``; ``"dense"`` writes a PNG mask per image.
        images: The images (dense images carry their mask).
        labels: ``label_id -> name``, or full ``labels.parquet`` row mappings.
        points: ``(image_id, row, col, label_id[, annotator])``; the annotator defaults to
            ``"default"``.
        label_counts: ``(image_id, label_id, n)`` rows; derived from the points or masks if omitted.
        name: Dataset name.
        version: Version string.

    Returns:
        ``(cache_dir, bucket, pin)``.
    """
    cache_dir = tmp_path / "cache"
    sizes = _write_files(cache_dir, images)
    # Real MD5 ETags let ``sync`` verify the staged images and masks from the cache.
    etags = {name: _md5(cache_dir / IMAGE_BUCKET / IMAGE_PREFIX / name) for name in sizes}
    tables = {
        "images.parquet": _images_table(images, sizes, etags, kind == "dense"),
        "labels.parquet": _labels_table(labels),
        "stats/label_counts.parquet": _counts_table(
            _derived_counts(images, points) if label_counts is None else label_counts
        ),
    }
    if kind == "points":
        tables["points.parquet"] = _points_table(points)
    validate_dataset_tables(kind, tables)

    prefix = cache_dir / BUCKET / "datasets" / name / version
    files: dict[str, FileEntry] = {}
    for file_name, table in tables.items():
        path = prefix / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        data = path.read_bytes()
        files[file_name] = FileEntry(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=table.num_rows
        )
    manifest = Manifest(
        schema_version=1,
        dataset=name,
        version=version,
        kind=kind,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        image_store=ImageStore(bucket=IMAGE_BUCKET, prefix=IMAGE_PREFIX, owner="segdata"),
        etl=EtlInfo(
            package="segdata-etl",
            command=f"segdata build {name}",
            git_sha="abc1234",
            git_dirty=False,
            uv_lock_sha256="0" * 64,
        ),
        source=SourceInfo(type="test", uri="s3://source", revision="r1"),
        files=files,
        stats=Stats(
            n_images=len(images),
            n_points=len(points),
            n_groups=len({s.group for s in images}),
            image_bytes=sum(sizes.values()),
            dropped={},
        ),
    )
    data = dump_manifest(manifest)
    (prefix / "manifest.json").write_bytes(data)
    pin = Pin(name=name, version=version, manifest_sha256=manifest_sha256(data))
    write_sync_marker(cache_dir, BUCKET, pin)
    return cache_dir, BUCKET, pin


def write_sync_marker(
    cache_dir: Path, bucket: str, pin: Pin, resize: ResizeSpec | None = None
) -> None:
    """Mark a hand-written cache version as synced with every group, as ``sync`` would."""
    directory = synced_marker_dir(cache_dir, resize_id(resize), bucket, pin.name, pin.version)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "marker.json").write_bytes(dump_marker(SyncMarker(resize, None)))


def open_written(tmp_path: Path, **kwargs: Any) -> DatasetVersion:
    """Write a version with :func:`write_version` and open it."""
    cache_dir, bucket, pin = write_version(tmp_path, **kwargs)
    return open_version(cache_dir, bucket, pin)


def _client_error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


@dataclass
class FakeS3:
    """The object surface of a boto3 S3 client over ``(bucket, key) -> bytes``."""

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def get_object(self, Bucket: str, Key: str, IfMatch: str | None = None) -> dict[str, Any]:
        data = self.objects.get((Bucket, Key))
        if data is None:
            raise _client_error("NoSuchKey", 404, "GetObject")
        if IfMatch is not None and hashlib.md5(data).hexdigest() != IfMatch.strip('"'):
            raise _client_error("PreconditionFailed", 412, "GetObject")
        return {"Body": io.BytesIO(data)}

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", 404, "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **_: Any) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = Body
        return {}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def publish_points(
    tmp_path: Path,
    *,
    images: Sequence[ImageSpec],
    labels: Mapping[int, str] | Sequence[Mapping[str, Any]],
    points: Sequence[PointRow],
) -> tuple[FakeS3, Pin]:
    """Serve a points version; the images' ETags are rewritten to the MD5 of the bytes."""
    cache_dir, _, pin = write_version(
        tmp_path, kind="points", images=images, labels=labels, points=points
    )
    prefix = cache_dir / BUCKET / "datasets" / pin.name / pin.version
    manifest = Manifest.model_validate_json((prefix / "manifest.json").read_bytes())
    store = manifest.image_store
    table = pq.read_table(prefix / "images.parquet")
    etags = [_md5(cache_dir / store.bucket / (store.prefix + k)) for k in table["key"].to_pylist()]
    table = table.set_column(
        table.schema.get_field_index("etag"), "etag", pa.array(etags, pa.string())
    )
    pq.write_table(table, prefix / "images.parquet")
    data = (prefix / "images.parquet").read_bytes()
    files = {
        **manifest.files,
        "images.parquet": FileEntry(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=table.num_rows
        ),
    }
    manifest_bytes = dump_manifest(manifest.model_copy(update={"files": files}))
    (prefix / "manifest.json").write_bytes(manifest_bytes)

    fake = FakeS3()
    key_prefix = f"datasets/{pin.name}/{pin.version}/"
    for file_name in [*files, "manifest.json"]:
        fake.objects[(DATASET_BUCKET, key_prefix + file_name)] = (prefix / file_name).read_bytes()
    for key in table["key"].to_pylist():
        path = cache_dir / store.bucket / (store.prefix + key)
        fake.objects[(store.bucket, store.prefix + key)] = path.read_bytes()
    return fake, Pin(
        name=pin.name, version=pin.version, manifest_sha256=manifest_sha256(manifest_bytes)
    )


def publish_taxonomy_version(
    fake: FakeS3,
    tmp_path: Path,
    version: str,
    benthic_attributes: pa.Table | None = None,
) -> Pin:
    """Stage taxonomy ``mermaid-benthic`` and copy it into ``fake`` under the dataset bucket."""
    repo = tmp_path / "repo"
    stage_dir = Path(tempfile.mkdtemp(dir=tmp_path))
    _, sha = stage_taxonomy(
        stage_dir,
        bucket=DATASET_BUCKET,
        name="mermaid-benthic",
        version=version,
        tables={
            "benthic_attributes.parquet": (
                benthic_attributes
                if benthic_attributes is not None
                else pa.table(
                    {"id": ["1", "2"], "name": ["Hard coral", "Acropora"], "parent_id": [None, "1"]}
                )
            ),
            "growth_forms.parquet": pa.table({"id": ["9"], "name": ["Branching"]}),
        },
        etl_package="segdata-etl",
        etl_command="segdata taxonomy",
        source=SourceInfo(type="test", uri="s3://src/", revision="r1"),
        repo=Publishable(
            root=repo,
            git=GitState(sha="abcdef1234567", dirty=False, pushed=True),
            uv_lock_sha256=hashlib.sha256(b"lock-bytes").hexdigest(),
        ),
    )
    root = stage_dir / DATASET_BUCKET
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        fake.objects[(DATASET_BUCKET, path.relative_to(root).as_posix())] = path.read_bytes()
    return Pin(name="mermaid-benthic", version=version, manifest_sha256=sha)


def write_taxonomy(
    cache_dir: Path, tmp_path: Path, benthic_attributes: pa.Table, version: str = VERSION
) -> Pin:
    """Write taxonomy ``mermaid-benthic`` into ``cache_dir``, as ``sync`` would, and return its pin."""
    fake = FakeS3()
    pin = publish_taxonomy_version(fake, tmp_path, version, benthic_attributes)
    for (bucket, key), data in fake.objects.items():
        path = cache_dir / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return pin
