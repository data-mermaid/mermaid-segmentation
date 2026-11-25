"""Shared build for dense datasets staged from a pinned Hugging Face revision.

The build streams parquet shards in small batches, writes each image and mask to the stage unchanged,
and stages the version with ``stage_dataset``. Nothing is written to S3; ``uploads.jsonl`` lists
the staged files a later promote step uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from segdata_etl.common.copy_plan import StagedFile, stage_bytes, write_uploads
from segdata_etl.common.s3_objects import (
    EXTENSIONS,
    image_size,
    md5_hex,
    sniff_media_type,
)
from segdata_etl.common.stage import next_version
from segdata_etl.label_maps.curated import curated_source, load_curated
from segdata_spec import DATASET_BUCKET
from segdata_spec.git import probe_git, refuse_unpublishable
from segdata_spec.manifest import ImageStore, LabelMapping, SourceInfo
from segdata_spec.s3 import crc64nvme_b64
from segdata_spec.schema import IMAGES_SCHEMA, LABEL_COUNTS_SCHEMA, LABELS_SCHEMA
from segdata_spec.stage import stage_dataset

logger = logging.getLogger(__name__)

BATCH_SIZE = 16
MASK_SUFFIX = "_label.png"

# The repo root, which holds ``uv.lock`` and the git checkout that built the version.
REPO_ROOT = Path(__file__).resolve().parents[5]
# A directory holding ``<dataset>.csv`` in place of the packaged curated maps.
CURATED_ROOT: Path | None = None

_PNG = "image/png"
# Pillow modes that decode to a 2-D integer array of label ids.
_ID_MODES = frozenset({"L", "P", "I;16", "I"})


@dataclass(frozen=True)
class RowInfo:
    """What identifies one source row: its group, file stem and the dataset-specific extras."""

    group: str
    stem: str
    source_name: str
    attrs: dict[str, str]


@dataclass(frozen=True)
class DenseSource:
    """A Hugging Face dataset and the dataset-specific reading of its rows and vocabulary."""

    dataset: str
    repo: str
    revision: str
    license: str
    citation: str
    vocab_file: str
    shard_dir: str  # every parquet under this directory must match shard_pattern
    shard_pattern: re.Pattern[str]  # group 1 is the split name the shard holds
    meta_columns: tuple[str, ...]
    parse_vocab: Callable[[bytes], dict[int, str]]
    describe: Callable[[Mapping[str, Any], str], RowInfo]
    assign_ids: Callable[[Sequence[RowInfo]], dict[tuple[str, str], str]]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def image_prefix(self) -> str:
        """Key prefix of the dataset's objects in the image store."""
        return f"images/{self.dataset}/"


def add_revision_argument(parser: argparse.ArgumentParser, source: DenseSource) -> None:
    """Add ``--revision``, defaulting to the revision pinned in code."""
    parser.add_argument(
        "--revision",
        default=source.revision,
        help="Hugging Face revision to stage (default: %(default)s)",
    )


def _list_repo_files(repo: str, revision: str) -> list[str]:
    return HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)


def select_shards(files: Sequence[str], pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    """Return ``(file, split)`` for each file the pattern matches, sorted by file name."""
    found = [(f, m.group(1)) for f in files if (m := pattern.fullmatch(f))]
    return sorted(found)


def _shards_or_raise(files: Sequence[str], source: DenseSource) -> list[tuple[str, str]]:
    """Return the shards to stage, requiring at least one and no stray parquet in the shard dir.

    Raises:
        ValueError: No file matches the pattern, or a parquet under ``source.shard_dir`` does not.
    """
    shards = select_shards(files, source.shard_pattern)
    if not shards:
        raise ValueError(f"no shard of {source.repo} matches {source.shard_pattern.pattern}")
    matched = {name for name, _ in shards}
    for name in sorted(files):
        if name.startswith(source.shard_dir) and name.endswith(".parquet") and name not in matched:
            raise ValueError(
                f"{name} is under {source.shard_dir} but does not match the shard pattern"
            )
    return shards


def read_header(image: bytes, mask: bytes) -> tuple[str, int, int] | None:
    """Return ``(media_type, width, height)`` of the image, or None if it or the mask is unreadable.

    The image must be JPEG or PNG with a readable size and the mask must be PNG.
    """
    media_type = sniff_media_type(image)
    if media_type is None or sniff_media_type(mask) != _PNG:
        return None
    dims = image_size(image)
    return None if dims is None else (media_type, dims[0], dims[1])


class MaskInvalid(ValueError):
    """A mask cannot serve as a label map for its image."""


def label_pixel_counts(
    mask: bytes, size: tuple[int, int], vocab_ids: Collection[int]
) -> dict[int, int]:
    """Count mask pixels per native label, excluding 0.

    The mask must match ``size`` (width, height), have a mode whose pixel values are label ids, and
    hold only ids in ``vocab_ids`` or 0.

    Raises:
        MaskInvalid: The mask is undecodable, mis-sized, not single-channel, or holds another id.
    """
    try:
        decoded = Image.open(io.BytesIO(mask))
        if decoded.size != size:
            raise MaskInvalid(
                f"mask is {decoded.size[0]}x{decoded.size[1]}, image is {size[0]}x{size[1]}"
            )
        if decoded.mode not in _ID_MODES:
            raise MaskInvalid(f"mask mode {decoded.mode} does not hold label ids")
        values, counts = np.unique(np.asarray(decoded), return_counts=True)
    except OSError as exc:
        raise MaskInvalid(f"mask cannot be decoded: {exc}") from exc
    stray = sorted({int(v) for v in values} - set(vocab_ids) - {0})
    if stray:
        raise MaskInvalid(f"mask holds ids outside the vocabulary: {stray[:5]}")
    return {int(v): int(n) for v, n in zip(values, counts, strict=True) if v != 0}


def _object_fields(data: bytes) -> tuple[str, int, str]:
    return md5_hex(data), len(data), crc64nvme_b64(data)


def image_row(
    image_id: str,
    info: RowInfo,
    image: bytes,
    mask: bytes,
    header: tuple[str, int, int],
) -> dict[str, Any]:
    """Build the ``images.parquet`` row; keys are relative to the dataset's image prefix."""
    media_type, width, height = header
    etag, size, crc = _object_fields(image)
    mask_etag, mask_size, mask_crc = _object_fields(mask)
    return {
        "image_id": image_id,
        "key": f"{info.group}/{info.stem}.{EXTENSIONS[media_type]}",
        "etag": etag,
        "size": size,
        "crc64nvme": crc,
        "media_type": media_type,
        "width": width,
        "height": height,
        "group": info.group,
        "mask_key": f"{info.group}/{info.stem}{MASK_SUFFIX}",
        "mask_etag": mask_etag,
        "mask_size": mask_size,
        "mask_crc64nvme": mask_crc,
        "source_fingerprint": hashlib.sha256(image).hexdigest(),
        "source_name": info.source_name,
        "attrs": json.dumps(info.attrs, sort_keys=True),
    }


def _batches(path: str, columns: Sequence[str]) -> Iterator[list[dict[str, Any]]]:
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=list(columns)):
            yield batch.to_pylist()


def read_infos(
    source: DenseSource, shards: Sequence[tuple[str, str]], revision: str
) -> list[RowInfo]:
    """Describe every row from the metadata columns alone, without reading image bytes."""
    infos: list[RowInfo] = []
    for name, split in shards:
        local = hf_hub_download(source.repo, name, repo_type="dataset", revision=revision)
        for rows in _batches(local, source.meta_columns):
            infos.extend(source.describe(row, split) for row in rows)
    return infos


def _bytes_of(cell: Mapping[str, Any] | None) -> bytes:
    return (cell or {}).get("bytes") or b""


def labels_table(vocab: Mapping[int, str], curated: pa.Table) -> tuple[pa.Table, int]:
    """Join the vocabulary to the curated map on ``label_id``; return the table and the mapped count.

    Raises:
        ValueError: A curated row names a label the vocabulary does not, or spells it differently.
    """
    curated_rows = {r["label_id"]: r for r in curated.to_pylist()}
    for label_id, row in curated_rows.items():
        if vocab.get(label_id) != row["name"]:
            raise ValueError(
                f"curated label {label_id} is named {row['name']!r}; "
                f"the vocabulary has {vocab.get(label_id)!r}"
            )
    ids = sorted(vocab)
    targets = [
        curated_rows[i]["mermaid_benthic_attribute"] if i in curated_rows else None for i in ids
    ]
    table = pa.table(
        {
            "label_id": ids,
            "name": [vocab[i] for i in ids],
            "mermaid_benthic_attribute_id": [None] * len(ids),
            "mermaid_benthic_attribute": targets,
            "mermaid_growth_form": [None] * len(ids),
            "mapping_source": ["curated"] * len(ids),
        },
        schema=LABELS_SCHEMA,
    )
    return table, sum(t is not None for t in targets)


def label_counts_table(counts: Sequence[tuple[str, int, int]]) -> pa.Table:
    """Build ``stats/label_counts.parquet`` from ``(image_id, label_id, n)`` triples."""
    return pa.table(
        {
            "image_id": [c[0] for c in counts],
            "label_id": [c[1] for c in counts],
            "n": [c[2] for c in counts],
        },
        schema=LABEL_COUNTS_SCHEMA,
    )


def _command_line(source: DenseSource, args: argparse.Namespace) -> str:
    return f"segdata build {source.dataset} --stage-dir {args.stage_dir} --revision {args.revision}"


@dataclass
class _Staged:
    rows: list[dict[str, Any]] = field(default_factory=list)
    counts: list[tuple[str, int, int]] = field(default_factory=list)
    files: list[StagedFile] = field(default_factory=list)
    unreadable: int = 0
    mask_invalid: int = 0


def _stage_shard(
    source: DenseSource,
    stage_dir: Path,
    local: str,
    split: str,
    ids: Mapping[tuple[str, str], str],
    vocab_ids: Collection[int],
    out: _Staged,
) -> None:
    columns = (*source.meta_columns, "image", "label")
    for rows in _batches(local, columns):
        for row in rows:
            info = source.describe(row, split)
            image_id = ids[(info.group, info.stem)]
            image, mask = _bytes_of(row["image"]), _bytes_of(row["label"])
            header = read_header(image, mask)
            if header is None:
                logger.warning("dropping image %s: unreadable image or mask header", image_id)
                out.unreadable += 1
                continue
            try:
                label_counts = label_pixel_counts(mask, header[1:], vocab_ids)
            except MaskInvalid as exc:
                logger.warning("dropping image %s: invalid mask: %s", image_id, exc)
                out.mask_invalid += 1
                continue
            record = image_row(image_id, info, image, mask, header)
            stage_bytes(stage_dir, source.image_prefix + record["key"], image)
            stage_bytes(stage_dir, source.image_prefix + record["mask_key"], mask)
            out.rows.append(record)
            out.counts.extend((image_id, label, n) for label, n in sorted(label_counts.items()))
            prefix = source.image_prefix
            out.files.append(
                StagedFile(prefix + record["key"], record["size"], record["crc64nvme"], header[0])
            )
            out.files.append(
                StagedFile(
                    prefix + record["mask_key"], record["mask_size"], record["mask_crc64nvme"], _PNG
                )
            )


def run_build(source: DenseSource, args: argparse.Namespace, client: Any) -> int:
    """Stage a dense dataset version from ``args.revision`` under ``args.stage_dir``.

    ``client`` is the real S3 client; it is only read, to keep the version clear of the bucket.

    Raises:
        ValueError: No shard matches, a stray parquet sits in the shard directory, image ids
            collide, or the curated map disagrees with the vocabulary.
        ObjectConflict: A staged file exists with other bytes.
        PublishRefused: The git tree is dirty or unpushed.
    """
    git = probe_git(REPO_ROOT)
    refuse_unpublishable(git, REPO_ROOT)
    shards = _shards_or_raise(_list_repo_files(source.repo, args.revision), source)
    vocab_path = hf_hub_download(
        source.repo, source.vocab_file, repo_type="dataset", revision=args.revision
    )
    vocab = source.parse_vocab(Path(vocab_path).read_bytes())
    labels, n_mapped = labels_table(vocab, load_curated(source.dataset, root=CURATED_ROOT))
    ids = source.assign_ids(read_infos(source, shards, args.revision))
    out = _Staged()
    for name, split in shards:
        local = hf_hub_download(source.repo, name, repo_type="dataset", revision=args.revision)
        _stage_shard(source, args.stage_dir, local, split, ids, vocab.keys(), out)
    dropped = {"image_unreadable_header": out.unreadable, "mask_invalid": out.mask_invalid}
    version = next_version(
        args.stage_dir, DATASET_BUCKET, source.dataset, git_sha=git.sha, client=client
    )
    manifest, _ = stage_dataset(
        args.stage_dir,
        parent_dirs=(args.stage_dir,),
        bucket=DATASET_BUCKET,
        dataset=source.dataset,
        version=version,
        kind="dense",
        tables={
            "images.parquet": pa.Table.from_pylist(out.rows, schema=IMAGES_SCHEMA),
            "labels.parquet": labels,
            "stats/label_counts.parquet": label_counts_table(out.counts),
        },
        image_store=ImageStore(bucket=DATASET_BUCKET, prefix=source.image_prefix, owner="segdata"),
        etl_package="segdata-etl",
        etl_command=_command_line(source, args),
        source=SourceInfo(
            type="huggingface",
            uri=f"hf://datasets/{source.repo}",
            revision=args.revision,
            citation=source.citation,
            license=source.license,
        ),
        label_mapping=LabelMapping(
            source="curated",
            sha256=curated_source(source.dataset, root=CURATED_ROOT).sha256,
            n_labels=len(vocab),
            n_mapped=n_mapped,
        ),
        dropped=dropped,
        repo_dir=REPO_ROOT,
        git=git,
    )
    write_uploads(args.stage_dir, source.dataset, version, out.files)
    logger.info(
        "staged %s %s: %d images, %d files, dropped %s",
        source.dataset,
        version,
        manifest.stats.n_images,
        len(out.files),
        dropped,
    )
    return 0


def stem_of(filename: str) -> str:
    """Return the file name without its directory or extension."""
    return PurePosixPath(filename).stem
