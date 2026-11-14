#!/usr/bin/env python3
"""Find CoralNet sources that duplicate another source with the same image count.

Sources are the image folders under ``s3://<bucket>/<prefix>/s<source_id>/images/``.
Only sources with the same number of image files are compared. Each image is reduced
to a 64-bit difference hash (dHash). A pair is a duplicate when every image matches,
a possible/partial overlap when some do, and not duplicated when none do. Sources
with a unique image count are not duplicated and their files are never downloaded.

Identical bytes (the same S3 ETag) are hashed once. Re-running the same output
directory resumes from ``hash_parts/`` and ``image_hashes.parquet``.

Usage:
    python scripts/coralnet_source_duplicates.py --workers 32
    python scripts/coralnet_source_duplicates.py --limit-sources 20 --output-dir /tmp/dupes
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import boto3
import pandas as pd
from botocore.client import BaseClient
from botocore.config import Config
from PIL import Image

logger = logging.getLogger("coralnet_source_duplicates")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "coralnet_source_duplicates"
DEFAULT_BUCKET = "dev-datamermaid-sm-sources"
DEFAULT_PREFIX = "coralnet-public-images"
DEFAULT_WORKERS = 32
DEFAULT_CHECKPOINT_EVERY = 20_000
DHASH_SIZE = 8

HASH_COLUMNS = ("source_id", "key", "etag", "dhash")
DUPLICATE_COLUMNS = (
    "source_a",
    "source_b",
    "n_images",
    "n_shared",
    "fraction",
    "cluster_id",
    "n_unreadable_a",
    "n_unreadable_b",
)
PARTIAL_COLUMNS = (
    "source_a",
    "source_b",
    "n_images",
    "n_shared",
    "fraction",
    "n_unreadable_a",
    "n_unreadable_b",
)
UNIQUE_COLUMNS = ("source_id", "n_images", "n_peers", "reason")

_thread_local = threading.local()
_MISSING = object()


@dataclass(frozen=True)
class ImageRef:
    source_id: int
    key: str
    etag: str


@dataclass(frozen=True)
class PairOverlap:
    """Multiset intersection of two same-sized sources."""

    source_a: int
    source_b: int
    n_images: int
    n_shared: int
    n_unreadable_a: int
    n_unreadable_b: int

    @property
    def fraction(self) -> float:
        if self.n_images == 0:
            return 0.0
        return round(self.n_shared / self.n_images, 6)

    @property
    def is_duplicate(self) -> bool:
        return self.n_images > 0 and self.n_shared == self.n_images


@dataclass(frozen=True)
class DuplicateReport:
    duplicates: pd.DataFrame
    partial: pd.DataFrame
    unique: pd.DataFrame
    summary: str


def dhash_bytes(data: bytes, size: int = DHASH_SIZE) -> int | None:
    """64-bit difference hash, or None when the bytes are not a readable image.

    JPEGs are decoded at a reduced scale when Pillow can do that, then resized to an ``(size + 1)``
    by ``size`` grayscale thumbnail. Each bit compares one horizontal pair of thumbnail pixels.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.draft("L", (size + 1, size))
            gray = image.convert("L")
            small = gray.resize((size + 1, size), Image.Resampling.BOX)
            pixels = small.tobytes()
    except Exception:
        logger.debug("unreadable image (%d bytes)", len(data), exc_info=True)
        return None
    width = size + 1
    if len(pixels) != width * size:
        return None
    bits = 0
    for row in range(size):
        offset = row * width
        for col in range(size):
            bits = (bits << 1) | (pixels[offset + col] > pixels[offset + col + 1])
    return bits


def parse_image_key(key: str, prefix: str) -> int | None:
    """Return the source id for ``{prefix}/s<id>/images/<file>``.

    Keys outside that exact layout are ignored, including resized copies under ``{prefix}/resized/``
    and files that are not direct children of ``images/``.
    """
    marker = f"{prefix.strip('/')}/s"
    if not key.startswith(marker):
        return None
    source_text, sep, tail = key[len(marker) :].partition("/")
    if not sep or not source_text.isdigit() or not tail.startswith("images/"):
        return None
    name = tail.removeprefix("images/")
    if not name or "/" in name:
        return None
    return int(source_text)


def group_by_count(counts: Mapping[int, int]) -> dict[int, list[int]]:
    """Map an image count to the sorted source ids that have that many files."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for source_id, n_images in counts.items():
        grouped[n_images].append(source_id)
    return {n_images: sorted(source_ids) for n_images, source_ids in grouped.items()}


def pairwise_overlap(group: Mapping[int, Sequence[int | None]]) -> list[PairOverlap]:
    """Pairs in one same-count group whose hash multisets intersect.

    ``None`` entries are unreadable images. They count toward the image total and never count as
    shared. Sources are returned with ``source_a < source_b``.
    """
    if len(group) < 2:
        return []
    lengths = {len(hashes) for hashes in group.values()}
    if len(lengths) != 1:
        raise ValueError(f"same-count group has mixed image counts: {sorted(lengths)}")
    n_images = lengths.pop()
    if n_images == 0:
        return []

    counters: dict[int, Counter[int]] = {}
    unreadable: dict[int, int] = {}
    inverted: dict[int, list[int]] = defaultdict(list)
    for source_id, hashes in group.items():
        counts: Counter[int] = Counter(
            image_hash for image_hash in hashes if image_hash is not None
        )
        counters[source_id] = counts
        unreadable[source_id] = sum(image_hash is None for image_hash in hashes)
        for image_hash in counts:
            inverted[image_hash].append(source_id)

    shared: dict[tuple[int, int], int] = defaultdict(int)
    for image_hash, source_ids in inverted.items():
        if len(source_ids) < 2:
            continue
        ordered = sorted(source_ids)
        for index, left in enumerate(ordered):
            left_count = counters[left][image_hash]
            for right in ordered[index + 1 :]:
                shared[(left, right)] += min(left_count, counters[right][image_hash])

    return [
        PairOverlap(
            source_a=left,
            source_b=right,
            n_images=n_images,
            n_shared=n_shared,
            n_unreadable_a=unreadable[left],
            n_unreadable_b=unreadable[right],
        )
        for (left, right), n_shared in shared.items()
        if n_shared > 0
    ]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, item: int) -> int:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while item != root:
            parent = self._parent[item]
            self._parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def _cluster_ids(pairs: Sequence[PairOverlap]) -> dict[int, int]:
    """Map each fully duplicated source to a 1-based cluster id.

    Clusters are numbered by their smallest source id.
    """
    if not pairs:
        return {}
    union = _UnionFind()
    members: set[int] = set()
    for pair in pairs:
        union.union(pair.source_a, pair.source_b)
        members.add(pair.source_a)
        members.add(pair.source_b)
    grouped: dict[int, list[int]] = defaultdict(list)
    for source_id in members:
        grouped[union.find(source_id)].append(source_id)
    ordered = sorted(
        (sorted(source_ids) for source_ids in grouped.values()), key=lambda ids: ids[0]
    )
    return {
        source_id: cluster_id
        for cluster_id, source_ids in enumerate(ordered, start=1)
        for source_id in source_ids
    }


def _cluster_descriptions(
    pairs: Sequence[PairOverlap], cluster_of: Mapping[int, int]
) -> list[tuple[int, int, list[int]]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    n_images: dict[int, int] = {}
    for pair in pairs:
        cluster_id = cluster_of[pair.source_a]
        n_images[cluster_id] = pair.n_images
    for source_id, cluster_id in cluster_of.items():
        grouped[cluster_id].append(source_id)
    return [
        (cluster_id, n_images[cluster_id], sorted(grouped[cluster_id]))
        for cluster_id in sorted(grouped)
    ]


def _reason_counts(unique: pd.DataFrame) -> dict[str, int]:
    counts = {"unique_count": 0, "empty": 0, "no_shared_images": 0}
    if unique.empty:
        return counts
    for reason in unique["reason"].tolist():
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def format_summary(
    *,
    n_sources: int,
    n_duplicate_sources: int,
    n_partial_sources: int,
    n_duplicate_pairs: int,
    n_partial_pairs: int,
    unique: pd.DataFrame,
    clusters: Sequence[tuple[int, int, list[int]]],
) -> str:
    """Plain-text report header with source counts and duplicate clusters."""
    reasons = _reason_counts(unique)
    n_clusters = len(clusters)
    cluster_word = "cluster" if n_clusters == 1 else "clusters"
    lines = [
        "CoralNet same-count duplicate report",
        "Sources are compared only when they have the same number of image files.",
        "A shared image is an exact 64-bit dHash match. Unreadable images do not match.",
        f"sources: {n_sources}",
        f"duplicate sources: {n_duplicate_sources} ({n_clusters} {cluster_word})",
        f"possible / partial sources: {n_partial_sources}",
        f"not duplicated: {len(unique)}",
        f"  unique image count: {reasons['unique_count']}",
        f"  empty: {reasons['empty']}",
        f"  no shared images: {reasons['no_shared_images']}",
        "",
        "clusters:",
    ]
    if clusters:
        for cluster_id, n_images, members in clusters:
            joined = ", ".join(str(source_id) for source_id in members)
            lines.append(f"  {cluster_id}  n_images={n_images}  sources={joined}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"duplicate pairs: {n_duplicate_pairs} (see duplicates.csv)")
    lines.append(f"possible / partial pairs: {n_partial_pairs} (see partial.csv)")
    lines.append("")
    return "\n".join(lines)


def _pairs_frame(
    pairs: Sequence[PairOverlap],
    columns: Sequence[str],
    cluster_of: Mapping[int, int] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in pairs:
        row: dict[str, object] = {
            "source_a": pair.source_a,
            "source_b": pair.source_b,
            "n_images": pair.n_images,
            "n_shared": pair.n_shared,
            "fraction": pair.fraction,
            "n_unreadable_a": pair.n_unreadable_a,
            "n_unreadable_b": pair.n_unreadable_b,
        }
        if cluster_of is not None:
            row["cluster_id"] = cluster_of[pair.source_a]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(columns))


def build_report(hashes_by_source: Mapping[int, Sequence[int | None]]) -> DuplicateReport:
    """Classify every source from per-image hashes.

    The length of each hash list is that source's image count. ``None`` is an unreadable image.
    Sources whose count is unique, sources with no images, and sources that share no hash with a
    same-sized peer are not duplicated. A source that fully matches one peer is a duplicate even if
    it only partly matches another.
    """
    counts = {source_id: len(hashes) for source_id, hashes in hashes_by_source.items()}
    groups = group_by_count(counts)
    pairs: list[PairOverlap] = []
    for n_images, source_ids in groups.items():
        if n_images == 0 or len(source_ids) < 2:
            continue
        group = {source_id: hashes_by_source[source_id] for source_id in source_ids}
        pairs.extend(pairwise_overlap(group))

    duplicate_pairs = [pair for pair in pairs if pair.is_duplicate]
    partial_pairs = [pair for pair in pairs if not pair.is_duplicate]
    cluster_of = _cluster_ids(duplicate_pairs)
    duplicate_sources = set(cluster_of)
    partial_sources: set[int] = set()
    for pair in partial_pairs:
        partial_sources.add(pair.source_a)
        partial_sources.add(pair.source_b)
    partial_sources -= duplicate_sources
    peers = {
        source_id: len(source_ids) - 1 for source_ids in groups.values() for source_id in source_ids
    }

    unique_rows: list[dict[str, object]] = []
    for source_id, n_images in sorted(counts.items()):
        if source_id in duplicate_sources or source_id in partial_sources:
            continue
        if n_images == 0:
            reason = "empty"
        elif peers[source_id] == 0:
            reason = "unique_count"
        else:
            reason = "no_shared_images"
        unique_rows.append(
            {
                "source_id": source_id,
                "n_images": n_images,
                "n_peers": peers[source_id],
                "reason": reason,
            }
        )

    duplicates = _pairs_frame(duplicate_pairs, DUPLICATE_COLUMNS, cluster_of)
    partial = _pairs_frame(partial_pairs, PARTIAL_COLUMNS, None)
    unique = (
        pd.DataFrame(unique_rows, columns=list(UNIQUE_COLUMNS))
        if unique_rows
        else pd.DataFrame(columns=list(UNIQUE_COLUMNS))
    )
    if not duplicates.empty:
        duplicates = duplicates.sort_values(
            ["cluster_id", "source_a", "source_b"], kind="mergesort"
        ).reset_index(drop=True)
    if not partial.empty:
        partial = partial.sort_values(
            ["fraction", "n_shared", "source_a", "source_b"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    summary = format_summary(
        n_sources=len(counts),
        n_duplicate_sources=len(duplicate_sources),
        n_partial_sources=len(partial_sources),
        n_duplicate_pairs=len(duplicate_pairs),
        n_partial_pairs=len(partial_pairs),
        unique=unique,
        clusters=_cluster_descriptions(duplicate_pairs, cluster_of),
    )
    return DuplicateReport(duplicates=duplicates, partial=partial, unique=unique, summary=summary)


def write_report(out_dir: Path, report: DuplicateReport) -> None:
    """Write ``summary.txt``, ``duplicates.csv``, ``partial.csv`` and ``unique.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report.duplicates.to_csv(out_dir / "duplicates.csv", index=False)
    report.partial.to_csv(out_dir / "partial.csv", index=False)
    report.unique.to_csv(out_dir / "unique.csv", index=False)
    (out_dir / "summary.txt").write_text(report.summary)


def make_s3_client(max_pool_connections: int = 32) -> BaseClient:
    """Boto3 S3 client with adaptive retries for a long listing or download pass."""
    return boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        ),
    )


def _thread_s3() -> BaseClient:
    client = getattr(_thread_local, "s3", None)
    if client is None:
        client = make_s3_client(max_pool_connections=4)
        _thread_local.s3 = client
    return client


def _clean_etag(raw: str | None, key: str) -> str:
    if not raw:
        return f"missing:{key}"
    return raw.strip().strip('"')


def list_source_images(client: BaseClient, bucket: str, prefix: str) -> dict[int, list[ImageRef]]:
    """List image objects under each ``s<source_id>/images/`` folder."""
    full_prefix = prefix.strip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    images: dict[int, list[ImageRef]] = defaultdict(list)
    pages = 0
    n_objects = 0
    n_images = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        pages += 1
        for obj in page.get("Contents", []):
            n_objects += 1
            key = obj["Key"]
            source_id = parse_image_key(key, prefix)
            if source_id is None:
                continue
            images[source_id].append(
                ImageRef(source_id=source_id, key=key, etag=_clean_etag(obj.get("ETag"), key))
            )
            n_images += 1
        if pages % 200 == 0:
            logger.info("listed %d pages, %d objects, %d source images", pages, n_objects, n_images)
    logger.info(
        "listing done: %d pages, %d objects, %d sources, %d images",
        pages,
        n_objects,
        len(images),
        n_images,
    )
    return dict(images)


def apply_source_limit(
    images: Mapping[int, list[ImageRef]], limit: int | None
) -> dict[int, list[ImageRef]]:
    """Keep the lowest ``limit`` source ids, for a smoke run."""
    if limit is None:
        return dict(images)
    keep = sorted(images)[:limit]
    return {source_id: images[source_id] for source_id in keep}


def _empty_hash_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": pd.Series(dtype="int32"),
            "key": pd.Series(dtype="string"),
            "etag": pd.Series(dtype="string"),
            "dhash": pd.Series(dtype="string"),
        }
    )


def _normalize_hash_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(HASH_COLUMNS)].copy()
    out["source_id"] = out["source_id"].astype("int32")
    out["key"] = out["key"].astype("string")
    out["etag"] = out["etag"].astype("string")
    out["dhash"] = out["dhash"].astype("string")
    return out


def _decode_hash(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return int(str(value), 16)


def _encode_hash(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value:016x}"


def load_hash_rows(out_dir: Path) -> pd.DataFrame:
    """Rows already hashed.

    Later part files override an older consolidated parquet.
    """
    paths: list[Path] = []
    final = out_dir / "image_hashes.parquet"
    if final.is_file():
        paths.append(final)
    parts_dir = out_dir / "hash_parts"
    if parts_dir.is_dir():
        paths.extend(sorted(path for path in parts_dir.glob("part-*.parquet") if path.is_file()))
    if not paths:
        return _empty_hash_frame()
    frames = [_normalize_hash_frame(pd.read_parquet(path)) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(["source_id", "key"], keep="last")
    return combined.reset_index(drop=True)


def _etag_cache(frame: pd.DataFrame) -> dict[str, int | None]:
    """Map ETag to hash.

    A later successful hash replaces an earlier unreadable one.
    """
    cache: dict[str, int | None] = {}
    if frame.empty:
        return cache
    for etag, raw_hash in zip(frame["etag"].tolist(), frame["dhash"].tolist(), strict=True):
        decoded = _decode_hash(raw_hash)
        current = cache.get(str(etag), _MISSING)
        if current is _MISSING or (current is None and decoded is not None):
            cache[str(etag)] = decoded
    return cache


def _done_hashes(frame: pd.DataFrame) -> dict[tuple[int, str], int | None]:
    done: dict[tuple[int, str], int | None] = {}
    if frame.empty:
        return done
    for source_id, key, raw_hash in zip(
        frame["source_id"].tolist(),
        frame["key"].tolist(),
        frame["dhash"].tolist(),
        strict=True,
    ):
        done[(int(source_id), str(key))] = _decode_hash(raw_hash)
    return done


def _next_part_index(parts_dir: Path) -> int:
    highest = -1
    if not parts_dir.is_dir():
        return 0
    for path in parts_dir.glob("part-*.parquet"):
        suffix = path.stem.removeprefix("part-")
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def _write_part(parts_dir: Path, rows: list[dict[str, object]], index: int) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    frame = _normalize_hash_frame(pd.DataFrame(rows, columns=list(HASH_COLUMNS)))
    dest = parts_dir / f"part-{index:05d}.parquet"
    tmp = parts_dir / f".part-{index:05d}.parquet.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.replace(dest)


class _Checkpointer:
    def __init__(self, out_dir: Path, every: int) -> None:
        self.parts_dir = out_dir / "hash_parts"
        self.every = every
        self.buffer: list[dict[str, object]] = []
        self.next_index = _next_part_index(self.parts_dir)

    def add(self, row: dict[str, object]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        _write_part(self.parts_dir, self.buffer, self.next_index)
        logger.info("checkpoint %d hashes -> part-%05d", len(self.buffer), self.next_index)
        self.next_index += 1
        self.buffer = []


def consolidate_hashes(out_dir: Path) -> pd.DataFrame:
    """Rewrite ``image_hashes.parquet`` from the checkpoint parts and any previous copy."""
    frame = load_hash_rows(out_dir)
    dest = out_dir / "image_hashes.parquet"
    tmp = out_dir / ".image_hashes.parquet.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.replace(dest)
    logger.info("wrote %d hash rows to %s", len(frame), dest)
    return frame


def _download_and_hash(bucket: str, key: str) -> tuple[bool, int | None]:
    """Return ``(definitive, dhash)``.

    S3 errors are not definitive and can be retried.
    """
    try:
        body = _thread_s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        logger.warning("download failed s3://%s/%s: %s", bucket, key, exc)
        return False, None
    return True, dhash_bytes(body)


def _run_downloads(
    bucket: str,
    jobs: Sequence[tuple[str, str]],
    workers: int,
    on_result: Callable[[str, bool, int | None], None],
) -> None:
    if not jobs:
        return
    total = len(jobs)
    if workers <= 1:
        for done, (etag, key) in enumerate(jobs, start=1):
            on_result(etag, *_download_and_hash(bucket, key))
            if done % 1000 == 0 or done == total:
                logger.info("hashed %d / %d downloads", done, total)
        return

    max_in_flight = max(1, workers * 2)
    job_iter = iter(jobs)
    in_flight: dict[Future[tuple[bool, int | None]], str] = {}

    def submit(pool: ThreadPoolExecutor) -> bool:
        try:
            etag, key = next(job_iter)
        except StopIteration:
            return False
        in_flight[pool.submit(_download_and_hash, bucket, key)] = etag
        return True

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(max_in_flight, total)):
            if not submit(pool):
                break
        while in_flight:
            finished, _pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
            for future in finished:
                etag = in_flight.pop(future)
                on_result(etag, *future.result())
                done += 1
                if done % 1000 == 0 or done == total:
                    logger.info("hashed %d / %d downloads", done, total)
                submit(pool)


def hash_candidate_images(
    bucket: str,
    images: Mapping[int, Sequence[ImageRef]],
    out_dir: Path,
    *,
    workers: int,
    checkpoint_every: int,
) -> dict[tuple[int, str], int | None]:
    """Hash every candidate image, reusing checkpointed keys and identical ETags.

    Download failures are returned as unreadable for this run and are not checkpointed, so a later
    run retries them. Decode failures are checkpointed.
    """
    # JPEG draft() scales the decode by at most 8, so a large photo can still trip
    # Pillow's decompression-bomb cap. Raise it once, before worker threads start.
    Image.MAX_IMAGE_PIXELS = None
    existing = load_hash_rows(out_dir)
    done = _done_hashes(existing)
    known_etags = _etag_cache(existing)
    checkpointer = _Checkpointer(out_dir, checkpoint_every)
    results: dict[tuple[int, str], int | None] = {}
    pending: dict[str, list[ImageRef]] = defaultdict(list)
    n_images = 0
    n_checkpoint = 0
    n_etag_hits = 0

    for refs in images.values():
        for ref in refs:
            n_images += 1
            identity = (ref.source_id, ref.key)
            if identity in done:
                results[identity] = done[identity]
                n_checkpoint += 1
                continue
            if ref.etag in known_etags:
                image_hash = known_etags[ref.etag]
                results[identity] = image_hash
                checkpointer.add(
                    {
                        "source_id": ref.source_id,
                        "key": ref.key,
                        "etag": ref.etag,
                        "dhash": _encode_hash(image_hash),
                    }
                )
                n_etag_hits += 1
                continue
            pending[ref.etag].append(ref)

    jobs = [(etag, members[0].key) for etag, members in pending.items()]
    logger.info(
        "%d images: %d checkpoint hits, %d etag cache hits, %d unique downloads",
        n_images,
        n_checkpoint,
        n_etag_hits,
        len(jobs),
    )
    checkpointer.flush()
    transient = 0

    def on_result(etag: str, definitive: bool, image_hash: int | None) -> None:
        nonlocal transient
        if definitive and image_hash is not None:
            known_etags[etag] = image_hash
        for ref in pending[etag]:
            results[(ref.source_id, ref.key)] = image_hash
            if definitive:
                checkpointer.add(
                    {
                        "source_id": ref.source_id,
                        "key": ref.key,
                        "etag": ref.etag,
                        "dhash": _encode_hash(image_hash),
                    }
                )
            else:
                transient += 1

    try:
        _run_downloads(bucket, jobs, workers, on_result)
    finally:
        checkpointer.flush()
        consolidate_hashes(out_dir)
    if transient:
        logger.warning(
            "%d images failed to download and were treated as unreadable; re-run to retry them",
            transient,
        )
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", default=os.getenv("MERMAID_CORALNET_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--prefix", default=os.getenv("MERMAID_CORALNET_PREFIX", DEFAULT_PREFIX))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Write a hash checkpoint after this many newly hashed images.",
    )
    parser.add_argument(
        "--limit-sources",
        type=int,
        default=None,
        help=(
            "Compare only the lowest N source ids (smoke test). "
            "Same-count peers outside this set are ignored."
        ),
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")
    if args.limit_sources is not None and args.limit_sources < 0:
        parser.error("--limit-sources must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = apply_source_limit(
        list_source_images(make_s3_client(), args.bucket, args.prefix),
        args.limit_sources,
    )
    counts = {source_id: len(refs) for source_id, refs in images.items()}
    groups = group_by_count(counts)
    candidate_images: dict[int, list[ImageRef]] = {}
    n_groups = 0
    for n_images, source_ids in groups.items():
        if n_images == 0 or len(source_ids) < 2:
            continue
        n_groups += 1
        for source_id in source_ids:
            candidate_images[source_id] = images[source_id]
    logger.info(
        "%d sources to compare in %d same-count groups (%d images); %d sources skipped",
        len(candidate_images),
        n_groups,
        sum(len(refs) for refs in candidate_images.values()),
        len(counts) - len(candidate_images),
    )
    hashed = hash_candidate_images(
        args.bucket,
        candidate_images,
        args.output_dir,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )
    hashes_by_source: dict[int, list[int | None]] = {
        source_id: [None] * n_images
        for source_id, n_images in counts.items()
        if source_id not in candidate_images
    }
    for source_id, refs in candidate_images.items():
        hashes_by_source[source_id] = [hashed[(source_id, ref.key)] for ref in refs]
    report = build_report(hashes_by_source)
    write_report(args.output_dir, report)
    print(report.summary, end="")
    logger.info("wrote report to %s", args.output_dir)


if __name__ == "__main__":
    main()
