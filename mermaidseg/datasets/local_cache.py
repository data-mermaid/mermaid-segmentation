"""Local-first S3 object cache for dataset I/O.

Mirrors S3 key layout under a configurable root directory. On cache miss,
fetches from S3 with per-key file locking (to avoid duplicate paid GETs from
DataLoader workers), write-throughs to disk, and tracks local vs S3 hit counts.
"""

from __future__ import annotations

import fcntl
import json
import logging
import multiprocessing as mp
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import io

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

_ENV_CACHE_DIR = "MERMAIDSEG_LOCAL_CACHE_DIR"


class DataLoadError(Exception):
    """Raised when an object cannot be loaded from storage or decoded."""


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of cache hit/miss counters for one training phase."""

    local_hits: int
    s3_fetches: int


@dataclass
class CacheStatsHandles:
    """Shared multiprocessing counters for cross-worker aggregation."""

    local_hits: Any  # mp.Value
    s3_fetches: Any  # mp.Value

    @classmethod
    def create(cls) -> CacheStatsHandles:
        return cls(
            local_hits=mp.Value("Q", 0),
            s3_fetches=mp.Value("Q", 0),
        )


def create_cache_stats() -> CacheStatsHandles:
    """Create fresh shared counters for a training run."""
    return CacheStatsHandles.create()


def parse_storage_ref(ref: str, default_bucket: str) -> tuple[str, str] | None:
    """Parse a storage reference into ``(bucket, key)`` for the cache layer.

    Returns ``None`` when *ref* is an absolute local filesystem path that
    should be read directly (bypassing the cache).
    """
    if ref.startswith("s3://"):
        without_scheme = ref[5:]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {ref!r}")
        return bucket, key

    path = Path(ref)
    if path.is_absolute():
        return None

    return default_bucket, ref.lstrip("/")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class LocalS3Cache:
    """Singleton local-first cache over S3 objects."""

    _instance: LocalS3Cache | None = None

    def __init__(self) -> None:
        self._root: Path | None = None
        self._write_through = True
        self._enabled = False
        self._stats: CacheStatsHandles | None = None
        self._s3_client: Any | None = None

    @classmethod
    def get(cls) -> LocalS3Cache:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton (for tests)."""
        cls._instance = None

    @classmethod
    def configure(
        cls,
        root: str | Path | None,
        *,
        write_through: bool = True,
        stats: CacheStatsHandles | None = None,
    ) -> LocalS3Cache:
        cache = cls.get()
        if root is None or (isinstance(root, str) and root.strip() in ("", "null", "None")):
            cache._root = None
            cache._enabled = False
        else:
            cache._root = Path(root).expanduser().resolve()
            cache._enabled = True
        cache._write_through = write_through
        if stats is not None:
            cache._stats = stats
        return cache

    @classmethod
    def configure_from_env(cls) -> LocalS3Cache:
        root = os.environ.get(_ENV_CACHE_DIR)
        cache = cls.get()
        if root:
            cache._root = Path(root).expanduser().resolve()
            cache._enabled = True
        return cache

    @property
    def enabled(self) -> bool:
        return self._enabled

    def attach_stats(self, stats: CacheStatsHandles) -> None:
        self._stats = stats

    def local_path(self, bucket: str, key: str) -> Path:
        if self._root is None:
            raise RuntimeError("LocalS3Cache is not configured with a root directory.")
        return self._root / bucket / key

    def _record_local_hit(self) -> None:
        if self._stats is not None:
            with self._stats.local_hits.get_lock():
                self._stats.local_hits.value += 1

    def _record_s3_fetch(self) -> None:
        if self._stats is not None:
            with self._stats.s3_fetches.get_lock():
                self._stats.s3_fetches.value += 1

    def snapshot_stats(self) -> CacheStats:
        if self._stats is None:
            return CacheStats(local_hits=0, s3_fetches=0)
        with self._stats.local_hits.get_lock():
            local_hits = int(self._stats.local_hits.value)
            self._stats.local_hits.value = 0
        with self._stats.s3_fetches.get_lock():
            s3_fetches = int(self._stats.s3_fetches.value)
            self._stats.s3_fetches.value = 0
        return CacheStats(local_hits=local_hits, s3_fetches=s3_fetches)

    def _get_s3_client(self) -> Any:
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def set_s3_client(self, client: Any) -> None:
        """Inject an S3 client (for tests)."""
        self._s3_client = client

    def _fetch_from_s3(self, bucket: str, key: str) -> bytes:
        s3 = self._get_s3_client()
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.warning(
                "S3 error loading object (bucket=%s, key=%s): %s %s",
                bucket,
                key,
                error_code,
                e,
            )
            raise

    def read_bytes(self, bucket: str, key: str) -> bytes:
        """Read object bytes, preferring the local cache when configured."""
        if self._enabled:
            local = self.local_path(bucket, key)
            if local.is_file() and local.stat().st_size > 0:
                self._record_local_hit()
                return local.read_bytes()

            if self._write_through:
                lock_path = local.with_suffix(local.suffix + ".lock")
                with _file_lock(lock_path):
                    if local.is_file() and local.stat().st_size > 0:
                        self._record_local_hit()
                        return local.read_bytes()
                    data = self._fetch_from_s3(bucket, key)
                    _atomic_write(local, data)
                    self._record_s3_fetch()
                    return data

        data = self._fetch_from_s3(bucket, key)
        self._record_s3_fetch()
        return data

    def read_parquet(self, bucket: str, key: str) -> pd.DataFrame:
        """Read a Parquet file via the local-first cache."""
        if self._enabled:
            local = self.local_path(bucket, key)
            if local.is_file() and local.stat().st_size > 0:
                self._record_local_hit()
                return pd.read_parquet(local)
            data = self.read_bytes(bucket, key)
            return pd.read_parquet(io.BytesIO(data))
        return pd.read_parquet(f"s3://{bucket}/{key}")

    def read_parquet_ref(self, ref: str, default_bucket: str) -> pd.DataFrame:
        """Read a Parquet file from a storage reference (S3 URI, relative key, or local path)."""
        parsed = parse_storage_ref(ref, default_bucket)
        if parsed is None:
            return pd.read_parquet(ref)
        bucket, key = parsed
        return self.read_parquet(bucket, key)

    def read_pil_image(self, bucket: str, key: str, *, thumbnail: bool = False) -> Image.Image:
        """Read an image object and return a PIL Image."""
        if thumbnail:
            key = key.replace(".png", "_thumbnail.png")

        try:
            image_data = self.read_bytes(bucket, key)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            raise DataLoadError(f"S3 ClientError for s3://{bucket}/{key}: {error_code}") from e

        try:
            return Image.open(io.BytesIO(image_data))
        except (UnidentifiedImageError, OSError) as e:
            logger.warning("Corrupted image (bucket=%s, key=%s): %s", bucket, key, e)
            raise DataLoadError(f"PIL cannot open image at s3://{bucket}/{key}") from e

    def read_json(self, bucket: str, key: str) -> dict[str, Any]:
        """Read a JSON object via the local-first cache."""
        body = self.read_bytes(bucket, key)
        return json.loads(body)


def setup_local_cache(
    data_cfg: Any,
    stats: CacheStatsHandles | None = None,
) -> CacheStatsHandles:
    """Configure the cache from data config and optional env override."""
    if stats is None:
        stats = create_cache_stats()

    root = os.environ.get(_ENV_CACHE_DIR)
    if not root:
        root = getattr(data_cfg, "local_cache_dir", None)
        if root is None and isinstance(data_cfg, dict):
            root = data_cfg.get("local_cache_dir")

    write_through = getattr(data_cfg, "local_cache_write_through", True)
    if isinstance(data_cfg, dict):
        write_through = data_cfg.get("local_cache_write_through", write_through)

    LocalS3Cache.configure(root, write_through=write_through, stats=stats)
    if root:
        resolved = str(Path(str(root)).expanduser().resolve())
        os.environ[_ENV_CACHE_DIR] = resolved
        logger.info("Local S3 cache enabled at %s (write_through=%s)", resolved, write_through)
    return stats
