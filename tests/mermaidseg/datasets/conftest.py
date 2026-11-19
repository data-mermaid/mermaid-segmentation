"""Fixtures for the dataset-class tests: a local-cache stand-in for the S3 bucket."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mermaidseg.datasets.local_cache import LocalS3Cache


class FakeBucketStore:
    """Writes objects where an offline ``LocalS3Cache`` reads them, so no S3 call is possible."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, bucket: str, key: str) -> Path:
        path = self.root / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_parquet(self, bucket: str, key: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self._path(bucket, key), index=False)

    def put_json(self, bucket: str, key: str, payload: Any) -> None:
        self._path(bucket, key).write_text(json.dumps(payload))

    def put_png(self, bucket: str, key: str, image: Image.Image) -> None:
        image.save(self._path(bucket, key), format="PNG")

    def put_solid(self, bucket: str, key: str, size: tuple[int, int], color: tuple[int, ...]):
        """Store a solid RGB image of ``size`` = (width, height)."""
        self.put_png(bucket, key, Image.new("RGB", size, color))

    def put_label(self, bucket: str, key: str, labels: np.ndarray) -> None:
        """Store a uint8 label array as an 8-bit grayscale PNG."""
        self.put_png(bucket, key, Image.fromarray(labels.astype(np.uint8), mode="L"))


@pytest.fixture
def bucket_store(tmp_path: Path) -> Iterator[FakeBucketStore]:
    """An offline local cache rooted in ``tmp_path``; any miss raises instead of using S3."""
    root = tmp_path / "cache"
    LocalS3Cache.reset_singleton()
    LocalS3Cache.configure(root, offline=True)
    yield FakeBucketStore(root)
    LocalS3Cache.reset_singleton()
