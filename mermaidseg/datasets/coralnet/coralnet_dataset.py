"""CoralNet PyTorch dataset.

Reads CoralNet point annotations from a self-contained training manifest
Parquet file on S3 and emits ``(image, source_labels)`` tuples where
``source_labels`` is in the **CoralNet provider label space** — i.e. the
integer IDs used by CoralNet's own label catalogue. Mapping into the MERMAID
benthic attribute target space is performed externally via
:mod:`mermaidseg.dataset_reconciliation.label_mapping`.

The training manifest (``coralnet_training_manifest_<run>.parquet``) carries
per-annotation rows with ``(row, col, coralnet_id)`` in the **resized**
coordinate space (longest edge clamped to 2048; see
``mermaidseg.datasets.coralnet.preprocessing.resize``) plus the resolved
``image_s3_key`` for each image (resized copy when the image was resized,
otherwise the original). Pin a specific build via
``MERMAID_CORALNET_MANIFEST_PATH`` or ``MERMAID_CORALNET_MANIFEST_VERSION``.
Images with no ``image_s3_key`` are skipped (never loaded at the wrong
resolution); the dataset recovers by serving a different image.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.local_cache import LocalS3Cache
from mermaidseg.datasets.utils import get_image_s3

logger = logging.getLogger(__name__)

_DEFAULT_RUN = "20260623_nogit"
_DEFAULT_MANIFEST_PATH = (
    f"etl-outputs/coralnet/{_DEFAULT_RUN}/coralnet_training_manifest_{_DEFAULT_RUN}.parquet"
)
_CORALNET_ID2NAME_KEY = "coralnet-public-images/temporary/coralnet_id2name.json"


def _resolve_default_manifest_path() -> str:
    """Resolve the default training manifest key from env vars.

    Precedence:
        1. ``MERMAID_CORALNET_MANIFEST_PATH`` — full S3 key.
        2. ``MERMAID_CORALNET_MANIFEST_VERSION`` — version tag; builds
           ``etl-outputs/coralnet/<version>/coralnet_training_manifest_<version>.parquet``.
        3. :data:`_DEFAULT_MANIFEST_PATH`.
    """
    explicit = os.getenv("MERMAID_CORALNET_MANIFEST_PATH")
    if explicit:
        return explicit
    version = os.getenv("MERMAID_CORALNET_MANIFEST_VERSION")
    if version:
        return f"etl-outputs/coralnet/{version}/coralnet_training_manifest_{version}.parquet"
    return _DEFAULT_MANIFEST_PATH


class CoralNetDataset(BaseCoralDataset):
    """A PyTorch Dataset for loading CoralNet annotated coral reef images from a Parquet file stored
    on S3.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is an integer mask in CoralNet's own provider-ID label
    space (or in the joint global space, if the dataset has been registered
    with a :class:`SourceLabelRegistry`).

    Attributes:
        manifest_path (str): Key (relative to ``source_bucket``) of the self-contained
            training manifest parquet.
        source_bucket (str): S3 bucket name containing the dataset files.
        source_s3_prefix (str): S3 prefix under which the per-source CoralNet image folders live.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        manifest_path (str, optional): Key (relative to ``source_bucket``) of the training
            manifest parquet. If ``None`` (default), resolved from
            ``MERMAID_CORALNET_MANIFEST_PATH`` or ``MERMAID_CORALNET_MANIFEST_VERSION``;
            falls back to the default published manifest.
        source_bucket (str, optional): S3 bucket name containing the dataset files.
        source_s3_prefix (str, optional): S3 prefix containing per-source CoralNet image folders.
        whitelist_sources / blacklist_sources: Optional CoralNet source-id allowlist/denylist.
        **base_kwargs: Forwarded to :class:`BaseCoralDataset`.
    """

    SOURCE_NAME = "coralnet"

    manifest_path: str
    source_ids: list[int | str]
    source_bucket: str
    source_s3_prefix: str
    s3: boto3.client
    whitelist_sources: list[int | str] | None
    blacklist_sources: list[int | str] | None

    def __init__(
        self,
        manifest_path: str | None = None,
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "coralnet-public-images",
        whitelist_sources: list[int | str] | None = None,
        blacklist_sources: list[int | str] | None = None,
        **base_kwargs: Any,
    ):
        self.manifest_path = manifest_path or _resolve_default_manifest_path()
        self.source_bucket = source_bucket
        self.source_s3_prefix = source_s3_prefix
        self.s3 = boto3.client("s3")
        self.whitelist_sources = whitelist_sources
        self.blacklist_sources = blacklist_sources
        if self.whitelist_sources is not None and self.blacklist_sources is not None:
            raise ValueError("Cannot specify both whitelist and blacklist sources.")

        df_annotations, df_images = self.load_annotations()

        if self.whitelist_sources is not None:
            df_annotations = df_annotations[
                df_annotations["source_id"].apply(lambda x: x in self.whitelist_sources)
            ]
        if self.blacklist_sources is not None:
            df_annotations = df_annotations[
                df_annotations["source_id"].apply(lambda x: x not in self.blacklist_sources)
            ]
        if self.whitelist_sources is not None or self.blacklist_sources is not None:
            df_annotations = df_annotations.reset_index(drop=True)
            df_images = self._derive_df_images_from_annotations(df_annotations)

        super().__init__(df_annotations=df_annotations, df_images=df_images, **base_kwargs)

    def load_annotations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load CoralNet annotations from the training manifest parquet on S3.

        The manifest carries per-annotation rows plus the resolved ``image_s3_key``
        for each image. ``source_label_name`` is derived from ``coralnet_id`` via
        ``coralnet_id2name.json`` (the manifest's ``source_label_name`` column
        holds numeric IDs and is overwritten). No CoralNet -> MERMAID translation
        happens here; that mapping is owned by
        :mod:`mermaidseg.dataset_reconciliation.label_mapping` and is applied
        at training time on the GPU via a long-tensor lookup.
        """
        df = LocalS3Cache.get().read_parquet(self.source_bucket, self.manifest_path)
        coralnet_id2name = LocalS3Cache.get().read_json(
            self.source_bucket,
            _CORALNET_ID2NAME_KEY,
        )

        df["coralnet_name"] = df["coralnet_id"].map(lambda x: coralnet_id2name.get(str(x)))
        df["source_label_name"] = df["coralnet_name"].astype(str).str.lower()
        df = df.rename(columns={"image_s3_key": "image_key"})

        df_annotations = df[
            [
                "source_id",
                "image_id",
                "row",
                "col",
                "coralnet_id",
                "coralnet_name",
                "source_label_name",
                "image_key",
            ]
        ]

        df_images = self._derive_df_images_from_annotations(df_annotations)
        return df_annotations, df_images

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        return (
            df_annotations[["source_id", "image_id", "image_key"]]
            .drop_duplicates(subset=["source_id", "image_id"])
            .reset_index(drop=True)
        )

    def read_image(
        self,
        image_id: str,
        source_id: str,
        image_key: str | None = None,
        **row_kwargs: Any,
    ) -> NDArray[Any]:
        # ``image_key`` (resized vs original) comes from the training manifest via df_images.
        # When the manifest has no entry we deliberately do NOT fall back to the original
        # full-resolution image: its pixels would be misaligned with the 2048-resized annotation
        # coordinates. Raising here lets BaseCoralDataset warn and recover with a different image.
        if not (isinstance(image_key, str) and image_key):
            raise KeyError(
                f"No manifest entry for source_id={source_id} image_id={image_id}; "
                f"refusing to load the original full-resolution image (its pixels would be "
                f"misaligned with the 2048-resized annotation coordinates)."
            )
        return np.array(
            get_image_s3(s3=None, bucket=self.source_bucket, key=image_key).convert("RGB")
        )
