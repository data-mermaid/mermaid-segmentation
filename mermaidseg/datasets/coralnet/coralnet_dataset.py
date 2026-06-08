"""CoralNet PyTorch dataset.

Reads CoralNet point annotations from a Parquet file on S3 and emits
``(image, source_labels)`` tuples where ``source_labels`` is in the
**CoralNet provider label space** — i.e. the integer IDs used by CoralNet's
own label catalogue. Mapping into the MERMAID benthic attribute target space
is performed externally via
:mod:`mermaidseg.dataset_reconciliation.label_mapping`.
"""

from __future__ import annotations

from typing import Any

import boto3
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.utils import get_image_s3

import json 
import fsspec
class CoralNetDataset(BaseCoralDataset):
    """A PyTorch Dataset for loading CoralNet annotated coral reef images from a Parquet file stored
    on S3.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is an integer mask in CoralNet's own provider-ID label
    space (or in the joint global space, if the dataset has been registered
    with a :class:`SourceLabelRegistry`).

    Attributes:
        annotations_path (str): Path to the Parquet file containing image annotations.
            This is created by merging all annotations of CoralNet sources in a separate
            notebook ``mermaidseg/datasets/coralnet/nbs/CoralNet_Annotations.ipynb``.
        source_bucket (str): S3 bucket name containing the dataset files.
        source_s3_prefix (str): S3 prefix under which the per-source CoralNet image folders live.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        annotations_path (str, optional): Key (relative to ``source_bucket``) of the parquet file with annotations.
        source_bucket (str, optional): S3 bucket name containing the dataset files.
        source_s3_prefix (str, optional): S3 prefix containing per-source CoralNet image folders.
        whitelist_sources / blacklist_sources: Optional CoralNet source-id allowlist/denylist.
        **base_kwargs: Forwarded to :class:`BaseCoralDataset`.
    """

    SOURCE_NAME = "coralnet"

    annotations_path: str
    source_ids: list[int | str]
    source_bucket: str
    source_s3_prefix: str
    s3: boto3.client
    whitelist_sources: list[int | str] | None
    blacklist_sources: list[int | str] | None

    def __init__(
        self,
        annotations_path: str = "etl-outputs/coralnet/20260526_807b611/coralnet_training_resized_20260526_807b611.parquet",
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "coralnet-public-images",
        whitelist_sources: list[int | str] | None = None,
        blacklist_sources: list[int | str] | None = None,
        **base_kwargs: Any,
    ):
        self.annotations_path = annotations_path
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
        """Load CoralNet annotations from a parquet file on S3.

        The dataset emits labels in CoralNet's own provider label space, so
        each row's ``coralnet_id`` (cast to ``str``) becomes the
        ``source_label_name``. No CoralNet -> MERMAID translation happens
        here; that mapping is owned by
        :mod:`mermaidseg.dataset_reconciliation.label_mapping` and is applied
        at training time on the GPU via a long-tensor lookup.
        """
        annotations_path = f"s3://{self.source_bucket}/{self.annotations_path}"
        df_annotations = pd.read_parquet(annotations_path)
        id2name_path = "s3://dev-datamermaid-sm-sources/coralnet-public-images/temporary/coralnet_id2name.json"

        with fsspec.open(id2name_path, "r") as f:
            coralnet_id2name = json.load(f)

        df_annotations["coralnet_name"] = df_annotations["coralnet_id"].map(lambda x: coralnet_id2name.get(str(x)))
        df_annotations["source_label_name"] = df_annotations["coralnet_name"].astype(str).str.lower()

        df_annotations = df_annotations[
            [
                "source_id",
                "image_id",
                "row",
                "col",
                "coralnet_id",
                "coralnet_name",
                "source_label_name",
            ]
        ]

        df_images = self._derive_df_images_from_annotations(df_annotations)
        return df_annotations, df_images

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        return (
            df_annotations[["source_id", "image_id"]]
            .drop_duplicates(subset=["source_id", "image_id"])
            .reset_index(drop=True)
        )

    def read_image(self, image_id: str, source_id: str, **row_kwargs: Any) -> NDArray[Any]:
        key = f"{self.source_s3_prefix}/s{source_id}/images/{image_id}.jpg"
        return np.array(get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB"))
