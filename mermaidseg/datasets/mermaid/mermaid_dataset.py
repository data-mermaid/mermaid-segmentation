"""MERMAID PyTorch dataset.

Reads MERMAID confirmed annotations from a parquet file on S3 and emits
``(image, source_labels)`` tuples where ``source_labels`` is in the MERMAID
benthic attribute label space — the canonical target space for this project.
"""

from __future__ import annotations

from typing import Any

import boto3
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.utils import get_image_s3


class MermaidDataset(BaseCoralDataset):
    """A PyTorch Dataset for loading MERMAID annotated coral reef images from a Parquet file stored
    on S3.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is an integer mask in the MERMAID benthic-attribute label
    space (or in the joint global space, if the dataset has been registered
    with a :class:`SourceLabelRegistry`).

    Attributes:
        annotations_path (str): Path to the Parquet file containing image annotations.
        source_bucket (str): S3 bucket name containing the dataset files.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        annotations_path (str, optional): S3 path to the Parquet file with annotations.
        source_bucket (str, optional): S3 bucket name containing the dataset files.
        **base_kwargs: Forwarded to :class:`BaseCoralDataset`.
    """

    SOURCE_NAME = "mermaid"

    annotations_path: str
    source_bucket: str
    s3: boto3.client

    def __init__(
        self,
        annotations_path: str = "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet",
        source_bucket: str = "coral-reef-training",
        **base_kwargs: Any,
    ):
        self.annotations_path = annotations_path
        self.source_bucket = source_bucket
        self.s3 = boto3.client("s3")

        df_annotations, df_images = self.load_annotations(self.annotations_path)
        super().__init__(df_annotations=df_annotations, df_images=df_images, **base_kwargs)

    def load_annotations(self, annotations_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load annotations from a Parquet file on S3.

        For MERMAID, the source-space label name *is* the canonical
        ``benthic_attribute_name`` field. This helper renames the column to the
        unified ``source_label_name`` convention used by
        :class:`BaseCoralDataset`.
        """
        df_annotations = pd.read_parquet(annotations_path)
        df_annotations = df_annotations.rename(
            columns={"benthic_attribute_name": "source_label_name"}
        )
        df_images = self._derive_df_images_from_annotations(df_annotations)
        return df_annotations, df_images

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        return (
            df_annotations[["image_id", "region_id", "region_name"]]
            .drop_duplicates(subset=["image_id"])
            .reset_index(drop=True)
        )

    def read_image(self, image_id: str, **row_kwargs: Any) -> NDArray[Any]:
        key = f"mermaid/{image_id}.png"
        return np.array(get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB"))
