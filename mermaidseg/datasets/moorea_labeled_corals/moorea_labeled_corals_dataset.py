"""Moorea Labeled Corals PyTorch dataset.

Reads Moorea Labeled Corals point annotations from a Parquet file on S3 and
emits ``(image, source_labels)`` tuples where ``source_labels`` is in the
**Moorea Labeled Corals label-name space** -- i.e. the canonical (case-merged)
label strings produced by
``mermaidseg/datasets/moorea_labeled_corals/verify_raw_data_and_add_to_s3.py``
from the per-image ``<stem>.<ext>.txt`` ``row;col;label`` files of the EDI
``knb-lter-mcr.5006`` package (Beijbom et al., 2012). Mapping into the MERMAID
benthic-attribute target space is performed externally via
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


class MooreaLabeledCoralsDataset(BaseCoralDataset):
    """A PyTorch Dataset for loading Moorea Labeled Corals annotated coral reef images
    from a Parquet file stored on S3.

    Each item returned is a tuple ``(image, source_labels)`` where
    ``source_labels`` is an integer mask in the Moorea Labeled Corals source
    label space (or in the joint global space, if the dataset has been
    registered with a :class:`SourceLabelRegistry`).

    Attributes:
        annotations_path (str): Key (relative to ``source_bucket``) of the
            Parquet file with annotations. Produced by
            ``mermaidseg/datasets/moorea_labeled_corals/verify_raw_data_and_add_to_s3.py``.
        source_bucket (str): S3 bucket name containing the dataset files.
        source_s3_prefix (str): S3 prefix under which the per-year
            ``images/<year>/<image_id><image_ext>`` folders live.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        annotations_path (str, optional): Key (relative to ``source_bucket``)
            of the Parquet file with annotations.
        source_bucket (str, optional): S3 bucket name containing the dataset files.
        source_s3_prefix (str, optional): S3 prefix containing the
            ``images/<year>/<image_id><image_ext>`` tree.
        whitelist_years / blacklist_years: Optional Moorea survey year
            allowlist/denylist (e.g. ``["2008", "2009"]``).
        **base_kwargs: Forwarded to :class:`BaseCoralDataset`.
    """

    SOURCE_NAME = "moorea_labeled_corals"

    annotations_path: str
    source_bucket: str
    source_s3_prefix: str
    s3: boto3.client
    whitelist_years: list[str] | None
    blacklist_years: list[str] | None

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "year",
        "image_id",
        "image_ext",
        "row",
        "col",
        "source_label_name",
    )

    def __init__(
        self,
        annotations_path: str = (
            "external_validation_datasets/moorea_labeled_corals/"
            "moorea_labeled_corals_annotations.parquet"
        ),
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "external_validation_datasets/moorea_labeled_corals",
        whitelist_years: list[str] | None = None,
        blacklist_years: list[str] | None = None,
        **base_kwargs: Any,
    ):
        if whitelist_years is not None and blacklist_years is not None:
            raise ValueError("Cannot specify both whitelist_years and blacklist_years.")

        self.annotations_path = annotations_path
        self.source_bucket = source_bucket
        self.source_s3_prefix = source_s3_prefix.rstrip("/")
        self.s3 = boto3.client("s3")
        self.whitelist_years = (
            [str(y) for y in whitelist_years] if whitelist_years is not None else None
        )
        self.blacklist_years = (
            [str(y) for y in blacklist_years] if blacklist_years is not None else None
        )

        df_annotations, df_images = self.load_annotations()
        super().__init__(df_annotations=df_annotations, df_images=df_images, **base_kwargs)

    def load_annotations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load Moorea Labeled Corals annotations from the Parquet file on S3.

        The dataset emits labels in the Moorea Labeled Corals source-name
        space, so each row's ``source_label_name`` already holds the canonical
        per-source label string. No Moorea -> MERMAID translation happens
        here; that mapping is owned by
        :mod:`mermaidseg.dataset_reconciliation.label_mapping` and is applied
        at training time on the GPU via a long-tensor lookup.
        """
        annotations_uri = f"s3://{self.source_bucket}/{self.annotations_path}"
        df_annotations = pd.read_parquet(annotations_uri)

        missing = set(self.REQUIRED_COLUMNS) - set(df_annotations.columns)
        if missing:
            raise ValueError(
                f"Moorea Labeled Corals annotations parquet at {annotations_uri} is missing "
                f"required columns: {sorted(missing)}"
            )

        df_annotations = df_annotations[list(self.REQUIRED_COLUMNS)].copy()
        df_annotations["year"] = df_annotations["year"].astype(str)
        df_annotations["image_id"] = df_annotations["image_id"].astype(str)
        df_annotations["image_ext"] = df_annotations["image_ext"].astype(str)
        df_annotations["source_label_name"] = df_annotations["source_label_name"].astype(str)

        if self.whitelist_years is not None:
            df_annotations = df_annotations[
                df_annotations["year"].isin(self.whitelist_years)
            ].reset_index(drop=True)
        if self.blacklist_years is not None:
            df_annotations = df_annotations[
                ~df_annotations["year"].isin(self.blacklist_years)
            ].reset_index(drop=True)

        df_images = self._derive_df_images_from_annotations(df_annotations)
        return df_annotations, df_images

    def _derive_df_images_from_annotations(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        return (
            df_annotations[["image_id", "year", "image_ext"]]
            .drop_duplicates(subset=["image_id", "year"])
            .reset_index(drop=True)
        )

    def read_image(
        self,
        image_id: str,
        year: str,
        image_ext: str,
        **row_kwargs: Any,
    ) -> NDArray[Any]:
        key = f"{self.source_s3_prefix}/images/{year}/{image_id}{image_ext}"
        return np.array(get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB"))
