"""
title: mermaidseg.datasets.dataset
abstract: Module that contains dataset classes & functionality.
author: Viktor Domazetoski
date: 30-11-2025

Classes:
    BaseCoralDataset - A base PyTorch Dataset for loading annotated coral reef images.
    MermaidDataset - A PyTorch Dataset for loading annotated coral reef images from MERMAID sources.
    CoralNetDataset - A PyTorch Dataset for loading annotated coral reef images from CoralNet sources.
"""

import io
from ast import Raise
from typing import Any, List, Optional, Tuple, Union

import albumentations as A
import boto3
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import tqdm
from mermaidseg.datasets.concepts import (
    initialize_benthic_concepts,
    initialize_benthic_hierarchy,
    labels_to_concepts,
)
from mermaidseg.datasets.utils import create_annotation_mask, get_image_s3
from numpy.typing import NDArray
from torch.utils.data import Dataset


class BaseCoralDataset(Dataset[Tuple[Union[torch.Tensor, NDArray[Any]], Any]]):
    """
    A base PyTorch Dataset for loading annotated coral reef images.
    This dataset reads image annotations from an annotations file, retrieves images from S3, and applies optional transformations.
    The dataset is designed to be extended for specific data sources by implementing the read_image method and initializing the df_images and df_annotation arguments.
    Attributes:
        df_annotations (pd.DataFrame): DataFrame with all annotation data.
        df_images (pd.DataFrame): DataFrame with unique image entries.
        split (Optional[str]): Optional dataset split identifier (e.g., 'train', 'val', 'test').
        transform (Optional): Optional transformation function to apply to images.
        padding (Optional[int]): Padding value for point annotations of the segmentation mask.
    Args:
        annotations_path (str, optional): S3 path to the Parquet file with annotations. Defaults to a preset path.
        split (Optional[str], optional): Dataset split identifier. Defaults to None.
        transform (Optional, optional): Transformation function for images. Defaults to None.
    Methods:
        __len__(): Returns the number of annotation entries.
        __getitem__(idx): Retrieves the image at the given index, applies transformations, and returns it with a placeholder target.
        read_image(image_id): Abstract method to read an image given its ID. Must be implemented in subclasses.
        collate_fn(batch): Custom collate function to handle batching of images, masks, and annotations.
    """

    df_annotations: pd.DataFrame
    df_images: pd.DataFrame
    split: Optional[str]
    transform: Optional[A.BasicTransform]
    padding: Optional[int]
    class_subset: Optional[List[str]]
    num_classes: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    concept_mapping_flag: bool = False
    num_concepts: Optional[int] = None
    id2concept: Optional[dict[int, str]] = None
    concept2id: Optional[dict[str, int]] = None
    benthic_concept_matrix: Optional[np.ndarray] = None
    conceptid2labelid: Optional[dict[int, int]] = None

    def __init__(
        self,
        df_annotations: pd.DataFrame,
        df_images: pd.DataFrame,
        split: Optional[str] = None,
        transform: Optional[A.BasicTransform] = None,
        padding: Optional[int] = None,
        class_subset: Optional[List[str]] = None,
        concept_mapping_flag: bool = False,
    ):
        self.df_annotations = df_annotations
        self.df_images = df_images
        self.split = split
        self.transform = transform
        self.padding = padding
        self.class_subset = class_subset
        self.concept_mapping_flag = concept_mapping_flag

        if self.class_subset is not None:
            self.df_annotations = self.df_annotations[
                self.df_annotations["benthic_attribute_name"].apply(
                    lambda x: x in self.class_subset
                )
            ]

            if "region_id" in self.df_annotations.columns:  # Added for MermaidDataset
                self.df_images = (
                    self.df_annotations[["image_id", "region_id", "region_name"]]
                    .drop_duplicates(subset=["image_id"])
                    .reset_index(drop=True)
                )
            elif (
                "source_id" in self.df_annotations.columns
            ):  # Added for CoralNetDataset
                self.df_images = (
                    self.df_annotations[
                        ["source_id", "image_id"]  # Can add new columns here if needed
                    ]
                    .drop_duplicates(subset=["source_id", "image_id"])
                    .reset_index(drop=True)
                )
            else:
                raise ValueError(
                    "Unknown dataset structure for filtering images based on class_subset."
                )

            self.num_classes = len(self.class_subset) + 1  # +1 for background
            self.id2label = {
                i: attribute for i, attribute in enumerate(self.class_subset, start=1)
            }
            self.label2id = {v: k for k, v in self.id2label.items()}

        else:
            self.num_classes = (
                self.df_annotations["benthic_attribute_name"].nunique() + 1
            )  # +1 for background

            self.id2label = {
                i: attribute
                for i, attribute in enumerate(
                    self.df_annotations["benthic_attribute_name"]
                    .value_counts()
                    .index.tolist(),
                    start=1,
                )
            }
            self.label2id = {v: k for k, v in self.id2label.items()}

        if concept_mapping_flag:
            self.initialize_concept_mapping()

    def __len__(self):
        return self.df_images.shape[0]

    def read_image(self, **row_kwargs) -> NDArray[Any]:
        """
        Read an image given its ID. Needs to be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, NDArray[Any]], Any]:
        image_id = self.df_images.loc[idx, "image_id"]
        row_kwargs = self.df_images.loc[idx].to_dict()
        try:
            image = self.read_image(**row_kwargs)
        except Exception:
            return None, None, None
        # column_name = "image_id" # This needs to be point_id for
        annotations = self.df_annotations.loc[
            self.df_annotations["image_id"] == image_id,
            [
                # "point_id",
                "row",
                "col",
                # "benthic_attribute_id",
                "benthic_attribute_name",
                # "growth_form_id",
                # "growth_form_name",
            ],
        ]
        ## TODO: Check if we need to scale annotations based on image size (if using thumbnails - old code)

        mask = create_annotation_mask(
            annotations, image.shape, self.label2id, padding=self.padding
        )

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].transpose(2, 0, 1)
            mask = transformed["mask"]

        return image, mask, annotations

    def collate_fn(self, batch):
        """
        Collate function for MermaidDataset and CoralNetDataset.
        Args:
            batch: List of tuples (image, mask, annotations)
        Returns:
            images: Tensor or ndarray batch of images
            masks: Tensor or ndarray batch of masks
            annotations: List of annotation DataFrames
        """
        # images, masks, annotations = zip(*batch)

        # Filter out entries where image or mask is None
        filtered = [
            (img, msk, ann)
            for img, msk, ann in batch
            if img is not None and msk is not None
        ]
        images, masks, annotations = zip(*filtered)

        # Handle empty batch
        if len(images) == 0:
            return torch.tensor([]), torch.tensor([]), []

        # Convert to tensors if they aren't already
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
            masks = torch.stack(masks)
        else:
            # Convert numpy arrays to tensors for consistency
            images = torch.stack(
                [
                    torch.from_numpy(img) if isinstance(img, np.ndarray) else img
                    for img in images
                ]
            )
            masks = torch.stack(
                [
                    torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
                    for mask in masks
                ]
            )

        return images, masks

    def initialize_concept_mapping(self):
        """
        Initialize concept mapping attributes for the dataset.
        Sets up the mapping between benthic attributes and concepts based on a predefined hierarchy.
        """
        hierarchy_dict = initialize_benthic_hierarchy()
        class_set = list(self.id2label.values())
        benthic_concept_set, benthic_concept_matrix = initialize_benthic_concepts(
            class_set, hierarchy_dict
        )
        self.num_concepts = len(benthic_concept_set)
        self.id2concept = {
            i: attribute for i, attribute in enumerate(benthic_concept_set, start=1)
        }
        self.concept2id = {v: k for k, v in self.id2concept.items()}
        self.benthic_concept_matrix = benthic_concept_matrix
        self.conceptid2labelid = {}
        for ind, label in self.id2label.items():
            col_ind = int(
                np.where(
                    self.benthic_concept_matrix.columns.get_level_values("concept")
                    == label
                )[0][0]
            )
            self.conceptid2labelid[col_ind] = ind


class MermaidDataset(BaseCoralDataset):
    """
    A PyTorch Dataset for loading MERMAID annotated coral reef images from a Parquet file stored on S3.
    This dataset reads image annotations from a Parquet file, retrieves images from S3, and applies optional transformations.
    Each item returned is a tuple containing the image (as a tensor or ndarray) and a placeholder for the target (currently None).
    Attributes:
        annotations_path (str): Path to the Parquet file containing image annotations.
        source_bucket (str): S3 bucket name containing the dataset files.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        annotations_path (str, optional): S3 path to the Parquet file with annotations. Defaults to a preset path.
        source_bucket (str, optional): S3 bucket name containing the dataset files. Defaults to "coral-reef-training".
    """

    annotations_path: str
    source_bucket: str
    s3: boto3.client

    def __init__(
        self,
        annotations_path: str = "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet",
        source_bucket: str = "coral-reef-training",
        **base_kwargs,
    ):
        self.annotations_path = annotations_path
        self.source_bucket = source_bucket
        self.s3 = boto3.client("s3")

        self.df_annotations, self.df_images = self.load_annotations(
            self.annotations_path
        )

        super().__init__(
            df_annotations=self.df_annotations, df_images=self.df_images, **base_kwargs
        )

    def load_annotations(
        self, annotations_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load annotations from a Parquet file on S3.
        """
        df_annotations = pd.read_parquet(annotations_path)
        df_images = (
            df_annotations[["image_id", "region_id", "region_name"]]
            .drop_duplicates(subset=["image_id"])
            .reset_index(drop=True)
        )
        return df_annotations, df_images

    def read_image(self, image_id: str, **row_kwargs) -> NDArray[Any]:
        """
        Read an image given its ID. Needs to be implemented in subclasses.
        """
        key = f"mermaid/{image_id}.png"  # f"mermaid/{image_id}_thumbnail.png"
        image = np.array(
            get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB")
        )
        return image


class CoralNetDataset(BaseCoralDataset):
    """
    A PyTorch Dataset for loading annotated coral reef images from a CoralNet sources.
    Each item returned is a tuple containing the image (as a tensor or ndarray) and a placeholder for the target (currently None).
    Attributes:
    Args:
    Methods:
    """

    source_ids: List[Union[int, str]]
    source_bucket: str
    source_s3_prefix: str
    s3: boto3.client
    whitelist_sources: Optional[List[Union[int, str]]]
    blacklist_sources: Optional[List[Union[int, str]]]

    def __init__(
        self,
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "coralnet-public-images",
        whitelist_sources: Optional[List[Union[int, str]]] = None,
        blacklist_sources: Optional[List[Union[int, str]]] = None,
        **base_kwargs,
    ):
        self.source_bucket = source_bucket
        self.source_s3_prefix = source_s3_prefix
        self.s3 = boto3.client("s3")
        self.whitelist_sources = whitelist_sources
        self.blacklist_sources = blacklist_sources
        if self.whitelist_sources is not None and self.blacklist_sources is not None:
            raise ValueError("Cannot specify both whitelist and blacklist sources.")

        self.labelmapping = self.initialize_coralnet_mapping()
        self.df_annotations, self.df_images = self.load_annotations()

        if self.whitelist_sources is not None:
            self.df_annotations = self.df_annotations[
                self.df_annotations["source_id"].apply(
                    lambda x: x in self.whitelist_sources
                )
            ]
        if self.blacklist_sources is not None:
            self.df_annotations = self.df_annotations[
                self.df_annotations["source_id"].apply(
                    lambda x: x not in self.blacklist_sources
                )
            ]
        if self.whitelist_sources is not None or self.blacklist_sources is not None:
            self.df_images = (
                self.df_annotations[
                    ["source_id", "image_id"]  # Can add new columns here if needed
                ]
                .drop_duplicates(subset=["source_id", "image_id"])
                .reset_index(drop=True)
            )
        super().__init__(
            df_annotations=self.df_annotations, df_images=self.df_images, **base_kwargs
        )

    def load_annotations(self):
        """
        Load annotations from a Parquet file on S3.
        """
        annotations_path = (
            f"s3://{self.source_bucket}/coralnet_annotations_30112025.parquet"
        )
        self.df_annotations = pd.read_parquet(annotations_path)

        self.df_annotations["benthic_attribute_name"] = self.df_annotations[
            "coralnet_id"
        ].apply(lambda x: self.labelmapping.get(str(x), None))

        self.df_annotations = self.df_annotations[
            ["source_id", "image_id", "row", "col", "benthic_attribute_name"]
        ]  # Can add new columns here if needed, keeping only most important for the start

        self.df_images = (
            self.df_annotations[
                ["source_id", "image_id"]  # Can add new columns here if needed
            ]
            .drop_duplicates(subset=["source_id", "image_id"])
            .reset_index(drop=True)
        )
        return self.df_annotations, self.df_images

    def read_image(self, image_id: str, source_id: str, **row_kwargs) -> NDArray[Any]:
        key = f"{self.source_s3_prefix}/s{source_id}/images/{image_id}.jpg"
        image = np.array(
            get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB")
        )
        return image

    def initialize_coralnet_mapping(
        self,
        mapping_endpoint="https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet",
    ):
        """
        Initialize CoralNet to MERMAID label mapping from provider API.
        """
        response = requests.get(mapping_endpoint)
        data = response.json()
        labelset = data["results"]

        while data["next"]:
            response = requests.get(data["next"])
            data = response.json()
            labelset.extend(data["results"])
        label_mapping = {
            label["provider_id"]: label["benthic_attribute_name"] for label in labelset
        }
        return label_mapping
