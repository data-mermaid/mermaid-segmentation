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

from typing import Any

import albumentations as A
import boto3
import numpy as np
import pandas as pd
import requests
import torch
from datasets import concatenate_datasets, load_dataset
from numpy.typing import NDArray
from torch.utils.data import Dataset

from mermaidseg.datasets.concepts import (
    initialize_benthic_concepts,
    initialize_benthic_hierarchy,
)
from mermaidseg.datasets.utils import create_annotation_mask, get_image_s3


class BaseCoralDataset(Dataset[tuple[torch.Tensor | NDArray[Any], Any]]):
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
        class_subset (Optional[List[str]]): Optional list of benthic attribute names to filter the dataset.
        num_classes (int): Number of unique benthic attribute classes in the dataset.
        id2label (dict[int, str]): Mapping from class IDs to benthic attribute names
        label2id (dict[str, int]): Mapping from benthic attribute names to class IDs
        concept_mapping_flag (bool): Flag indicating whether to initialize concept mapping.
        num_concepts (Optional[int]): Number of unique concepts in the dataset (if concept mapping is enabled).
        id2concept (Optional[dict[int, str]]): Mapping from concept IDs to concept names (if concept mapping is enabled).
        concept2id (Optional[dict[str, int]]): Mapping from concept names to concept IDs (if concept mapping is enabled).
        benthic_concept_matrix (Optional[np.ndarray]): Benthic attribute to concept mapping matrix (if concept mapping is enabled).
        conceptid2labelid (Optional[dict[int, int]]): Mapping from concept IDs to benthic attribute class IDs (if concept mapping is enabled).
    Args:
        df_annotations (pd.DataFrame): DataFrame with all annotation data.
        df_images (pd.DataFrame): DataFrame with unique image entries.
        split (Optional[str], optional): Dataset split identifier. Defaults to None.
        transform (Optional, optional): Transformation function for images. Defaults to None.
        padding (Optional[int], optional): Padding value for point annotations of the segmentation mask. Defaults to None.
        class_subset (Optional[List[str]], optional): Optional list of benthic attribute names to filter the dataset. Defaults to None.
        concept_mapping_flag (bool, optional): Flag indicating whether to initialize concept mapping. Defaults to False.
    Methods:
        __len__(): Returns the number of annotation entries.
        __getitem__(idx): Retrieves the image at the given index, applies transformations, and returns it with a placeholder target.
        read_image(image_id): Abstract method to read an image given its ID. Must be implemented in subclasses.
        collate_fn(batch): Custom collate function to handle batching of images, masks, and annotations.
        initialize_concept_mapping(): Initializes concept mapping attributes for the dataset.
    """

    df_annotations: pd.DataFrame
    df_images: pd.DataFrame
    split: str | None
    transform: A.BasicTransform | None
    padding: int | None
    class_subset: list[str] | None
    num_classes: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    concept_mapping_flag: bool = False
    num_concepts: int | None = None
    id2concept: dict[int, str] | None = None
    concept2id: dict[str, int] | None = None
    benthic_concept_matrix: np.ndarray | None = None
    conceptid2labelid: dict[int, int] | None = None

    def __init__(
        self,
        df_annotations: pd.DataFrame,
        df_images: pd.DataFrame,
        split: str | None = None,
        transform: A.BasicTransform | None = None,
        padding: int | None = None,
        class_subset: list[str] | None = None,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        image_id = self.df_images.loc[idx, "image_id"]
        row_kwargs = self.df_images.loc[idx].to_dict()
        try:
            image = self.read_image(**row_kwargs)
        except Exception:
            return None, None, None
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

        return image, mask  # , annotations

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
    Each item returned is a tuple containing the image (as a tensor or ndarray) and and the target.
    Attributes:
        annotations_path (str): Path to the Parquet file containing image annotations.
        source_bucket (str): S3 bucket name containing the dataset files.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        annotations_path (str, optional): S3 path to the Parquet file with annotations. Defaults to a preset path.
        source_bucket (str, optional): S3 bucket name containing the dataset files. Defaults to "coral-reef-training".
        **base_kwargs: Additional keyword arguments passed to the BaseCoralDataset constructor.
    Methods:
        load_annotations(annotations_path): Loads annotations from the specified Parquet file on S3.
        read_image(image_id): Reads an image given its ID from S3.
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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    A PyTorch Dataset for loading CoralNet annotated coral reef images from a Parquet file stored on S3.
    This dataset reads image annotations from a Parquet file, retrieves images from S3, and applies optional transformations.
    Each item returned is a tuple containing the image (as a tensor or ndarray) and and the target.
    Attributes:
        annotations_path (str): Path to the Parquet file containing image annotations.
        This is created by merging all annotations of CoralNet sources in a separate notebook /nbs/datasets/CoralNet_Annotations.ipynb.
        source_bucket (str): S3 bucket name containing the dataset files.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        annotations_path (str, optional): S3 path to the Parquet file with annotations. Defaults to a preset path.
        source_bucket (str, optional): S3 bucket name containing the dataset files. Defaults to "coral-reef-training".
        **base_kwargs: Additional keyword arguments passed to the BaseCoralDataset constructor.
    Methods:
        load_annotations(annotations_path): Loads annotations from the specified Parquet file on S3.
        read_image(image_id): Reads an image given its ID from S3.
    """

    source_ids: list[int | str]
    source_bucket: str
    source_s3_prefix: str
    s3: boto3.client
    whitelist_sources: list[int | str] | None
    blacklist_sources: list[int | str] | None

    def __init__(
        self,
        annotations_path="coralnet_annotations_30112025.parquet",
        source_bucket: str = "dev-datamermaid-sm-sources",
        source_s3_prefix: str = "coralnet-public-images",
        whitelist_sources: list[int | str] | None = None,
        blacklist_sources: list[int | str] | None = None,
        **base_kwargs,
    ):
        self.annotations_path = annotations_path
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
        annotations_path = f"s3://{self.source_bucket}/{self.annotations_path}"
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
        """
        Read an image given its ID from S3.
        """
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


class CoralscapesDataset(Dataset[tuple[torch.Tensor | NDArray[Any], Any]]):
    """
    A PyTorch Dataset for loading Coralscapes annotated coral reef images from the Hugging Face Datasets library and mapping them to specified MERMAID classes.
    This dataset reads image annotations from the Coralscapes dataset, retrieves images, and applies optional transformations.
    Each item returned is a tuple containing the image (as a tensor or ndarray) and and the target.
    Attributes:
        split (Optional[str]): Optional dataset split identifier (e.g., 'train', 'val', 'test').
        transform (Optional): Optional transformation function to apply to images.
        padding (Optional[int]): Padding value for point annotations of the segmentation mask.
        class_subset (Optional[List[str]]): Optional list of benthic attribute names to filter the dataset.
        num_classes (int): Number of unique benthic attribute classes in the dataset.
        id2label (dict[int, str]): Mapping from class IDs to benthic attribute names
        label2id (dict[str, int]): Mapping from benthic attribute names to class IDs
        concept_mapping_flag (bool): Flag indicating whether to initialize concept mapping.
        num_concepts (Optional[int]): Number of unique concepts in the dataset (if concept mapping is enabled).
        id2concept (Optional[dict[int, str]]): Mapping from concept IDs to concept names (if concept mapping is enabled).
        concept2id (Optional[dict[str, int]]): Mapping from concept names to concept IDs (if concept mapping is enabled).
        benthic_concept_matrix (Optional[np.ndarray]): Benthic attribute to concept mapping matrix (if concept mapping is enabled).
        conceptid2labelid (Optional[dict[int, int]]): Mapping from concept IDs to benthic attribute class IDs (if concept mapping is enabled).
    """

    split: str | None
    transform: A.BasicTransform | None
    padding: int | None
    class_subset: list[str] | None
    num_classes: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    concept_mapping_flag: bool = False
    num_concepts: int | None = None
    id2concept: dict[int, str] | None = None
    concept2id: dict[str, int] | None = None
    benthic_concept_matrix: np.ndarray | None = None
    conceptid2labelid: dict[int, int] | None = None

    def __init__(
        self,
        split: str | None = None,
        transform: A.BasicTransform | None = None,
        class_subset: list[str] | None = None,
        concept_mapping_flag: bool = False,
    ):

        self.split = split
        self.transform = transform
        self.class_subset = class_subset
        self.concept_mapping_flag = concept_mapping_flag

        self.dataset = load_dataset("EPFL-ECEO/coralscapes")

        if self.split is not None:
            self.dataset = self.dataset[self.split]
        else:
            self.dataset = concatenate_datasets(
                [
                    self.dataset["train"],
                    self.dataset["validation"],
                    self.dataset["test"],
                ]
            )

        if self.class_subset is not None:
            self.num_classes = len(self.class_subset) + 1  # +1 for background
            self.id2label = {
                i: attribute for i, attribute in enumerate(self.class_subset, start=1)
            }
            self.label2id = {v: k for k, v in self.id2label.items()}

        self.labelmapping = self.initialize_coralscapes_mapping()

        if concept_mapping_flag:
            self.initialize_concept_mapping()

    def initialize_coralscapes_mapping(self):
        """
        Initialize the label mapping between the Coralscapes 39-class dataset and MERMAID.
        """
        id2label_coralscapes = {
            "1": "seagrass",
            "2": "trash",
            "3": "other coral dead",
            "4": "other coral bleached",
            "5": "sand",
            "6": "other coral alive",
            "7": "human",
            "8": "transect tools",
            "9": "fish",
            "10": "algae covered substrate",
            "11": "other animal",
            "12": "unknown hard substrate",
            "13": "background",
            "14": "dark",
            "15": "transect line",
            "16": "massive/meandering bleached",
            "17": "massive/meandering alive",
            "18": "rubble",
            "19": "branching bleached",
            "20": "branching dead",
            "21": "millepora",
            "22": "branching alive",
            "23": "massive/meandering dead",
            "24": "clam",
            "25": "acropora alive",
            "26": "sea cucumber",
            "27": "turbinaria",
            "28": "table acropora alive",
            "29": "sponge",
            "30": "anemone",
            "31": "pocillopora alive",
            "32": "table acropora dead",
            "33": "meandering bleached",
            "34": "stylophora alive",
            "35": "sea urchin",
            "36": "meandering alive",
            "37": "meandering dead",
            "38": "crown of thorn",
            "39": "dead clam",
        }
        coralscapes_39_to_mermaid = {
            "human": ["Unknown"],
            "background": ["Unknown", "Obscured"],
            "fish": ["Unknown"],
            "sand": ["Sand"],
            "rubble": ["Rubble"],
            "unknown hard substrate": ["Bare substrate"],
            "algae covered substrate": ["Turf algae"],
            "dark": ["Unknown"],
            "branching bleached": ["Bleached coral"],
            "branching dead": ["Dead coral"],
            "branching alive": ["Hard coral"],
            "stylophora alive": ["Stylophora"],
            "pocillopora alive": ["Pocillopora"],
            "acropora alive": ["Acropora"],
            "table acropora alive": ["Acropora"],
            "table acropora dead": ["Dead coral"],
            "millepora": ["Milleporidae"],
            "turbinaria": ["Turbinaria reniformis"],
            "other coral": ["Bleached coral"],
            "other coral dead": ["Dead coral"],
            "other coral alive": [
                "Hard coral"
            ],  # Not fully correct as it can include soft corals but no specific mapping available
            "other coral bleached": ["Bleached coral"],  # Newly added
            "massive/meandering alive": ["Hard coral"],
            "massive/meandering dead": ["Dead coral"],
            "massive/meandering bleached": ["Bleached coral"],
            "meandering alive": ["Hard coral"],
            "meandering dead": ["Dead coral"],
            "meandering bleached": ["Bleached coral"],
            "transect line": ["Tape"],
            "transect tools": ["Unknown"],
            "sea urchin": ["Sea urchin"],
            "sea cucumber": ["Sea cucumber"],
            "anemone": ["Anemone"],
            "sponge": ["Sponge"],
            "clam": ["Tridacna giant clam"],
            "other animal": ["Other invertebrates"],
            "trash": ["Trash"],
            "seagrass": ["Seagrass"],
            "crown of thorn": ["Acanthaster planci"],
            "dead clam": ["Unknown"],
        }
        if self.class_subset is None:
            self.class_subset = set(
                [
                    mermaid_class[0]
                    for mermaid_class in coralscapes_39_to_mermaid.values()
                ]
            )
            self.id2label = {
                i: label for i, label in enumerate(sorted(self.class_subset), start=1)
            }
            self.label2id = {label: i for i, label in self.id2label.items()}
            self.num_classes = len(self.class_subset) + 1  # +1 for background
        id_coralscapes_to_mermaid = {
            int(k): self.label2id.get(coralscapes_39_to_mermaid[v][0], 0)
            for k, v in id2label_coralscapes.items()
        }
        id_coralscapes_to_mermaid[0] = (
            0  # Adding mapping for background / unlabeled class
        )

        return id_coralscapes_to_mermaid

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | NDArray[Any], Any]:
        image, mask = np.array(self.dataset[idx]["image"]), self.dataset[idx]["label"]
        mask = np.vectorize(self.labelmapping.get)(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].transpose(2, 0, 1)
            mask = transformed["mask"]

        return image, mask

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

        images, masks = zip(*batch)

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
