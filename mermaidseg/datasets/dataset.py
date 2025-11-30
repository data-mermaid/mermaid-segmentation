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
from typing import Any, List, Optional, Tuple, Union

import albumentations as A
import boto3
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import tqdm
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
    vis_dict: dict[str, dict[str, Union[str, Any]]]

    def __init__(
        self,
        df_annotations: pd.DataFrame,
        df_images: pd.DataFrame,
        split: Optional[str] = None,
        transform: Optional[A.BasicTransform] = None,
        padding: Optional[int] = None,
        class_subset: Optional[List[str]] = None,
    ):
        self.df_annotations = df_annotations
        self.df_images = df_images
        self.split = split
        self.transform = transform
        self.padding = padding
        self.class_subset = class_subset

        if self.class_subset is not None:
            self.df_annotations = self.df_annotations[
                self.df_annotations["benthic_attribute_name"].apply(
                    lambda x: x in self.class_subset
                )
            ]

            self.df_images = (
                self.df_annotations[["image_id", "region_id", "region_name"]]
                .drop_duplicates(subset=["image_id"])
                .reset_index(drop=True)  # TODO: Update based on CoralNet
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

    def __len__(self):
        return self.df_images.shape[0]

    def read_image(self, image_id: str) -> NDArray[Any]:
        """
        Read an image given its ID. Needs to be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, NDArray[Any]], Any]:
        image_id = self.df_images.loc[idx, "image_id"]
        image = self.read_image(image_id)

        annotations = self.df_annotations.loc[
            self.df_annotations["image_id"] == image_id,
            [
                "point_id",
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
        images, masks, annotations = zip(*batch)

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


class MermaidDataset(BaseCoralDataset):
    """
    A PyTorch Dataset for loading annotated coral reef images from a Parquet file stored on S3.
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

    def read_image(self, image_id: str) -> NDArray[Any]:
        """
        Read an image given its ID. Needs to be implemented in subclasses.
        """
        key = f"mermaid/{image_id}.png"  # f"mermaid/{image_id}_thumbnail.png"
        image = np.array(
            get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB")
        )
        return image


class Mermaid15Dataset(MermaidDataset):
    """
    A subset of the Mermaid dataset with only 15 select classes for experimentation purposes.
    Labels chosen as most common 15 classes, defined explicitly to avoid discrepancies based on data distribution.
    """

    def __init__(
        self,
        **mermaid_kwargs,
    ):
        classes_mermaid15 = [
            "Macroalgae",
            "Rubble",
            "Sand",
            "Porites",
            "Crustose coralline algae",
            "Bare substrate",
            "Hard coral",
            "Turf algae",
            "Millepora",
            "Tape",
            "Soft coral",
            "Acropora",
            "Pocillopora",
            "Porites lobata",
            "Montipora",
        ]

        mermaid_kwargs.pop("class_subset", None)  # Remove if exists to avoid conflict

        super().__init__(
            class_subset=classes_mermaid15,
            **mermaid_kwargs,
        )


# class CoralNetDataset(Dataset[Tuple[Union[torch.Tensor, NDArray[Any]], Any]]):
#     """
#     A PyTorch Dataset for loading annotated coral reef images from a CoralNet sources.
#     Each item returned is a tuple containing the image (as a tensor or ndarray) and a placeholder for the target (currently None).
#     Attributes:
#         source_path (str): Path to the Parquet file containing image annotations.
#         df_annotations (pd.DataFrame): DataFrame with all annotation data.
#         df_images (pd.DataFrame): DataFrame with unique image entries.
#         split (Optional[str]): Optional dataset split identifier (e.g., 'train', 'val', 'test').
#         transform (Optional): Optional transformation function to apply to images.
#         s3 (boto3.client): Boto3 S3 client for accessing images.
#     Args:
#         annotations_path (str, optional): S3 path to the Parquet file with annotations. Defaults to a preset path.
#         split (Optional[str], optional): Dataset split identifier. Defaults to None.
#         transform (Optional, optional): Transformation function for images. Defaults to None.
#     Methods:
#         __len__(): Returns the number of annotation entries.
#         __getitem__(idx): Retrieves the image at the given index, applies transformations, and returns it with a placeholder target.
#     """

#     source_ids: List[Union[int, str]]
#     source_bucket: str
#     source_s3_prefix: str
#     s3: boto3.client
#     df_annotations: pd.DataFrame
#     df_images: pd.DataFrame
#     split: Optional[str]
#     transform: Optional[A.BasicTransform]

#     def __init__(
#         self,
#         source_ids: List[Union[int, str]],
#         source_bucket: str = "dev-datamermaid-sm-sources",
#         source_s3_prefix: str = "coralnet-public-images",
#         split: Optional[str] = None,
#         transform: Optional[A.BasicTransform] = None,
#     ):
#         self.source_ids = source_ids
#         self.source_bucket = source_bucket
#         self.source_s3_prefix = source_s3_prefix
#         self.s3 = boto3.client("s3")

#         print("Initialize CoralNet to Mermaid LabelMapping")
#         self.labelmapping = self.initialize_coralnet_mapping()

#         print("Reading annotations from sources")
#         for i, source_id in tqdm.tqdm(enumerate(source_ids)):
#             if i == 0:
#                 self.df_annotations = pd.read_csv(
#                     f"s3://{self.source_bucket}/{self.source_s3_prefix}/s{source_id}/annotations.csv"
#                 )
#                 df_images = pd.read_csv(
#                     f"s3://{self.source_bucket}/{self.source_s3_prefix}/s{source_id}/image_list.csv"  # Perhaps this is unnecessary and can just use tha annotations as in Mermaid
#                 )
#                 df_images["Name"] = df_images["Name"].apply(
#                     lambda x: x.replace(" - Confirmed", "")
#                 )
#                 df_images["image_id"] = df_images["Image Page"].apply(
#                     lambda x: x.replace("/image/", "").replace("/view/", "")
#                 )
#                 self.df_annotations = pd.merge(
#                     self.df_annotations,
#                     df_images,
#                     left_on="Name",
#                     right_on="Name",
#                     how="left",
#                     suffixes=("", "_y"),
#                 )
#                 self.df_annotations["source_id"] = source_id
#             else:
#                 obj = self.s3.get_object(
#                     Bucket=self.source_bucket,
#                     Key=f"{self.source_s3_prefix}/s{source_id}/annotations.csv",
#                 )
#                 df_tmp = pd.read_csv(io.BytesIO(obj["Body"].read()))

#                 obj = self.s3.get_object(
#                     Bucket=self.source_bucket,
#                     Key=f"{self.source_s3_prefix}/s{source_id}/image_list.csv",
#                 )
#                 df_images = pd.read_csv(io.BytesIO(obj["Body"].read()))
#                 # df_tmp = pd.read_csv(
#                 #     f"s3://{self.source_bucket}/{self.source_s3_prefix}/s{source_id}/annotations.csv"
#                 # )
#                 # df_images = pd.read_csv(
#                 #     f"s3://{self.source_bucket}/{self.source_s3_prefix}/s{source_id}/image_list.csv"
#                 # )
#                 df_images["Name"] = df_images["Name"].apply(
#                     lambda x: x.replace(" - Confirmed", "")
#                 )
#                 df_images["image_id"] = df_images["Image Page"].apply(
#                     lambda x: x.replace("/image/", "").replace("/view/", "")
#                 )
#                 df_tmp = pd.merge(
#                     df_tmp,
#                     df_images,
#                     left_on="Name",
#                     right_on="Name",
#                     how="left",
#                     suffixes=("", "_y"),
#                 )
#                 df_tmp["source_id"] = source_id
#                 self.df_annotations = pd.concat(
#                     [self.df_annotations, df_tmp], ignore_index=True
#                 )

#         self.df_annotations = self.df_annotations.rename(
#             columns={
#                 "image_id": "image_id",
#                 "Row": "row",
#                 "Column": "col",
#                 "Label ID": "coralnet_id",
#             }
#         )

#         self.df_annotations["benthic_attribute_name"] = self.df_annotations[
#             "coralnet_id"
#         ].apply(lambda x: self.labelmapping.get(str(x), None))

#         self.df_annotations = self.df_annotations[
#             ["source_id", "image_id", "row", "col", "benthic_attribute_name"]
#         ]  # Can add new columns here if needed, keeping only most important for the start

#         self.df_images = (
#             self.df_annotations[
#                 ["source_id", "image_id"]  # Can add new columns here if needed
#             ]
#             .drop_duplicates(subset=["source_id", "image_id"])
#             .reset_index(drop=True)
#         )
#         self.split = split
#         self.transform = transform

#         self.num_classes = (
#             self.df_annotations["benthic_attribute_name"].nunique() + 1
#         )  # +1 for background
#         self.id2label = {
#             i: attribute
#             for i, attribute in enumerate(
#                 self.df_annotations[
#                     "benthic_attribute_name"
#                 ]  ##TODO: See if you need to update these things (id2label needs to start with one, here and above)
#                 .value_counts()
#                 .index.tolist(),
#                 start=1,
#             )
#         }
#         self.label2id = {v: k for k, v in self.id2label.items()}

#         self.vis_dict = {}
#         self.vis_dict["benthic"] = dict(
#             zip(
#                 self.df_annotations["benthic_attribute_name"]
#                 .value_counts()
#                 .index.tolist(),
#                 sns.color_palette(
#                     "deep",
#                     n_colors=self.num_classes,
#                 ).as_hex(),
#             )
#         )

#     def __len__(self):
#         return self.df_images.shape[0]

#     def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, NDArray[Any]], Any]:
#         image_id = self.df_images.loc[idx, "image_id"]
#         source_id = self.df_images.loc[idx, "source_id"]
#         key = f"{self.source_s3_prefix}/s{source_id}/images/{image_id}.jpg"
#         image = np.array(
#             get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB")
#         )

#         annotations = self.df_annotations.loc[
#             (self.df_annotations["source_id"] == source_id)
#             * (self.df_annotations["image_id"] == image_id),
#             [
#                 # "point_id",
#                 "row",
#                 "col",
#                 # "benthic_attribute_id",
#                 "benthic_attribute_name",
#                 # "growth_form_id",
#                 # "growth_form_name",
#             ],
#         ]
#         annotations["benthic_color"] = annotations["benthic_attribute_name"].apply(
#             lambda x: self.vis_dict["benthic"].get(x, "#222222")
#         )

#         mask = create_annotation_mask(annotations, image.shape, self.label2id)
#         if self.transform:
#             transformed = self.transform(image=image)
#             image = transformed["image"]
#         return image, mask, annotations

#     def initialize_coralnet_mapping(
#         self,
#         mapping_endpoint="https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet",
#     ):
#         response = requests.get(mapping_endpoint)
#         data = response.json()
#         labelset = data["results"]

#         while data["next"]:
#             response = requests.get(data["next"])
#             data = response.json()
#             labelset.extend(data["results"])
#         label_mapping = {
#             label["provider_id"]: label["benthic_attribute_name"] for label in labelset
#         }
#         return label_mapping


# class CoralNet15Dataset(CoralNetDataset):
#     """
#     A subset of the Coralnet dataset with only 15 select classes for experimentation purposes. Labels chosen as most common ones and a few select ones.
#     Each item returned is a tuple containing the image (as a tensor or ndarray) and a placeholder for the target (currently None).
#     Attributes:
#         source_bucket (str): S3 bucket name containing the dataset files.
#         source_s3_prefix (str): S3 prefix path to the dataset files.
#         split (Optional[str]): Dataset split identifier (e.g., 'train', 'val', 'test').
#         transform (Optional[A.BasicTransform]): Albumentations transform to apply to images.
#         padding (Optional[int]): Padding value for image processing.
#     Args:
#         annotations_path (str, optional): S3 path to the Parquet file with annotations. Defaults to a preset path.
#         split (Optional[str], optional): Dataset split identifier. Defaults to None.
#         transform (Optional, optional): Transformation function for images. Defaults to None.
#     Methods:
#         __len__(): Returns the number of annotation entries.
#         __getitem__(idx): Retrieves the image at the given index, applies transformations, and returns it with a placeholder target.
#     """

#     source_bucket: str
#     source_s3_prefix: str
#     s3: boto3.client
#     df_annotations: pd.DataFrame
#     df_images: pd.DataFrame
#     split: Optional[str]
#     transform: Optional[A.BasicTransform]
#     padding: Optional[int]

#     def __init__(
#         self,
#         source_bucket: str = "dev-datamermaid-sm-sources",
#         source_s3_prefix: str = "coralnet-public-images",
#         split: Optional[str] = None,
#         transform: Optional[A.BasicTransform] = None,
#         padding: Optional[int] = None,
#     ):
#         self.__init__(
#             source_ids=["1"],  # Dummy source id to initialize
#             source_bucket=source_bucket,
#             source_s3_prefix=source_s3_prefix,
#             split=split,
#             transform=transform,
#             padding=padding,
#         )

#         self.source_bucket = source_bucket
#         self.source_s3_prefix = source_s3_prefix
#         self.s3 = boto3.client("s3")
#         self.padding = padding

#         print("Initialize CoralNet to Mermaid LabelMapping")
#         self.labelmapping = self.initialize_coralnet_mapping()

#         self.df_annotations = pd.read_csv(
#             f"s3://{self.source_bucket}/annotations_1-100.csv"
#         )
#         self.df_images = pd.read_csv(f"s3://{self.source_bucket}/images_1-100.csv")

#         self.split = split
#         self.transform = transform

#         self.classes_mermaid15 = [
#             "Macroalgae",
#             "Rubble",
#             "Sand",
#             "Porites",
#             "Crustose coralline algae",
#             "Bare substrate",
#             "Hard coral",
#             "Turf algae",
#             "Millepora",
#             # "Tape",
#             "Soft coral",
#             "Acropora",
#             "Pocillopora",
#             "Porites lobata",
#             "Montipora",
#         ]

#         self.df_annotations = self.df_annotations[
#             self.df_annotations["benthic_attribute_name"].apply(
#                 lambda x: x in self.classes_mermaid15
#             )
#         ]
#         self.df_annotations = self.df_annotations[
#             self.df_annotations["image_id"].notna()
#         ]
#         self.df_annotations["image_id"] = self.df_annotations["image_id"].astype(int)

#         image_filter = self.df_annotations["image_id"].unique()
#         self.df_images = self.df_images[
#             self.df_images["image_id"].apply(lambda x: x in image_filter)
#         ].reset_index(drop=True)
#         self.df_images["image_id"] = self.df_images["image_id"].astype(int)

#         self.num_classes = (
#             self.df_annotations["benthic_attribute_name"].nunique() + 1
#         )  # +1 for background
#         self.id2label = {
#             i: attribute for i, attribute in enumerate(self.classes_mermaid15, start=1)
#         }
#         # self.id2label = {
#         #     i: attribute
#         #     for i, attribute in enumerate(
#         #         self.df_annotations[
#         #             "benthic_attribute_name"
#         #         ]  ##TODO: See if you need to update these things (id2label needs to start with one, here and above)
#         #         .value_counts()
#         #         .index.tolist()
#         #     )
#         # }
#         self.label2id = {v: k for k, v in self.id2label.items()}

#         self.vis_dict = {}
#         self.vis_dict["benthic"] = dict(
#             zip(
#                 self.df_annotations["benthic_attribute_name"]
#                 .value_counts()
#                 .index.tolist(),
#                 sns.color_palette(
#                     "deep",
#                     n_colors=self.num_classes,
#                 ).as_hex(),
#             )
#         )

#     def __len__(self):
#         return self.df_images.shape[0]

#     def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, NDArray[Any]], Any]:
#         image_id = self.df_images.loc[idx, "image_id"]
#         source_id = self.df_images.loc[idx, "source_id"]
#         key = f"{self.source_s3_prefix}/s{source_id}/images/{image_id}.jpg"
#         image = np.array(
#             get_image_s3(s3=self.s3, bucket=self.source_bucket, key=key).convert("RGB")
#         )

#         annotations = self.df_annotations.loc[
#             (self.df_annotations["source_id"] == source_id)
#             * (self.df_annotations["image_id"] == image_id),
#             [
#                 "row",
#                 "col",
#                 "benthic_attribute_name",
#             ],
#         ]
#         annotations["benthic_color"] = annotations["benthic_attribute_name"].apply(
#             lambda x: self.vis_dict["benthic"].get(x, "#222222")
#         )

#         mask = create_annotation_mask(
#             annotations,
#             image.shape,
#             self.label2id,
#             padding=int(self.padding * image.shape[0] / 1000),
#         )

#         if self.transform:
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed["image"].transpose(2, 0, 1)
#             mask = transformed["mask"]
#         return image, mask, annotations

#     def initialize_coralnet_mapping(
#         self,
#         mapping_endpoint="https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet",
#     ):
#         response = requests.get(mapping_endpoint)
#         data = response.json()
#         labelset = data["results"]

#         while data["next"]:
#             response = requests.get(data["next"])
#             data = response.json()
#             labelset.extend(data["results"])
#         label_mapping = {
#             label["provider_id"]: label["benthic_attribute_name"] for label in labelset
#         }
#         return label_mapping

#     def collate_fn(self, batch):
#         """
#         Collate function for MermaidDataset and CoralNetDataset.
#         Args:
#             batch: List of tuples (image, mask, annotations)
#         Returns:
#             images: Tensor or ndarray batch of images
#             masks: Tensor or ndarray batch of masks
#             annotations: List of annotation DataFrames
#         """
#         images, masks, annotations = zip(*batch)

#         # Handle empty batch
#         if len(images) == 0:
#             return torch.tensor([]), torch.tensor([]), []

#         # Convert to tensors if they aren't already
#         if isinstance(images[0], torch.Tensor):
#             images = torch.stack(images)
#             masks = torch.stack(masks)
#         else:
#             # Convert numpy arrays to tensors for consistency
#             images = torch.stack(
#                 [
#                     torch.from_numpy(img) if isinstance(img, np.ndarray) else img
#                     for img in images
#                 ]
#             )
#             masks = torch.stack(
#                 [
#                     torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
#                     for mask in masks
#                 ]
#             )

#         return images, masks
