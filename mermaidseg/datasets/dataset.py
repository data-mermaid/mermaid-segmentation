"""
title: mermaidseg.datasets.dataset
abstract: Module that contains dataset classes & functionality.
author: Viktor Domazetoski
date: 21-08-2025

Classes:
    MermaidDataset
"""

from typing import Any, Optional, Tuple, Union

import boto3
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mermaidseg.datasets.utils import get_image_s3, create_annotation_mask
from numpy.typing import NDArray
from torch.utils.data import Dataset


class MermaidDataset(Dataset[Tuple[Union[torch.Tensor, NDArray[Any]], Any]]):
    """
    A PyTorch Dataset for loading annotated coral reef images from a Parquet file stored on S3.
    This dataset reads image annotations from a Parquet file, retrieves images from S3, and applies optional transformations.
    Each item returned is a tuple containing the image (as a tensor or ndarray) and a placeholder for the target (currently None).
    Attributes:
        annotations_path (str): Path to the Parquet file containing image annotations.
        df_annotations (pd.DataFrame): DataFrame with all annotation data.
        df_images (pd.DataFrame): DataFrame with unique image entries.
        split (Optional[str]): Optional dataset split identifier (e.g., 'train', 'val', 'test').
        transform (Optional): Optional transformation function to apply to images.
        s3 (boto3.client): Boto3 S3 client for accessing images.
    Args:
        annotations_path (str, optional): S3 path to the Parquet file with annotations. Defaults to a preset path.
        split (Optional[str], optional): Dataset split identifier. Defaults to None.
        transform (Optional, optional): Transformation function for images. Defaults to None.
    Methods:
        __len__(): Returns the number of annotation entries.
        __getitem__(idx): Retrieves the image at the given index, applies transformations, and returns it with a placeholder target.
    """

    annotations_path: str
    df_annotations: pd.DataFrame
    df_images: pd.DataFrame
    split: Optional[str]
    transform: Optional

    def __init__(
        self,
        annotations_path: str = "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet",
        split: Optional[str] = None,
        transform: Optional = None,
    ):
        self.annotations_path = annotations_path
        self.df_annotations = pd.read_parquet(self.annotations_path)
        self.df_images = self.df_annotations[
            ["image_id", "region_id", "region_name"]
        ].drop_duplicates(subset=["image_id"]).reset_index(drop=True) ## NEWLY Added
        self.split = split
        self.transform = transform
        self.s3 = boto3.client("s3")
        
        self.N_classes = self.df_annotations["benthic_attribute_name"].nunique()
        self.id2label = {i:attribute for i, attribute in 
                                enumerate(
                                 self.df_annotations["benthic_attribute_name"].value_counts().index.tolist()
                                    )
                        }
        self.label2id = {v:k for k,v in self.id2label.items()}
        
        self.vis_dict = {}
        self.vis_dict["benthic"] = dict(
            zip(
                self.df_annotations["benthic_attribute_name"]
                .value_counts()
                .index.tolist(),
                sns.color_palette(
                    "deep",
                    n_colors=self.N_classes,
                ).as_hex(),
            )
        )

        self.vis_dict["growth_form"] = dict(
            zip(
                self.df_annotations["growth_form_name"]
                .astype(str)
                .value_counts()
                .index.tolist(),
                ["s", "o", ".", "v", "^", "<", ">", "2", "p", "*", "+", "x", "D", "d"][
                    : self.df_annotations["growth_form_name"].nunique()
                ],
            )
        )

    def __len__(self):
        return self.df_images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, NDArray[Any]], Any]:
        image_id = self.df_images.loc[idx, "image_id"]
        image = np.array(get_image_s3(image_id, self.s3, thumbnail=True).convert("RGB"))

        annotations = self.df_annotations.loc[
            self.df_annotations["image_id"] == image_id,
            [
                "point_id",
                "row",
                "col",
                "benthic_attribute_id",
                "benthic_attribute_name",
                "growth_form_id",
                "growth_form_name",
            ],
        ]

        # Scale annotations to image size (if using thumbnails)
        annotations["row"] = (
            annotations["row"]
            / (annotations["row"].max() + annotations["row"].min())
            * image.shape[0]
        ).astype(int)

        annotations["col"] = (
            annotations["col"]
            / (annotations["col"].max() + annotations["col"].min())
            * image.shape[1]
        ).astype(int)

        annotations["benthic_color"] = annotations["benthic_attribute_name"].apply(lambda x: self.vis_dict["benthic"][x])
        annotations["growth_form_marker"] = annotations["growth_form_name"].apply(lambda x: self.vis_dict["growth_form"][str(x)])
        
        mask = create_annotation_mask(annotations, image.shape, self.label2id)
        # if self.transform:
        #     transformed = self.transform(image=image)
        #     image = transformed["image"]
        return image, mask, annotations
