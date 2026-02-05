"""
title: mermaidseg.datasets.utils
abstract: Module that contains dataset loading and utility functions.
author: Viktor Domazetoski
date: 28-08-2025

Functions:
    get_image_s3: Fetches an image from an S3 bucket and returns it as a PIL Image object.
    create_annotation_mask: Creates an annotation mask for a given image.
    calculate_weights: Calculate class weights for a given dataset.
"""

import io
from typing import Dict, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm


def get_image_s3(
    s3: boto3.client,
    bucket: str,
    key: str,
    thumbnail: bool = False,
):
    """
    Fetches an image from an S3 bucket and returns it as a PIL Image object.
    Args:
        s3 (boto3.client): The Boto3 S3 client used to interact with S3.
        bucket (str): The name of the S3 bucket.
        key (str): The key (path) of the image in the S3 bucket.
        thumbnail (bool, optional): If True, fetches the thumbnail version of the image by modifying the key. Defaults to False.
    Returns:
        PIL.Image.Image: The image loaded from S3 as a PIL Image object.
    """

    if thumbnail:
        key = key.replace(".png", "_thumbnail.png")

    response = s3.get_object(Bucket=bucket, Key=key)
    image_data = response["Body"].read()

    image = Image.open(io.BytesIO(image_data))
    return image


def create_annotation_mask(
    annotations: pd.DataFrame,
    shape: Tuple[int, int],
    label2id: Dict[str, int],
    padding: Optional[int] = None,
) -> np.ndarray:
    """
    Creates an annotation mask for a given image based on provided annotations.
    Args:
        annotations (pd.DataFrame): DataFrame containing annotation rows with 'row', 'col', and 'benthic_attribute_name' columns.
        shape (Tuple[int, int]): Shape of the output mask (height, width).
        label2id (Dict[str, int]): Mapping from label names to integer IDs.
    Returns:
        np.ndarray: Annotation mask with integer class IDs.
    """
    ## TODO: Make Padding percentage based so that it is applicable to all class sizes
    mask = np.zeros(shape[:2])
    for _, annotation in annotations.iterrows():
        if annotation["benthic_attribute_name"] is not None:
            if padding is not None and padding > 0:
                mask[
                    annotation["row"] - padding : annotation["row"] + padding,
                    annotation["col"] - padding : annotation["col"] + padding,
                ] = label2id[annotation["benthic_attribute_name"]]
            else:
                mask[annotation["row"], annotation["col"]] = label2id[
                    annotation["benthic_attribute_name"]
                ]

    return mask


def calculate_weights(dataset: Dataset, const: int = 2000000) -> torch.Tensor:
    """
    Calculate class weights for a given dataset.
    This function computes the weights for each class in the dataset based on
    the frequency of each class label. The weights are inversely proportional
    to the square root of the class frequency, adjusted by a constant value.
    Args:
        dataset (Dataset): The dataset object which contains images and labels.
                            It should have an attribute `N_classes` indicating
                            the number of classes.
        const (int, optional): A constant value added to the class frequency
                                to avoid division by zero and to smooth the weights.
                                Default is 2000000.
    Returns:
        torch.Tensor: A tensor of weights for each class, normalized by the mean weight.
    """

    label_counts = {i: 0 for i in range(dataset.N_classes)}
    for i in tqdm(range(len(dataset))):
        _, label, _ = dataset[i]
        unique_labels = np.unique(label, return_counts=True)
        for label_id, count in zip(*unique_labels):
            label_counts[label_id] += int(count)

    weights = np.zeros(dataset.N_classes)
    for index, count in label_counts.items():
        weights[index] = 1 / np.sqrt(count + const)
    weight = torch.tensor(weights).float()
    weight /= weight.mean()

    return weight


def _joint_collate(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to combine a list of samples into a batch.
    """
    images, labels = zip(*batch)
    images = default_collate(images)
    labels = default_collate(labels)
    return images, labels


def get_coralnet_sources():
    """
    Discover and validate CoralNet source folders stored in the S3 bucket "dev-datamermaid-sm-sources".
    Returns:
        whitelist: A list of all valid CoralNet source folder names that contain both
              'annotations.csv' and 'image_list.csv' files.
    """

    s3 = boto3.client("s3")
    bucket_name = "dev-datamermaid-sm-sources"

    response = s3.list_objects_v2(Bucket=bucket_name, Delimiter="/")
    if "CommonPrefixes" in response:
        folders_new = [prefix["Prefix"] for prefix in response["CommonPrefixes"]]
        folder = "coralnet-public-images/"
        sub_response = s3.list_objects_v2(
            Bucket=bucket_name, Prefix=folder, Delimiter="/"
        )
        if "CommonPrefixes" in sub_response:
            print("Subfolders in coralnet-public-images/:")
            folders_new = [
                prefix["Prefix"] for prefix in sub_response["CommonPrefixes"]
            ]
            folders_new = [
                folder.replace("coralnet-public-images/", "") for folder in folders_new
            ]
        else:
            print("No subfolders found in coralnet-public-images/")
    else:
        print("No folders found in the bucket")

    whitelist_sources = []
    for source in tqdm(folders_new):
        if not source.startswith("s"):
            print(source)

        file_key = f"coralnet-public-images/{source}annotations.csv"

        try:
            s3.head_object(Bucket=bucket_name, Key=file_key)
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"File {file_key} not found in bucket")
                continue

        file_key = f"coralnet-public-images/{source}image_list.csv"

        try:
            s3.head_object(Bucket=bucket_name, Key=file_key)
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"File {file_key} not found in bucket")
                continue
        whitelist_sources.append(source)
    return whitelist_sources
