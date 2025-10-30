"""
title: mermaidseg.visualization
abstract: Module that contains visualization functions and utilities.
author: Viktor Domazetoski
date: 30-10-2025

Functions:
    get_legend_elements: Generate legend elements for benthic attributes and optionally growth forms for use in matplotlib plots.
    denormalize_image: Denormalizes an image that was previously normalized using the given mean and standard deviation.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_legend_elements(
    annotations: pd.DataFrame, include_growth_form: bool = False
) -> Union[List, Tuple[List, List]]:
    """
    Generate legend elements for benthic attributes and optionally growth forms for use in matplotlib plots.
    Parameters
    ----------
    annotations : pandas.DataFrame
        DataFrame containing at least the columns 'benthic_attribute_name', 'benthic_color'.
        If `include_growth_form` is True, must also contain 'growth_form_name' and 'growth_form_marker'.
    include_growth_form : bool, optional
        Whether to include legend elements for growth forms. Default is False.
    Returns
    -------
    list or tuple of lists
        If `include_growth_form` is False, returns a list of matplotlib Line2D objects for benthic attributes.
        If `include_growth_form` is True, returns a tuple:
            (benthic_legend_elements, growth_legend_elements)
        where each is a list of matplotlib Line2D objects for the respective legend.
    Notes
    -----
    The function assumes that the color and marker codes are compatible with matplotlib.
    """

    unique_benthic = annotations[
        ["benthic_attribute_name", "benthic_color"]
    ].drop_duplicates()
    benthic_legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            label=name,
            markersize=10,
        )
        for name, color in zip(
            unique_benthic["benthic_attribute_name"], unique_benthic["benthic_color"]
        )
    ]
    if not include_growth_form:
        return benthic_legend_elements

    unique_growth = (
        annotations[["growth_form_name", "growth_form_marker"]]
        .astype(str)
        .drop_duplicates()
    )

    growth_legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#4c72b0",
            label=name,
            markersize=10,
            linestyle="None",
        )
        for name, marker in zip(
            unique_growth["growth_form_name"], unique_growth["growth_form_marker"]
        )
    ]

    return benthic_legend_elements, growth_legend_elements


def denormalize_image(
    image,
    mean=np.array([0.485, 0.456, 0.406]),
    std=np.array([0.229, 0.224, 0.225]),
):
    """
    Denormalizes an image that was previously normalized using the given mean and standard deviation.
    Args:
        image (Union[torch.Tensor, numpy.ndarray]): The normalized image to be denormalized.
                                                    Expected shape is (C, H, W).
        mean (numpy.ndarray, optional): The mean used for normalization.
                                        Default is np.array([0.485, 0.456, 0.406]) as used in ImageNet.
        std (numpy.ndarray, optional): The standard deviation used for normalization.
                                        Default is np.array([0.229, 0.224, 0.225]) as used in ImageNet.
    Returns:
        numpy.ndarray: The denormalized image with pixel values in the range [0, 255] and dtype uint8.
    """

    unnormalized_image = (image * std[:, None, None]) + mean[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    return unnormalized_image
