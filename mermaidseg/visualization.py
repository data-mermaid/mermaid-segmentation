import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_legend_elements(
    annotations: pd.DataFrame, include_growth_form: bool = False
) -> list | tuple[list, list]:
    """Generate matplotlib legend elements for benthic attributes and optionally growth forms.

    Args:
        annotations (pd.DataFrame): Must contain 'benthic_attribute_name' and 'benthic_color'.
            If `include_growth_form` is True, must also contain 'growth_form_name' and
            'growth_form_marker'.
        include_growth_form (bool, optional): Whether to include growth form legend elements.
            Defaults to False.
    Returns:
        list | tuple[list, list]: If `include_growth_form` is False, a list of Line2D legend
            elements for benthic attributes. If True, a tuple
            ``(benthic_legend_elements, growth_legend_elements)``.
    """
    unique_benthic = annotations[["benthic_attribute_name", "benthic_color"]].drop_duplicates()
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
            unique_benthic["benthic_attribute_name"], unique_benthic["benthic_color"], strict=False
        )
    ]
    if not include_growth_form:
        return benthic_legend_elements

    unique_growth = (
        annotations[["growth_form_name", "growth_form_marker"]].astype(str).drop_duplicates()
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
            unique_growth["growth_form_name"], unique_growth["growth_form_marker"], strict=False
        )
    ]

    return benthic_legend_elements, growth_legend_elements


def denormalize_image(
    image: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Reverse ImageNet normalisation and return a uint8 image.

    Args:
        image (np.ndarray): Normalised image with shape (C, H, W).
        mean (np.ndarray | None): Per-channel mean used during normalisation.
            Defaults to ImageNet mean ``[0.485, 0.456, 0.406]``.
        std (np.ndarray | None): Per-channel std used during normalisation.
            Defaults to ImageNet std ``[0.229, 0.224, 0.225]``.
    Returns:
        np.ndarray: Denormalised image with shape (C, H, W), dtype uint8, values in [0, 255].
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])

    unnormalized_image = (image * std[:, None, None]) + mean[:, None, None]
    return (unnormalized_image * 255).astype(np.uint8)
