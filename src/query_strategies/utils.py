"""Module containing functions used for different query strategies."""

from typing import Callable, List, Literal, Tuple
import numpy as np

import torch


def select_uncertainty_calculation(
    calculation_method: Literal["distance", "entropy"]
) -> Callable:
    """
    Selects the calculation function based on the provided name.
    Args:
        calculation_method (str): Name of the calculation method.
            values: `"distance"` |  `"entropy"`.

    Returns:
        A callable function to calculate uncertainty based on predictions.
    """
    if calculation_method == "distance":
        return distance_to_max_uncertainty
    if calculation_method == "entropy":
        return entropy
    print(
        "No valid calculation method provided, choosing default method: distance to max uncertainty."
    )
    return distance_to_max_uncertainty


def distance_to_max_uncertainty(
    predictions: torch.Tensor, max_uncertainty_value: float = 0.5, **kwargs
) -> np.ndarray:
    r"""
    Calculates the uncertainties based on the distance to a maximum uncertainty value:
        .. math::
            \sum | max\_uncertainty\_value - predictions |
    Args:
        predictions (torch.Tensor): The predictions of the model.
        max_uncertainty_value (float, optional): The maximum value of uncertainty in the predictions.
            (default = 0.5)
        **kwargs: Keyword arguments specific for this calculation.

    Returns:
        Uncertainty value for each image in the batch of predictions.
    """
    if kwargs.get("exclude_background", False):
        predictions = predictions[:, 1:, :, :]
    uncertainty = (
        torch.sum(torch.abs(max_uncertainty_value - predictions), (1, 2, 3))
        .cpu()
        .numpy()
    )
    return uncertainty


def entropy(
    predictions: torch.Tensor, max_uncertainty_value: float = 0.5, **kwargs
) -> np.ndarray:
    r"""
    Calculates the uncertainties based on the entropy of the distance to a maximum uncertainty value:
        .. math::
            - \sum | max\_uncertainty\_value - predictions | \cdot | \log({max\_uncertainty\_value - predictions}) |
    Args:
        predictions (torch.Tensor): The predictions of the model.
        max_uncertainty_value (float, optional): The maximum value of uncertainty in the predictions.
          (default = 0.5)
        **kwargs: Keyword arguments specific for this calculation.
            epsilon (float): The smoothing value to avoid the magic number.
                (default = 1e-10)

    Returns:
        Uncertainty value for each image in the batch of predictions.
    """
    # pylint: disable=unused-argument
    # Smoothing to avoid taking log of zero
    epsilon = kwargs.get("epsilon", 1e-10)
    predictions[predictions == max_uncertainty_value] = max_uncertainty_value + epsilon
    uncertainty = (
        -torch.sum(
            torch.multiply(
                torch.abs(max_uncertainty_value - predictions),
                torch.log(torch.abs(max_uncertainty_value - predictions)),
            ),
            (1, 2, 3),
        )
        .cpu()
        .numpy()
    )
    return uncertainty


def clean_duplicate_scans(
    uncertainties: List[Tuple[float, str]], items_to_label: int
) -> List[Tuple[float, str]]:
    """
    Cleans the list from duplicate scans if possible. If minimum number of samples can't be reached without
    duplicates, duplicates are kept.
    Args:
        uncertainties (List[Tuple[float, str]]): List with tuples of uncertainty value and case id.
        items_to_label (int): Number of items that should be selected for labeling.

    Returns:
        A cleaned list of tuples.
    """
    cleaned_uncertainties, scan_ids, duplicate_slides = [], [], []
    for value, case_id in uncertainties:
        if case_id.split("-")[0] not in scan_ids:
            scan_ids.append(case_id.split("-")[0])
            cleaned_uncertainties.append((value, case_id))
        else:
            duplicate_slides.append((value, case_id))
    if len(cleaned_uncertainties) < items_to_label:
        return [
            *cleaned_uncertainties,
            *duplicate_slides[: (items_to_label - len(uncertainties))],
        ]
    return cleaned_uncertainties
