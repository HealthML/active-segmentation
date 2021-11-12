""" Module containing segmentation metrics """

import torch


def dsc(prediction: torch.Tensor, target: torch.Tensor, smoothing=1) -> torch.Tensor:
    """
    Computes Dice similarity coefficient (DSC) for two binary segmentation masks.

    :param prediction: First binary segmentation mask.
    :param target: Second binary segmentation mask.
    :param smoothing: Smoothing factor.
    :return: Dice similarity coefficient as 1-element tensor.
    """

    flattened_target = torch.flatten(target).float()
    flattened_prediction = torch.flatten(prediction).float()
    intersection = (flattened_target * flattened_prediction).sum()
    score = (2.0 * intersection + smoothing) / (
        (flattened_target * flattened_target).sum()
        + (flattened_prediction * flattened_prediction).sum()
        + smoothing
    )

    return score


def recall(prediction: torch.Tensor, target: torch.Tensor, smoothing=1) -> torch.Tensor:
    """
    Computes number of false positive pixels for a predicted binary segmentation mask.

    :param prediction: Predicted binary segmentation mask.
    :param target: Target binary segmentation mask.
    :param smoothing: Smoothing factor.
    :return: Number of false positives as 1-element tensor.
    """

    flattened_target = torch.flatten(target).float()
    flattened_prediction = torch.flatten(prediction).float()

    true_positives = flattened_target.sum()
    intersection = (flattened_target * flattened_prediction).sum()
    return (intersection + smoothing) / (true_positives + smoothing)
