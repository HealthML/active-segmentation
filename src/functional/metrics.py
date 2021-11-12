import torch


def dsc(input: torch.Tensor, target: torch.Tensor, smoothing=1) -> torch.Tensor:
    """
    Computes Dice similarity coefficient (DSC) for two binary segmentation masks.

    :param input: First binary segmentation mask.
    :param target: Second binary segmentation mask.
    :param smoothing: Smoothing factor.
    :return: Dice similarity coefficient as 1-element tensor.
    """

    flattened_target = torch.flatten(target).float()
    flattened_input = torch.flatten(input).float()
    intersection = (flattened_target * flattened_input).sum()
    score = (2.0 * intersection + smoothing) / (
        (flattened_target * flattened_target).sum()
        + (flattened_input * flattened_input).sum()
        + smoothing
    )

    return score


def recall(input: torch.Tensor, target: torch.Tensor, smoothing=1) -> torch.Tensor:
    """
    Computes number of false positive pixels for a predicted binary segmentation mask.

    :param input: Predicted binary segmentation mask.
    :param target: Target binary segmentation mask.
    :param smoothing: Smoothing factor.
    :return: Number of false positives as 1-element tensor.
    """

    flattened_target = torch.flatten(target).float()
    flattened_input = torch.flatten(input).float()

    true_positives = flattened_target.sum()
    intersection = (flattened_target * flattened_input).sum()
    return (intersection + smoothing) / (true_positives + smoothing)
