"""Utilities for metric and loss computations."""

from typing import Literal, Tuple

import torch


def flatten_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    convert_to_one_hot: bool = True,
    include_background: bool = True,
) -> Tuple[torch.Tensor]:
    r"""
    Reshapes and flattens prediction and target tensors except for the first dimension (class dimension).

    Args:
        prediction (Tensor): The prediction tensor (either label-encoded, one-hot encoded or multi-hot encoded).
        target (Tensor): The target tensor (either label-encoded, one-hot encoded or multi-hot encoded).
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).

    Returns:
        Tuple[Tensor]: Flattened prediction and target tensors (one-hot or multi-hot encoded).

    Shape:
        - Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
            and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
            encoding (`C = number of classes`).
        - Target: Must have same shape and type as prediction.
        - Output: :math:`(C, X * Y * ...)` where each element is in :math:`\{0, 1\}` indicating the absence /
            presence of the respective class (one-hot or multi-hot encoding).
    """
    assert (
        prediction.shape == target.shape
    ), "Prediction and target need to have the same shape."
    assert (
        prediction.dtype == torch.int
    ), "Prediction and target both need to be of integer type."

    if convert_to_one_hot is True:
        # for single-label segmentation tasks, prediction and target are label encoded
        # they have the shape (X, Y, ...)

        assert prediction.dim() in [
            2,
            3,
        ], "Prediction and target need to be either two- or three-dimensional if `convert_to_one_hot` is True."

        # convert label encoding into one-hot encoding
        prediction = one_hot_encode(prediction, num_classes)
        target = one_hot_encode(target, num_classes)

    else:
        # for single-label segmentation tasks, prediction and target are one-hot or multi-hot encoded
        # they have the shape (C, X, Y, ...)

        assert _is_binary(
            prediction
        ), "Prediction needs to be binary if `convert_to_one_hot` is False."
        assert _is_binary(
            target
        ), "Target needs to be binary if `convert_to_one_hot` is False."

        assert prediction.dim() in [
            3,
            4,
        ], "Prediction and target need to be either three- or four-dimensional if `convert_to_one_hot` is False."

    if not include_background:
        # drop the channel of the background class
        prediction = prediction[1:]
        target = target[1:]

    # flatten tensors except for the first channel (class dimension)
    flattened_prediction = prediction.view(num_classes, -1)
    flattened_target = target.view(num_classes, -1)

    return flattened_prediction.float(), flattened_target.float()


def is_binary(tensor_to_check: torch.Tensor) -> bool:
    """
    Checks whether the input contains only zeros and ones.

    Args:
        input (Tensor): tensor to check.
    Returns:
        bool: True if contains only zeros and ones, False otherwise.
    """

    return torch.equal(
        tensor_to_check, tensor_to_check.bool().to(dtype=tensor_to_check.dtype)
    )


def one_hot_encode(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    r"""
    Comverts a label encoded tensor to a one-hot encoded tensor.

    Args:
        tensor (Tensor): Label encoded tensor that is to be converted to one-hot encoding.
        num_classes (int): Number of classes.

    Returns:
        Tensor: One-hot encoded tensor.

    Shape:
        - Tensor: :math:`(N, X, Y, ...)` where each element represents a class index of integer type and `N = batch
            size`.
        - Output: :math:`(N, C, X, Y, ...)` where each element represent a binary class label.
    """

    tensor_one_hot = torch.nn.functional.one_hot(tensor.long(), num_classes)

    # one_hot outputs a tensor of shape (N, X, Y, ..., C)
    # this tensor is converted to a tensor of shape (N, C, X, Y, ...)
    return tensor_one_hot.permute(
        (0, tensor_one_hot.ndim - 1, *range(1, tensor_one_hot.ndim - 1))
    )


def reduce_metric(
    metric: torch.Tensor, reduction: Literal["mean", "min", "max", "none"]
) -> torch.Tensor:
    r"""
    Aggregates the metric values of the different classes.

    Args:
        metric (Tensor): Metrics to be aggregated.
        reduction (string): A method to reduce metric scores of multiple classes.
            - ``"none"``: no reduction will be applied
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum

    Returns:
        Tensor: Aggregated metric value.

    Shape:
        - Metric: :math:`(C)`, where `C = number of classes`.
        - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
    """

    if reduction == "mean":
        return metric.mean()
    if reduction == "min":
        return metric.min()
    if reduction == "max":
        return metric.max()
    if reduction == "none":
        return metric
    raise ValueError("Invalid reduction method.")
