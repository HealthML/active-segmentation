"""Utilities for metric and loss computations."""

from typing import Literal, Optional, Tuple, Union

import torch


def flatten_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    include_background: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Reshapes and flattens prediction and target tensors except for the first dimension (class dimension).

    Args:
        prediction (Tensor): The prediction tensor (one-hot encoded or multi-hot encoded).
        target (Tensor): The target tensor (one-hot encoded or multi-hot encoded).
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).

    Returns:
        Tuple[Tensor]: Flattened prediction and target tensors (one-hot or multi-hot encoded).

    Shape:
        - Prediction: :math:`(C, X, Y, ...)`, where `C = number of classes` and each value is in :math:`\{0, 1\}`.
        - Target: Must have same shape and type as prediction.
        - Output: :math:`(C, X * Y * ...)` where each element is in :math:`\{0, 1\}` indicating the absence /
            presence of the respective class (one-hot or multi-hot encoding).
    """

    if not include_background:
        # drop the channel of the background class
        prediction = prediction[1:]
        target = target[1:]

    # flatten tensors except for the first channel (class dimension)
    flattened_prediction = prediction.contiguous().view(prediction.shape[0], -1)
    flattened_target = target.contiguous().view(target.shape[0], -1)

    return flattened_prediction.float(), flattened_target.float()


def is_binary(tensor_to_check: torch.Tensor) -> bool:
    """
    Checks whether the input contains only zeros and ones.

    Args:
        tensor_to_check (Tensor): tensor to check.
    Returns:
        bool: True if contains only zeros and ones, False otherwise.
    """

    return torch.equal(
        tensor_to_check, tensor_to_check.bool().to(dtype=tensor_to_check.dtype)
    )


def mask_tensor(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: Optional[int] = None,
    mask_value: float = 0,
) -> torch.Tensor:
    r"""
    Replaces the tensor's values in the positions where the mask is equal to `ignore_index` with `mask_value`.

    Args:
        tensor (Tensor): A tensor in which is to be masked.
        mask (Tensor): A mask tensor containing the :attr:`ignore_index` at the positions to be masked.
        ignore_index (int, optional): Label index indicating the positions to be masked.
        mask_value (float, optional): Value that should be inserted at the masked positions. Defaults to 0.

    Returns:
        Tensor: Masked tensor.

    Shape:
        Tensor: :math:`(N, C, X, Y, ...)` or :math:`(C, X, Y, ...)`.
        Mask: :math:`(N, 1, X, Y, ...)` / :math:`(N, C, X, Y, ...)` or :math:`(1, X, Y, ...)` / :math:`(C, X, Y, ...)`.
        Output: Same shape as input.
    """

    if ignore_index is not None:
        # set positions where tensor is equal to ignore_index to mask_value
        tensor = tensor.clone()
        tensor = (mask != ignore_index) * tensor + (mask == ignore_index) * mask_value

    return tensor


def one_hot_encode(
    tensor: torch.Tensor, dim: int, num_classes: int, ignore_index: Optional[int] = None
) -> torch.Tensor:
    r"""
    Converts a label encoded tensor to a one-hot encoded tensor.

    Args:
        tensor (Tensor): Label encoded tensor that is to be converted to one-hot encoding.
        dim (int): Dimensionality of the input. Either 2 or 3.
        num_classes (int): Number of classes (excluding the class labeled with :attr:`ignore_label`).
        ignore_index (int, optional): Class value for which no one-hot encoded channel should be created in the output.

    Returns:
        Tensor: One-hot encoded tensor.

    Shape:
        - Tensor: :math:`(N, X, Y, ...)` or :math:`(X, Y, ...)` where each element represents a class index of integer
            type and `N = batch size`.
        - Output: :math:`(N, C, X, Y, ...)` or :math:`(C, X, Y, ...)` where each element represent a binary class label
            and :math:`C` is the number of classes (excluding the ignored class labeled with :attr:`ignore_label`).
    """

    tensor = tensor.clone()

    if ignore_index is not None:
        # shift labels since `torch.nn.functional.one_hot` only accepts positive labels
        tensor[tensor == ignore_index] = -1
        tensor += 1
        num_classes += 1

    tensor_one_hot = torch.nn.functional.one_hot(tensor.long(), num_classes).int()

    if ignore_index is not None:
        # drop ignored channel
        tensor_one_hot = tensor_one_hot[..., 1:]

    # one_hot outputs a tensor of shape (N, X, Y, ..., C) or (X, Y, ..., C)
    # this tensor is converted to a tensor of shape (N, C, X, Y, ...) or (C, X, Y, ...)
    if tensor.dim() == dim + 1:
        # tensor has a batch dimension
        return torch.moveaxis(tensor_one_hot, tensor_one_hot.ndim - 1, 1)

    # tensor has no batch dimension
    return torch.moveaxis(tensor_one_hot, tensor_one_hot.ndim - 1, 0)


def remove_padding(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: Union[int, None],
    is_label_encoded: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Removes padding from prediction and target tensor. For this purpose, the areas where the target tensor is
    equal to :attr:`ignore_index` are removed from both tensors. It is assumed that whole rows or columns are padded
    always.

    Args:
        prediction (Tensor): A prediction tensor (label-encoded, one-hot encoded, or multi-hot encoded).
        target (Tensor): A target tensor (with same encoding as prediction tensor).
        ignore_index (int): Specifies the target value that is used as label for padded areas.
        is_label_encoded (bool): Whether the input data are label encoded or one-hot / multi-hot encoded.

    Returns:
        Tuple[Tensor, Tensor]: Prediction and target tensors without padding areas.

    Shape:
        - Prediction: :math:`(X, Y, ...)` in case of label encoding and :math:`(C, X, Y, ...)`, in case of one-hot
            or multi-hot encoding (`C = number of classes`).
        - Target: Same shape as prediction.
        - Output: :math:`(X - P_x, Y - P_y, ...)` in case of label encoding and :math:`(C, X - P_x, Y - P_y, ...)`,
            in case of one-hot or multi-hot encoding (`C = number of classes`, `P_x = padding width on x-axis`).
    """

    if ignore_index is None:
        return prediction, target

    assert (
        prediction.shape == target.shape
    ), "Prediction and target must have the same shape"

    if is_label_encoded:
        first_spatial_dim = 0
        dimensionality = target.dim()
    else:
        first_spatial_dim = 1
        dimensionality = target.dim() - 1

    # for 3d images, first slices that only contain padding are removed
    if dimensionality == 3:
        is_padding = (
            target == ignore_index if is_label_encoded else target[0] == ignore_index
        )

        if not is_label_encoded:
            assert torch.equal(
                (target[0] == ignore_index).int() * len(target),
                (target == ignore_index).sum(dim=0).int(),
            ), "All class channels have the same padding size"

        is_padding_slice = is_padding.flatten(start_dim=-2).all(dim=-1)

        all_indices = torch.arange(is_padding.shape[0], device=prediction.device)
        indices_to_keep = torch.masked_select(all_indices, ~is_padding_slice)

        target = target.index_select(first_spatial_dim, indices_to_keep)
        prediction = prediction.index_select(first_spatial_dim, indices_to_keep)

    # afterwards single padded rows and columns are removed

    # the first 2d slice is selected and it is assumed that all other slices have the same padding size
    all_target_slices = target.view(-1, *target.shape[-2:])
    first_slice = all_target_slices[0]

    if target.dim() > 2:
        assert torch.equal(
            (first_slice == ignore_index).int() * len(all_target_slices),
            (all_target_slices == ignore_index).sum(dim=0).int(),
        ), "All slices have the same padding size."

    for dim in [0, 1]:
        is_padding = (first_slice == ignore_index).all(dim=dim)
        all_indices = torch.arange(len(is_padding), device=prediction.device)
        indices_to_keep = torch.masked_select(all_indices, ~is_padding)

        target = target.index_select(-2 + dim, indices_to_keep)
        prediction = prediction.index_select(-2 + dim, indices_to_keep)

    assert (
        prediction != ignore_index
    ).all(), "Prediction does not contain padded areas after padding removal."
    assert (
        target != ignore_index
    ).all(), "Target does not contain padded areas after padding removal."

    return prediction, target


def validate_metric_inputs(
    prediction: torch.Tensor, target: torch.Tensor, convert_to_one_hot: bool
) -> None:
    """
    Validates the inputs for segmentation metric computations:
        - Checks that prediction and target are both on the same device.
        - Checks that prediction and target have the correct shape and type.

    Args:
        prediction (Tensor): The prediction tensor to be validated.
        target (Tensor): The target tensor to be validated.
        convert_to_one_hot (bool, optional): Determines if data is label encoded and is intended to be converted to
            one-hot encoding or not (default = `True`).

    Raises:
        AssertionError: if input tensors violate any of the validation criteria.
    """

    assert prediction.device == target.device

    assert (
        prediction.shape == target.shape
    ), "Prediction and target need to have the same shape."

    assert prediction.dtype in [
        torch.int,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ], "Prediction has to be of integer type."
    assert target.dtype in [
        torch.int,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ], "Target has to be of integer type."

    if convert_to_one_hot is True:
        # for single-label segmentation tasks, prediction and target are expected be label encoded
        # they are expected to have the shape (X, Y, ...)

        assert prediction.dim() in [
            2,
            3,
        ], "Prediction and target need to be either two- or three-dimensional if `convert_to_one_hot` is True."
    else:
        # for single-label segmentation tasks, prediction and target are expected to be one-hot or multi-hot encoded
        # they are expected to have the shape (C, X, Y, ...)

        assert is_binary(
            prediction
        ), "Prediction needs to be binary if `convert_to_one_hot` is False."
        assert is_binary(
            target * (target >= 0)  # target excluding ignore index
        ), "Target needs to be binary if `convert_to_one_hot` is False."

        assert prediction.dim() in [
            3,
            4,
        ], "Prediction and target need to be either three- or four-dimensional if `convert_to_one_hot` is False."


def preprocess_metric_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    convert_to_one_hot: bool = True,
    ignore_index: Optional[int] = None,
    ignore_value: float = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This method implements preprocessing steps that are needed for most segmentation metrics:

    1. Validation of input shape and type
    2. Conversion from label encoding to one-hot encoding if necessary
    3. Mapping of pixels/voxels labeled with the :attr:`ignore_index` to true negatives or true positives

    Args:
        prediction (Tensor): The prediction tensor to be preprocessed.
        target (Tensor): The target tensor to be preprocessed.
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        ignore_value (float, optional): Value that should be inserted at the positions where the target is equal to
            `ignore_index`. Defaults to 0.

    Returns:
        Tuple[Tensor, Tensor]: The preprocessed prediction and target tensors.
    """

    validate_metric_inputs(prediction, target, convert_to_one_hot)

    # during one-hot encoding the ignore index is removed, therefore the original target including the ignore index
    # is copied
    target_including_ignore_index = target
    target = target.clone()

    if convert_to_one_hot:
        prediction = one_hot_encode(
            prediction, prediction.dim(), num_classes, ignore_index=ignore_index
        )
        target = one_hot_encode(
            target, target.dim(), num_classes, ignore_index=ignore_index
        )

    # map values where the target is set to `ignore_index` to zero

    prediction = mask_tensor(
        prediction,
        target_including_ignore_index,
        ignore_index=ignore_index,
        mask_value=ignore_value,
    )

    target = mask_tensor(
        target,
        target_including_ignore_index,
        ignore_index=ignore_index,
        mask_value=ignore_value,
    )

    return prediction, target


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
