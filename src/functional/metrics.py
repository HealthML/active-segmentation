"""
Module containing model evaluation metrics.

The metric implementations are based on the TorchMetrics framework. For instructions on how to implement custom metrics
with this framework, see https://torchmetrics.readthedocs.io/en/latest/pages/implement.html.
"""
import abc
from typing import List, Literal, Optional, Tuple, Union

import psutil
import torch
import torchmetrics

from .utils import (
    flatten_tensors,
    is_binary,
    reduce_metric,
    remove_padding,
    preprocess_metric_inputs,
)

# pylint: disable=too-many-lines


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    convert_to_one_hot: bool = True,
    epsilon: float = 0,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    reduction: Literal["mean", "min", "max", "none"] = "none",
) -> torch.Tensor:
    r"""
    Computes the Dice similarity coefficient (DSC) between a predicted segmentation mask and the target mask:

        :math:`DSC = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}`

    Using the `epsilon` parameter, Laplacian smoothing can be applied:

        :math:`DSC = \frac{2 \cdot TP + \lambda}{2 \cdot TP + FP + FN + \lambda}`

    This metric supports both single-label and multi-label segmentation tasks.

    Args:
        prediction (Tensor): The prediction tensor.
        target (Tensor): The target tensor.
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        epsilon (float, optional): Laplacian smoothing factor (default = 0).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (string): A method to reduce metric scores of multiple classes.
            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum

    Returns:
        Tensor: Dice similarity coefficient.

    Shape:
        - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
          | and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot encoding
          | (`C = number of classes`).
        - Target: Same shape and type as prediction.
        - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
    """

    prediction, target = preprocess_metric_inputs(
        prediction,
        target,
        num_classes,
        convert_to_one_hot=convert_to_one_hot,
        ignore_index=ignore_index,
    )

    flattened_prediction, flattened_target = flatten_tensors(
        prediction, target, include_background=include_background
    )

    intersection = (flattened_prediction * flattened_target).sum(dim=1)
    per_class_dice_score = (2.0 * intersection + epsilon) / (
        flattened_prediction.sum(dim=1) + flattened_target.sum(dim=1) + epsilon
    )

    return reduce_metric(per_class_dice_score, reduction)


def sensitivity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    convert_to_one_hot: bool = True,
    epsilon: float = 0,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    reduction: Literal["mean", "min", "max", "none"] = "none",
) -> torch.Tensor:
    r"""
    Computes the sensitivity from a predicted segmentation mask and the target mask:

        :math:`Sensitivity = \frac{TP}{TP + FN}`

    Using the `epsilon` parameter, Laplacian smoothing can be applied:

        :math:`Sensitivity = \frac{TP + \lambda}{TP + FN + \lambda}`

    This metric supports both single-label and multi-label segmentation tasks.

    Args:
        prediction (Tensor): The prediction tensor.
        target (Tensor): The target tensor.
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        epsilon (float, optional): Laplacian smoothing factor (default = 0).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (string): A method to reduce metric scores of multiple classes.
            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum

    Returns:
        Tensor: Sensitivity.

    Shape:
        - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
          | and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot encoding
          | (`C = number of classes`).
        - Target: Same shape and type as prediction.
        - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
    """

    prediction, target = preprocess_metric_inputs(
        prediction,
        target,
        num_classes,
        convert_to_one_hot=convert_to_one_hot,
        ignore_index=ignore_index,
    )

    flattened_prediction, flattened_target = flatten_tensors(
        prediction, target, include_background=include_background
    )

    true_positives = (flattened_prediction * flattened_target).sum(dim=1)
    true_positives_false_negatives = flattened_target.sum(dim=1)

    per_class_sensitivity = (true_positives + epsilon) / (
        true_positives_false_negatives + epsilon
    )

    return reduce_metric(per_class_sensitivity, reduction)


def specificity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    convert_to_one_hot: bool = True,
    epsilon: float = 0,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    reduction: Literal["mean", "min", "max", "none"] = "none",
) -> torch.Tensor:
    r"""
    Computes the specificity from a predicted segmentation mask and the target mask:

        :math:`Specificity = \frac{TN}{TN + FP}`

    Using the `epsilon` parameter, Laplacian smoothing can be applied:

        :math:`Specificity = \frac{TN + \lambda}{TN + FP + \lambda}`

    This metric supports both single-label and multi-label segmentation tasks.

    Args:
        prediction (Tensor): The prediction tensor.
        target (Tensor): The target tensor.
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        epsilon (float, optional): Laplacian smoothing factor (default = 0).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (string): A method to reduce metric scores of multiple classes.
            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum

    Returns:
        Tensor: Specificity.

    Shape:
        - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
          | and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot encoding
          | (`C = number of classes`).
        - Target: Same shape and type as prediction.
        - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
    """

    prediction, target = preprocess_metric_inputs(
        prediction,
        target,
        num_classes,
        convert_to_one_hot=convert_to_one_hot,
        ignore_index=ignore_index,
        ignore_value=1,
    )

    flattened_prediction, flattened_target = flatten_tensors(
        prediction, target, include_background=include_background
    )

    ones = torch.ones(flattened_prediction.shape, device=prediction.device)

    true_negatives = ((ones - flattened_prediction) * (ones - flattened_target)).sum(
        dim=1
    )
    true_negatives_false_positives = (ones - flattened_target).sum(dim=1)

    per_class_specificity = (true_negatives + epsilon) / (
        true_negatives_false_positives + epsilon
    )

    return reduce_metric(per_class_specificity, reduction)


def _distance_matrix(
    first_point_set: torch.Tensor, second_point_set: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes shortest Euclidean distance from all points in the first tensor to any point in the second tensor.

    Args:
        first_point_set (Tensor): A tensor representing a list of points in an Euclidean space (typically 2d or 3d)
        second_point_set (Tensor): A tensor representing a list of points in an Euclidean space (typically 2d or 3d).
    Returns:
        Tensor: Matrix containing the shortest Euclidean distances between all points in the first tensor and any
        point in the second tensor.
    Shape:
        first_point_set: :math:`(N, D)` where :math:`N` is the number of points in the first set and :math:`D` is the
            dimensionality of the Euclidean space.
        second_point_set: :math:`(M, D)` where :math:`M` is the number of points in the second set and :math:`D` is the
            dimensionality of the Euclidean space.
        Output: :math:`(N)` where :math:`N` is the number of points in the first set.
    """

    if len(first_point_set) == 0:
        return torch.as_tensor(float("nan"), device=first_point_set.device)

    if torch.cuda.is_available():
        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    else:
        free_memory = psutil.virtual_memory().available

    second_point_set_memory = (
        second_point_set.element_size() * second_point_set.nelement()
    )
    split_size = int(max(free_memory / second_point_set_memory, 1)) - 5

    minimum_distances = torch.zeros(len(first_point_set), device=first_point_set.device)

    first_point_sets = torch.split(first_point_set, split_size)

    for idx, first_point_set_split in enumerate(first_point_sets):
        distances = torch.cdist(first_point_set_split, second_point_set.float())
        minimum_distances_for_current_split, _ = distances.min(axis=-1)
        start_index = idx * split_size
        end_index = (idx + 1) * split_size
        minimum_distances[start_index:end_index] = minimum_distances_for_current_split

    return minimum_distances


def _compute_all_image_locations(shape: Tuple[int], device: Optional[str] = None):
    r"""
    Computes a tensor of points corresponding to all pixel locations of a 2d or a 3d image in Euclidean space.

    Args:
        shape (Tuple[int]): The shape of the image for which the pixel locations are to be computed.
        device (str, optional): Device as defined by PyTorch.
    Returns:
        A tensor of points corresponding to all pixel locations of the image.
    Shape:
        Output: :math:`(width \cdot height, 2)` for 2d images, :math:`(S \cdot width \cdot height, 3)`, where `S =
            number of slices`, for 3d images.
    """

    return torch.cartesian_prod(
        *[torch.arange(dim, device=device).float() for dim in shape]
    )


def _binary_erosion(input_image: torch.Tensor) -> torch.Tensor:
    r"""
    Applies an erosion filter to a tensor representing a binary 2d or 3d image.

    Args:
        input_image (Tensor): Binary image to be eroded.
    Returns:
        Tensor: Eroded imaged.
    Shape:
        Input Image: :math:`(height, width)` or :math:`(S, height, width)` where :math:`S = number of slices`.
        Output: Has the same dimensions as the input.
    """

    assert input_image.dim() in [
        2,
        3,
    ], "Input must be a two- or three-dimensional tensor"
    assert is_binary(input_image), "Input must be binary."

    if input_image.dim() == 2:
        max_pooling = torch.nn.MaxPool2d(
            3, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )
    elif input_image.dim() == 3:
        max_pooling = torch.nn.MaxPool3d(
            3, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )

    # to implement erosion filtering, max pooling with a 3x3 kernel is applied to the inverted image
    inverted_input = (
        torch.ones(input_image.shape, device=input_image.device) - input_image
    )

    # pad image with ones to maintain the input's dimensions
    if input_image.dim() == 2:
        inverted_input_padded = torch.nn.functional.pad(
            inverted_input, (1, 1, 1, 1), "constant", 1
        )
    else:
        inverted_input_padded = torch.nn.functional.pad(
            inverted_input, (1, 1, 1, 1, 1, 1), "constant", 1
        )
    # create class and batch dimension as this is required by the max pooling module
    inverted_input_padded = inverted_input_padded.unsqueeze(dim=0).unsqueeze(dim=0)

    # apply the max pooling and invert the result
    return torch.ones(input_image.shape, device=input_image.device) - max_pooling(
        inverted_input_padded
    ).squeeze(dim=0).squeeze(dim=0)


# pylint: disable=too-many-locals
def single_class_hausdorff_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalize: bool = False,
    percentile: float = 0.95,
    all_image_locations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Computes the Hausdorff distance between a predicted segmentation mask and the target mask.

    Note:
        In this method, the `prediction` tensor is considered as a segmentation mask of a single 2D or 3D image for a
        given class and thus the distances are calculated over all channels and dimensions.
        As this method is implemented using the scipy package, the returned value is not differentiable in PyTorch and
        can therefore not be used in loss functions.
    Args:
        prediction (Tensor): The predicted segmentation mask, where each value is in :math:`\{0, 1\}`.
        target (Tensor): The target segmentation mask, where each value is in :math:`\{0, 1\}`.
        normalize (bool, optional): Whether the Hausdorff distance should be normalized by dividing it by the diagonal
            distance.
        percentile (float, optional): Percentile for which the Hausdorff distance is to be calculated, must be in
            :math:`\[0, 1\]`.
        all_image_locations (Tensor, optional): A pre-computed tensor containing one point for each pixel in the
            prediction representing the pixel's location in 2d or 3d Euclidean space. Passing a pre-computed tensor to
            this parameter can be used to speed up computation.
    Returns:
        Tensor: Hausdorff distance.
    Shape:
        - | Prediction: Can have arbitrary dimensions. Typically :math:`(S, height, width)`, where
          | `S = number of slices`, or `(height, width)` for single image segmentation tasks.
        - Target: Must have the same dimensions as the prediction.
        - | All_image_locations: Must contain one element per-pixel in the prediction. Each element represents an
          | n-dimensional point. Typically :math:`(S \cdot height \cdot width, 3)`, where `S = number of slices`, or
          | :math:`(height \cdot width, 2)`
        - Output: Scalar.
    """

    # adapted code from
    # https://github.com/PiechaczekMyller/brats/blob/eb9f7eade1066dd12c90f6cef101b74c5e974bfa/brats/functional.py#L135

    assert prediction.device == target.device
    assert (
        prediction.shape == target.shape
    ), "Prediction and target must have the same dimensions."
    assert (
        prediction.dim() == 2 or prediction.dim() == 3
    ), "Prediction and target must have either two or three dimensions."
    assert is_binary(prediction), "Predictions must be binary."
    assert is_binary(target), "Target must be binary."

    if torch.count_nonzero(prediction) == 0 or torch.count_nonzero(target) == 0:
        return torch.as_tensor(float("nan"), device=prediction.device)

    if all_image_locations is None:
        all_image_locations = _compute_all_image_locations(
            prediction.shape, device=prediction.device
        )

    # to reduce computational effort, the distance calculations are only done for the border points of the predicted and
    # the target shape
    # to identify border points, binary erosion is used
    prediction_boundaries = torch.logical_xor(
        prediction, _binary_erosion(prediction)
    ).int()
    target_boundaries = torch.logical_xor(target, _binary_erosion(target)).int()

    flattened_prediction = prediction_boundaries.view(-1).float()
    flattened_target = target_boundaries.view(-1).float()

    # select those points that belong to the target segmentation mask
    target_mask = flattened_target.eq(1)
    target_locations = all_image_locations[target_mask, :]

    # select those points that belong to the predicted segmentation mask
    prediction_mask = flattened_prediction.eq(1)
    prediction_locations = all_image_locations[prediction_mask, :]

    # for each point in the predicted segmentation mask, compute the Euclidean distance to all points of the target
    # segmentation mask
    distances_to_target = _distance_matrix(prediction_locations, target_locations)

    # for each point in the target segmentation mask, compute the Euclidean distance to all points of the predicted
    # segmentation mask
    distances_to_prediction = _distance_matrix(target_locations, prediction_locations)

    distances = torch.cat((distances_to_target, distances_to_prediction), 0)

    if normalize:
        maximum_distance = torch.norm(
            torch.tensor(list(prediction.shape), dtype=torch.float) - 1
        )
        distances = distances / maximum_distance

    return torch.quantile(distances, q=percentile, keepdim=False).float()


# pylint: disable=too-many-arguments
def hausdorff_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    all_image_locations: Optional[torch.Tensor] = None,
    convert_to_one_hot: bool = True,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    normalize: bool = False,
    percentile: float = 0.95,
    reduction: Literal["mean", "min", "max", "none"] = "none",
) -> torch.Tensor:
    r"""
    Computes the Hausdorff distance between a predicted segmentation mask and the target mask.

    This metric supports both single-label and multi-label segmentation tasks.

    Args:
        prediction (Tensor): The prediction tensor.
        target (Tensor): The target tensor.
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        all_image_locations (Tensor, optional): A pre-computed tensor containing one point for each pixel in the
            prediction representing the pixel's location in 2d or 3d Euclidean space. Passing a pre-computed tensor to
            this parameter can be used to speed up computation.
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        normalize (bool, optional): Whether the Hausdorff distance should be normalized by dividing it by the diagonal
            distance.
        percentile (float, optional): Percentile for which the Hausdorff distance is to be calculated, must be in
            :math:`\[0, 1\]`.
        reduction (string): A method to reduce metric scores of multiple classes.
            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum

    Returns:
        Tensor: Hausdorff distance.

    Shape:
        - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
          | and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot encoding
          | (`C = number of classes`).
        - Target: Same shape and type as prediction.
        - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
    """

    # adapted code from
    # https://github.com/PiechaczekMyller/brats/blob/eb9f7eade1066dd12c90f6cef101b74c5e974bfa/brats/functional.py#L135

    prediction, target = remove_padding(
        prediction, target, ignore_index, convert_to_one_hot
    )
    prediction, target = preprocess_metric_inputs(
        prediction,
        target,
        num_classes,
        convert_to_one_hot=convert_to_one_hot,
        ignore_index=ignore_index,
    )

    if not include_background:
        num_classes = num_classes - 1
        # drop the channel of the background class
        prediction = prediction[1:]
        target = target[1:]

    per_class_hausdorff_distances = torch.ones(num_classes, device=prediction.device)

    for i in range(num_classes):
        per_class_hausdorff_distances[i] = single_class_hausdorff_distance(
            prediction[i],
            target[i],
            normalize=normalize,
            percentile=percentile,
            all_image_locations=all_image_locations,
        )
    return reduce_metric(per_class_hausdorff_distances, reduction)


class SegmentationMetric(torchmetrics.Metric, abc.ABC):
    """
    Base class for segmentation metrics.

    Args:
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        ignore_value (float, optional): Value that should be inserted at the positions where the target is equal to
            `ignore_index`. Defaults to 0.
        reduction (string, optional): A method to reduce metric scores of multiple classes.

            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum
    """

    def __init__(
        self,
        num_classes: int,
        convert_to_one_hot: bool = True,
        ignore_index: Optional[bool] = None,
        ignore_value: float = 0,
        include_background: bool = True,
        reduction: Literal["mean", "min", "max", "none"] = "none",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.convert_to_one_hot = convert_to_one_hot
        self.ignore_index = ignore_index
        self.ignore_value = ignore_value
        self.include_background = include_background
        if reduction not in ["mean", "min", "max", "none"]:
            raise ValueError("Invalid reduction method.")
        self.reduction = reduction

    def _flatten_tensors(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Reshapes and flattens prediction and target tensors except for the first dimension (class dimension).

        Args:
            prediction (Tensor): The prediction tensor (either label-encoded, one-hot encoded or multi-hot encoded).
            target (Tensor): The target tensor (either label-encoded, one-hot encoded or multi-hot encoded).

        Returns:
            Tuple[Tensor]: Flattened prediction and target tensors (one-hot or multi-hot encoded).

        Shape:
            - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label
              | encoding and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or
              | multi-hot encoding (`C = number of classes`).
            - Target: Same shape and type as prediction.
            - | Output: :math:`(C, X * Y * ...)` where each element is in :math:`\{0, 1\}` indicating the absence /
              | presence of the respective class (one-hot / multi-hot encoding).
        """

        return flatten_tensors(
            prediction,
            target,
            include_background=self.include_background,
        )

    def _preprocess_inputs(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method implements three preprocessing steps that are needed for most segmentation metrics:

        1. Validation of input shape and type
        2. Conversion from label encoding to one-hot encoding if necessary
        3. Mapping of pixels/voxels labeled with the :attr:`ignore_index` to true negatives or true positives

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Returns:
            Tuple[Tensor, Tensor]: The preprocessed prediction and target tensors.
        """

        return preprocess_metric_inputs(
            prediction,
            target,
            self.num_classes,
            convert_to_one_hot=self.convert_to_one_hot,
            ignore_index=self.ignore_index,
            ignore_value=self.ignore_value,
        )


class DiceScore(SegmentationMetric):
    """
    Computes the Dice similarity coefficient (DSC). Can be used for 3D images whose slices are scattered over multiple
    batches.

    Args:
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = 0).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (string, optional): A method to reduce metric scores of multiple classes.

            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum
    """

    def __init__(
        self,
        num_classes: int,
        convert_to_one_hot: bool = True,
        epsilon: float = 0,
        ignore_index: Optional[bool] = None,
        include_background: bool = True,
        reduction: Literal["none", "mean", "min", "max"] = "none",
    ):
        super().__init__(
            num_classes,
            convert_to_one_hot=convert_to_one_hot,
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
        )
        self.epsilon = epsilon
        num_classes = num_classes if self.include_background else num_classes - 1
        self.numerator = torch.zeros(num_classes)
        self.denominator = torch.zeros(num_classes)
        self.add_state("numerator", torch.zeros(num_classes))
        self.add_state("denominator", torch.zeros(num_classes))

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        r"""
        Updates metric using the provided prediction.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Shape:
            - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label
              | encoding and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or
              | multi-hot encoding (`C = number of classes`).
            - Target: Same shape and type as prediction.
        """

        prediction, target = self._preprocess_inputs(prediction, target)

        flattened_prediction, flattened_target = self._flatten_tensors(
            prediction, target
        )

        self.numerator += (flattened_prediction * flattened_target).sum(dim=1)
        self.denominator += flattened_prediction.sum(dim=1) + flattened_target.sum(
            dim=1
        )

    def compute(self) -> torch.Tensor:
        """
        Computes the DSC  over all slices that were registered using the `update` method.

        Returns:
            Tensor: Dice similarity coefficient.

        Shape:
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
        """

        per_class_dice_score = (2.0 * self.numerator + self.epsilon) / (
            self.denominator + self.epsilon
        )

        return reduce_metric(per_class_dice_score, self.reduction)


class Sensitivity(SegmentationMetric):
    """
    Computes the sensitivity. Can be used for 3D images whose slices are scattered over multiple batches.

    Args:
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = 0).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (string, optional): A method to reduce metric scores of multiple classes.

            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum
    """

    def __init__(
        self,
        num_classes: int,
        convert_to_one_hot: bool = True,
        epsilon: float = 0,
        ignore_index: Optional[bool] = None,
        include_background: bool = True,
        reduction: Literal["none", "mean", "min", "max"] = "none",
    ):
        super().__init__(
            num_classes,
            convert_to_one_hot=convert_to_one_hot,
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
        )
        self.epsilon = epsilon
        num_classes = num_classes if self.include_background else num_classes - 1
        self.true_positives = torch.zeros(num_classes)
        self.true_positives_false_negatives = torch.zeros(num_classes)
        self.add_state("true_positives", torch.zeros(num_classes))
        self.add_state("true_positives_false_negatives", torch.zeros(num_classes))

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        r"""
        Updates metric using the provided prediction.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Shape:
            - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label
              | encoding and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or
              | multi-hot encoding (`C = number of classes`).
            - Target: Same shape and type as prediction.
        """

        prediction, target = self._preprocess_inputs(prediction, target)

        flattened_prediction, flattened_target = self._flatten_tensors(
            prediction, target
        )

        self.true_positives += (flattened_prediction * flattened_target).sum(dim=1)
        self.true_positives_false_negatives += flattened_target.sum(dim=1)

    def compute(self) -> torch.Tensor:
        """
        Computes the sensitivity  over all slices that were registered using the `update` method.

        Returns:
            Tensor: Sensitivity.

        Shape:
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
        """

        per_class_sensitivity = (self.true_positives + self.epsilon) / (
            self.true_positives_false_negatives + self.epsilon
        )

        return reduce_metric(per_class_sensitivity, self.reduction)


class Specificity(SegmentationMetric):
    """
    Computes the specificity. Can be used for 3D images whose slices are scattered over multiple batches.

    Args:
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = 0).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (string, optional): A method to reduce metric scores of multiple classes (default = `"none"`).

            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum
    """

    def __init__(
        self,
        num_classes: int,
        convert_to_one_hot: bool = True,
        epsilon: float = 0,
        ignore_index: Optional[bool] = None,
        include_background: bool = True,
        reduction: Literal["none", "mean", "min", "max"] = "none",
    ):
        super().__init__(
            num_classes,
            convert_to_one_hot=convert_to_one_hot,
            ignore_index=ignore_index,
            ignore_value=1,
            include_background=include_background,
            reduction=reduction,
        )
        self.epsilon = epsilon
        num_classes = num_classes if self.include_background else num_classes - 1
        self.true_negatives = torch.zeros(num_classes)
        self.true_negatives_false_positives = torch.zeros(num_classes)
        self.add_state("true_negatives", torch.zeros(num_classes))
        self.add_state("true_negatives_false_positives", torch.zeros(num_classes))

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        r"""
        Updates metric using the provided prediction.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Shape:
            - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label
              | encoding and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or
              | multi-hot encoding (`C = number of classes`).
            - Target: Same shape and type as prediction.
        """

        prediction, target = self._preprocess_inputs(prediction, target)

        flattened_prediction, flattened_target = self._flatten_tensors(
            prediction, target
        )

        ones = torch.ones(flattened_prediction.shape, device=prediction.device)

        self.true_negatives += (
            (ones - flattened_prediction) * (ones - flattened_target)
        ).sum(dim=1)
        self.true_negatives_false_positives += (ones - flattened_target).sum(dim=1)

    def compute(self) -> torch.Tensor:
        """
        Computes the sensitivity  over all slices that were registered using the `update` method.

        Returns:
            Tensor: Sensitivity.

        Shape:
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
        """

        per_class_sensitivity = (self.true_negatives + self.epsilon) / (
            self.true_negatives_false_positives + self.epsilon
        )

        return reduce_metric(per_class_sensitivity, self.reduction)


class HausdorffDistance(SegmentationMetric):
    r"""
    Computes the Hausdorff distance. Can be used for 3D images whose slices are scattered over multiple batches.

    Args:
        num_classes (int): Number of classes (for single-label segmentation tasks including the background class).
        slices_per_image (int): Number of slices per 3d image.
        convert_to_one_hot (bool, optional): Determines if data is label encoded and needs to be converted to one-hot
            encoding or not (default = `True`).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric.
            Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        normalize (bool, optional): Whether the Hausdorff distance should be normalized by dividing it by the diagonal
            distance.
        percentile (float, optional): Percentile for which the Hausdorff distance is to be calculated, must be in
            :math:`\[0, 1\]`.
        reduction (string, optional): A method to reduce metric scores of multiple classes (default = `"none"`).

            - ``"none"``: no reduction will be applied (default)
            - ``"mean"``: takes the mean
            - ``"min"``: takes the minimum
            - ``"max"``: takes the maximum
    """

    def __init__(
        self,
        num_classes: int,
        slices_per_image: int,
        convert_to_one_hot: bool = True,
        ignore_index: Optional[bool] = None,
        include_background: bool = True,
        normalize: bool = False,
        percentile: float = 0.95,
        reduction: Literal["none", "mean", "min", "max"] = "none",
    ):
        super().__init__(
            num_classes,
            convert_to_one_hot=convert_to_one_hot,
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
        )
        num_classes = num_classes if self.include_background else num_classes - 1

        self.normalize = normalize
        self.percentile = percentile
        self.predictions = torch.empty(slices_per_image)
        self.targets = torch.empty(slices_per_image)
        self.add_state("predictions", torch.empty(slices_per_image))
        self.add_state("targets", torch.empty(slices_per_image))
        self.hausdorff_distance = torch.zeros(num_classes)
        self.add_state("hausdorff_distance", torch.zeros(num_classes))
        self.all_image_locations = None
        self.hausdorff_distance_cached = False
        self.slices_per_image = slices_per_image
        self.number_of_slices = 0
        self.collected_slice_ids = set()

    def reset(self) -> None:
        """
        Resets the metric state.
        """

        super().reset()
        self.all_image_locations = None
        self.hausdorff_distance_cached = False
        self.number_of_slices = 0
        self.collected_slice_ids = set()

    # pylint: disable=arguments-differ
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        slice_ids: Union[int, List[int]] = None,
    ) -> None:
        r"""
        Updates metric using the provided prediction.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.
            slice_ids (Union[int, List[int]]): Indices of the image slices represented by `prediction` and `target`.
                Must be provided if `target` is a single slice or a subset of slices taken from a 3d image.

        Shape:
            - | Prediction: :math:`(X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label
              | encoding and :math:`(C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or
              | multi-hot encoding (`C = number of classes`).
            - Target: Same shape and type as prediction.
        """

        prediction, target = remove_padding(
            prediction, target, self.ignore_index, self.convert_to_one_hot
        )
        prediction, target = self._preprocess_inputs(prediction, target)

        # the prediction and target tensor include an additional class dimension
        dimensionality = 2 if prediction.dim() == 3 else 3

        added_slices = 1 if dimensionality == 2 else prediction.shape[1]

        if slice_ids is None:
            assert self.slices_per_image == 1 or (
                dimensionality == 3 and self.slices_per_image == prediction.shape[1]
            ), (
                "The `slice_ids` parameter must be provided when `prediction` and `target` are a single slice or a "
                "subset of slices taken from a 3d image."
            )

            self.predictions = prediction
            self.targets = target
        else:
            if isinstance(slice_ids, int):
                slice_ids = [slice_ids]

            assert (
                len(slice_ids) == added_slices
            ), "The length of `slice_ids` must be equal to the number of slices in `prediction` and `target`."

            assert (
                len(self.collected_slice_ids.intersection(set(slice_ids))) == 0
            ), "Tried to update the Hausdorff distance with a slice that was already added."

            # in this case we just collect all slices of the image since the Hausdorff distance needs to be computed
            # over all slices
            # note that this will be memory-intensive if this is done for multiple 3d images in parallel

            if self.number_of_slices == 0:
                # create tensors that have the shape of the original 3d images

                shape_of_full_image = (
                    prediction.shape[0],
                    self.slices_per_image,
                    prediction.shape[-2],
                    prediction.shape[-1],
                )

                self.predictions = torch.zeros(*shape_of_full_image, dtype=torch.int)
                self.targets = torch.zeros(*shape_of_full_image, dtype=torch.int)

            self.collected_slice_ids.update(slice_ids)

            for idx, slice_id in enumerate(slice_ids):
                self.predictions[:, slice_id] = (
                    prediction[:, idx] if dimensionality == 3 else prediction
                )
                self.targets[:, slice_id] = (
                    target[:, idx] if dimensionality == 3 else target
                )

        self.hausdorff_distance_cached = False
        self.number_of_slices += added_slices

        if self.number_of_slices == self.slices_per_image:
            self.compute()
        assert self.number_of_slices <= self.slices_per_image

    def compute(self) -> torch.Tensor:
        """
        Computes the Hausdorff distance over all slices that were registered using the `update` method as the maximum
        per slice-distance.

        Returns:
            Tensor: Hausdorff distance.

        Shape:
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(C)`. Otherwise, scalar.
        """

        if self.hausdorff_distance_cached:
            assert (
                self.number_of_slices == 0
            ), "A cached value for the Hausdorff distance is used even though there were slices added afterwards."
            return self.hausdorff_distance

        assert self.number_of_slices == self.slices_per_image, (
            "The compute method was called on the Hausdorff distance module before all slices of the image were "
            "provided."
        )

        hausdorff_dist = hausdorff_distance(
            self.predictions,
            self.targets,
            self.num_classes,
            convert_to_one_hot=False,
            ignore_index=None,
            include_background=self.include_background,
            normalize=self.normalize,
            percentile=self.percentile,
        )

        self.hausdorff_distance = hausdorff_dist
        self.hausdorff_distance_cached = True

        # free memory
        del self.predictions
        del self.targets
        self.predictions = torch.empty(self.slices_per_image)
        self.targets = torch.empty(self.slices_per_image)
        self.number_of_slices = 0

        return reduce_metric(hausdorff_dist, self.reduction)
