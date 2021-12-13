"""
Module containing model evaluation metrics.

The metric implementations are based on the TorchMetrics framework. For instructions on how to implement custom metrics
with this framework, see https://torchmetrics.readthedocs.io/en/latest/pages/implement.html.
"""

from typing import Optional, Tuple

import torch
import torchmetrics


def _is_binary(tensor_to_check: torch.Tensor) -> bool:
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


def dice_score(
    prediction: torch.Tensor, target: torch.Tensor, smoothing: float = 0
) -> torch.Tensor:
    r"""
    Computes the Dice similarity coefficient (DSC) between a predicted segmentation mask and the target mask:

        :math:`DSC = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}`

    Using the `smoothing` parameter, Laplacian smoothing can be applied:

        :math:`DSC = \frac{2 \cdot TP + \lambda}{2 \cdot TP + FP + FN + \lambda}`

    Note:
        In this method, the `prediction` tensor is considered as a segmentation mask of a single 2D or 3D image for a
        given class and thus it aggregates over :math:`TP`, :math:`FP`, and :math:`FN` over all channels and dimensions.

    Args:
        prediction (Tensor): The predicted segmentation mask, where each value is in :math:`[0, 1]`.
        target (Tensor): The target segmentation mask, where each value is in :math:`\{0, 1\}`.
        smoothing (float, optional): Laplacian smoothing factor.

    Returns:
        Tensor: Dice similarity coefficient.

    Shape:
        - Prediction: Can have arbitrary dimensions. Typically :math:`(S, height, width)`, where `S = number of slices`,
          or `(height, width)` for single image segmentation tasks.
        - Target: Must have the same dimensions as the prediction.
        - Output: Scalar.
    """

    assert prediction.device == target.device
    assert prediction.shape == target.shape

    flattened_prediction = prediction.view(-1).float()
    flattened_target = target.view(-1).float()

    intersection = (flattened_prediction * flattened_target).sum()
    score = (2.0 * intersection + smoothing) / (
        flattened_prediction.sum() + flattened_target.sum() + smoothing
    )

    return score


def sensitivity(
    prediction: torch.Tensor, target: torch.Tensor, smoothing: float = 0
) -> torch.Tensor:
    r"""
    Computes the sensitivity from a predicted segmentation mask and the target mask:

        :math:`Sensitivity = \frac{TP}{TP + FN}`

    Using the `smoothing` parameter, Laplacian smoothing can be applied:

        :math:`Sensitivity = \frac{TP + \lambda}{TP + FN + \lambda}`

    Note:
        In this method, the `prediction` tensor is considered as a segmentation mask of a single 2D or 3D image for a
        given class and thus it aggregates over :math:`TP`, and :math:`FN` over all channels and dimensions.

    Args:
        prediction (Tensor): The predicted segmentation mask, where each value is in :math:`[0, 1]`.
        target (Tensor): The target segmentation mask, where each value is in :math:`\{0, 1\}`.
        smoothing (float, optional): Laplacian smoothing factor.

    Returns:
        Tensor: Sensitivity.

    Shape:
        - Prediction: Can have arbitrary dimensions. Typically :math:`(S, height, width)`, where `S = number of slices`,
          or `(height, width)` for single image segmentation tasks.
        - Target: Must have the same dimensions as the prediction.
        - Output: Scalar.
    """

    assert prediction.device == target.device
    assert prediction.shape == target.shape

    flattened_prediction = prediction.view(-1).float()
    flattened_target = target.view(-1).float()

    true_positives = (flattened_prediction * flattened_target).sum()
    true_positives_false_negatives = flattened_target.sum()
    return (true_positives + smoothing) / (true_positives_false_negatives + smoothing)


def specificity(
    prediction: torch.Tensor, target: torch.Tensor, smoothing: float = 0
) -> torch.Tensor:
    r"""
    Computes the specificity from a predicted segmentation mask and the target mask:

        :math:`Specificity = \frac{TN}{TN + FP}`

    Using the `smoothing` parameter, Laplacian smoothing can be applied:

        :math:`Specificity = \frac{TN + \lambda}{TN + FP + \lambda}`

    Note:
        In this method, the `prediction` tensor is considered as a segmentation mask of a single 2D or 3D image for a
        given class and thus it aggregates over :math:`TP`, and :math:`FN` over all channels and dimensions.

    Args:
        prediction (Tensor): The predicted segmentation mask, where each value is in :math:`[0, 1]`.
        target (Tensor): The target segmentation mask, where each value is in :math:`\{0, 1\}`.
        smoothing (float, optional): Laplacian smoothing factor.

    Returns:
        Tensor: Specificity.

    Shape:
        - Prediction: Can have arbitrary dimensions. Typically :math:`(S, height, width)`, where `S = number of slices`,
          or `(height, width)` for single image segmentation tasks.
        - Target: Must have the same dimensions as the prediction.
        - Output: Scalar.
    """

    assert prediction.device == target.device
    assert prediction.shape == target.shape

    flattened_prediction = prediction.view(-1).float()
    flattened_target = target.view(-1).float()

    ones = torch.ones(flattened_prediction.shape, device=prediction.device)

    true_negatives = ((ones - flattened_prediction) * (ones - flattened_target)).sum()
    true_negatives_false_positives = (ones - flattened_target).sum()
    return (true_negatives + smoothing) / (true_negatives_false_positives + smoothing)


def _distance_matrix(
    first_point_set: torch.Tensor, second_point_set: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes Euclidean distances between all points in the first tensor and all points in the second tensor.
    Args:
        first_point_set (Tensor): A tensor representing a list of points in an Euclidean space (typically 2d or 3d)
        second_point_set (Tensor): A tensor representing a list of points in an Euclidean space (typically 2d or 3d).
    Returns:
        Matrix containing the Euclidean distances between all points in the first tensor and all points.
    Shape:
        first_point_set: :math:`(N, D)` where :math:`N` is the number of points in the first set and :math:`D` is the
            dimensionality of the Euclidean space.
        second_point_set: :math:`(M, D)` where :math:`M` is the number of points in the second set and :math:`D` is the
            dimensionality of the Euclidean space.
        Output: :math:`(N, M)` where :math:`N` is the number of points in the first set and :math:`M` is the number of
            points in the second set
    """

    if len(first_point_set) == 0:
        return torch.as_tensor(float("nan"), device=first_point_set.device)

    distances = torch.cdist(first_point_set.float(), second_point_set.float())
    minimum_distances, _ = distances.min(axis=-1)

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
    assert _is_binary(input_image), "Input must be binary."

    if input_image.dim() == 2:
        max_pooling = torch.nn.MaxPool2d(
            3, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )
    elif input_image.dim() == 3:
        max_pooling = torch.nn.MaxPool2d(
            3, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False
        )

    # to implement erosion filtering, max ppooling with a 3x3 kernel is applied to the inverted image
    inverted_input = torch.as_tensor(1.0, device=input_image.device) - input_image

    # pad image with ones to maintain the input's dimensions
    inverted_input_padded = torch.nn.functional.pad(
        inverted_input, (1, 1, 1, 1), "constant", 1
    )

    # apply the max pooling and invert the result
    return torch.as_tensor(1.0, device=input_image.device) - max_pooling(
        inverted_input_padded.unsqueeze(dim=0)
    ).squeeze(dim=0)


# pylint: disable=too-many-locals
def hausdorff_distance(
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
        - Prediction: Can have arbitrary dimensions. Typically :math:`(S, height, width)`, where `S = number of slices`,
          or `(height, width)` for single image segmentation tasks.
        - Target: Must have the same dimensions as the prediction.
        - All_image_locations: Must contain one element per-pixel in the prediction. Each element represents an
            n-dimensional point. Typically :math:`(S \cdot height \cdot width, 3)`, where `S = number of slices`, or
            :math:`(height \cdot width, 2)`
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
    assert _is_binary(prediction), "Predictions must be binary."
    assert _is_binary(target), "Target must be binary."

    if torch.count_nonzero(prediction) == 0 or torch.count_nonzero(target) == 0:
        return torch.as_tensor(float("nan"), device=prediction.device)

    if all_image_locations is None:
        all_image_locations = _compute_all_image_locations(
            prediction.shape, device=prediction.device
        )

    # to reduce computational effort, the distance calculations are only done for the border points of the predicted and
    # the target shape
    # to identify border pointds, binary erosion is used
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
            torch.tensor(list(prediction.shape), dtype=torch.float)
        )
        distances = distances / maximum_distance

    return torch.quantile(distances, q=percentile, keepdim=False).float()


class DiceScore(torchmetrics.Metric):
    """
    Computes the Dice similarity coefficient (DSC). Can be used for 3D images whose slices are scattered over multiple
    batches.

    Args:
        smoothing (int, optional): Laplacian smoothing factor.
    """

    def __init__(self, smoothing: float = 0):
        super().__init__()
        self.smoothing = smoothing

        self.numerator = torch.tensor(0.0)
        self.denominator = torch.tensor(0.0)
        self.add_state("numerator", torch.tensor(0.0))
        self.add_state("denominator", torch.tensor(0.0))

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        assert prediction.device == target.device
        assert prediction.shape == target.shape

        flattened_prediction = prediction.view(-1).float()
        flattened_target = target.view(-1).float()

        self.numerator += (flattened_prediction * flattened_target).sum()
        self.denominator += flattened_prediction.sum() + flattened_target.sum()

    def compute(self) -> torch.Tensor:
        """
        Computes the DSC  over all slices that were registered using the `update` method.

        Returns:
            Tensor: Dice similarity coefficient.
        """

        return (2.0 * self.numerator + self.smoothing) / (
            self.denominator + self.smoothing
        )


class Sensitivity(torchmetrics.Metric):
    """
    Computes the sensitivity. Can be used for 3D images whose slices are scattered over multiple batches.

    Args:
        smoothing (int, optional): Laplacian smoothing factor.
    """

    def __init__(self, smoothing: float = 0):
        super().__init__()
        self.smoothing = smoothing
        self.true_positives = torch.tensor(0.0)
        self.true_positives_false_negatives = torch.tensor(0.0)
        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("true_positives_false_negatives", torch.tensor(0.0))

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        assert prediction.device == target.device
        assert prediction.shape == target.shape

        flattened_prediction = prediction.view(-1).float()
        flattened_target = target.view(-1).float()

        self.true_positives += (flattened_prediction * flattened_target).sum()
        self.true_positives_false_negatives += flattened_target.sum()

    def compute(self) -> torch.Tensor:
        """
        Computes the sensitivity  over all slices that were registered using the `update` method.

        Returns:
            Tensor: Sensitivity.
        """

        return (self.true_positives + self.smoothing) / (
            self.true_positives_false_negatives + self.smoothing
        )


class Specificity(torchmetrics.Metric):
    """
    Computes the specificity. Can be used for 3D images whose slices are scattered over multiple batches.

    Args:
        smoothing (int, optional): Laplacian smoothing factor.
    """

    def __init__(self, smoothing: float = 0):
        super().__init__()
        self.smoothing = smoothing
        self.true_negatives = torch.tensor(0.0)
        self.true_negatives_false_positives = torch.tensor(0.0)
        self.add_state("true_negatives", torch.tensor(0.0))
        self.add_state("true_negatives_false_positives", torch.tensor(0.0))

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        assert prediction.device == target.device
        assert prediction.shape == target.shape

        flattened_prediction = prediction.view(-1).float()
        flattened_target = target.view(-1).float()

        ones = torch.ones(flattened_prediction.shape, device=prediction.device)

        self.true_negatives += (
            (ones - flattened_prediction) * (ones - flattened_target)
        ).sum()
        self.true_negatives_false_positives += (ones - flattened_target).sum()

    def compute(self) -> torch.Tensor:
        """
        Computes the sensitivity  over all slices that were registered using the `update` method.

        Returns:
            Tensor: Sensitivity.
        """

        return (self.true_negatives + self.smoothing) / (
            self.true_negatives_false_positives + self.smoothing
        )


class HausdorffDistance(torchmetrics.Metric):
    r"""
    Computes the Hausdorff distance. Can be used for 3D images whose slices are scattered over multiple batches.

    Args:
        slices_per_image (int): Number of slices per 3d image.
        percentile (float, optional): Percentile for which the Hausdorff distance is to be calculated, must be in
            :math:`\[0, 1\]`.
        normalize (bool, optional): Whether the Hausdorff distance should be normalized by dividing it by the diagonal
            distance.
    """

    def __init__(
        self,
        slices_per_image: int,
        normalize: bool = False,
        percentile: float = 0.95,
    ):
        super().__init__()
        self.normalize = normalize
        self.percentile = percentile
        self.predictions = []
        self.targets = []
        self.hausdorff_distance = torch.tensor(0.0)
        self.add_state("predictions", [])
        self.add_state("targets", [])
        self.add_state("hausdorff_distance", torch.tensor(0.0))
        self.all_image_locations = None
        self.hausdorff_distance_cached = False
        self.slices_per_image = slices_per_image
        self.number_of_slices = 0

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        assert (
            prediction.shape == target.shape
        ), "Prediction and target must have the same dimensions."
        assert prediction.ndim in [
            2,
            3,
        ], "Prediction and target must have either two or three dimensions."
        self.hausdorff_distance_cached = False

        # we just collect all slices of the image since, for 3d images, the Hausdorff distance needs to be computed over
        # all slices
        # note that this will be memory-intensive if this is done for multiple 3d images in parallel
        self.predictions.append(prediction)
        self.targets.append(target)

        added_slices = 1 if prediction.ndim == 2 else prediction.shape[0]
        self.number_of_slices += added_slices

        if self.number_of_slices == self.slices_per_image:
            self.compute()
        assert self.number_of_slices <= self.slices_per_image

    def compute(self) -> torch.Tensor:
        """
        Computes the Hausdorf distance over all slices that were registered using the `update` method as the maximum per
         slice-distance.

        Returns:
            Tensor: Hausdorff distance.
        """

        if self.hausdorff_distance_cached:
            return self.hausdorff_distance

        if self.predictions[0].ndim == 2:
            predictions = torch.stack(self.predictions)
            targets = torch.stack(self.targets)
        else:
            predictions = torch.cat(self.predictions, dim=0)
            targets = torch.cat(self.targets, dim=0)

        hausdorff_dist = hausdorff_distance(
            predictions,
            targets,
            normalize=self.normalize,
            percentile=self.percentile,
        )

        self.hausdorff_distance = hausdorff_dist
        self.hausdorff_distance_cached = True

        for tensor in self.predictions:
            del tensor
        for tensor in self.targets:
            del tensor

        # free memory
        self.predictions = []
        self.targets = []
        self.number_of_slices = 0

        return hausdorff_dist
