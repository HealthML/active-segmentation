"""
Module containing model evaluation metrics.

The metric implementations are based on the TorchMetrics framework. For instructions on how to implement custom metrics
with this framework, see https://torchmetrics.readthedocs.io/en/latest/pages/implement.html.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torchmetrics
from scipy.ndimage.morphology import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)


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


def _distances_to_surface(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    r"""
    For each border point of a predicted shape, computes the distance to the nearest point of the target shape.

    Args:
        prediction: The predicted segmentation mask, where each value is in :math:`\{0, 1\}`.
        target: The target segmentation mask, where each value is in :math:`\{0, 1\}`.

    Returns:
        Array of distance values with one entry per border point of the predicted shape.
    """

    # code adapted from
    # https://github.com/loli/medpy/blob/39131b94f0ab5328ab14a874229320efc2f74d98/medpy/metric/binary.py#L1195

    assert (
        prediction.shape == target.shape
    ), "Prediction and target must have the same dimensions."
    assert prediction.ndim in [
        2,
        3,
        4,  # ToDo: Only input one channel.
    ], "Prediction and target must have either two or three dimensions."

    erosion_structure = generate_binary_structure(prediction.ndim, connectivity=1)

    # extract 1-pixel border line of predicted shapes using XOR
    prediction_border = prediction ^ binary_erosion(
        prediction, structure=erosion_structure, iterations=1
    )

    # scipys distance transform is calculated only inside the borders of the foreground objects, therefore the target
    # has to be reversed
    distances_to_target = distance_transform_edt(~target, sampling=None)

    return distances_to_target[prediction_border]


def hausdorff_distance(
    prediction: torch.Tensor, target: torch.Tensor, percentile: float = 0.95
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
        percentile (float, optional): Percentile for which the Hausdorff distance is to be calculated, must be in
            :math:`\[0, 1\]`.

    Returns:
        Tensor: Hausdorff distance.

    Shape:
        - Prediction: Can have arbitrary dimensions. Typically :math:`(S, height, width)`, where `S = number of slices`,
          or `(height, width)` for single image segmentation tasks.
        - Target: Must have the same dimensions as the prediction.
        - Output: Scalar.
    """

    # adapted code from
    # https://github.com/PiechaczekMyller/brats/blob/eb9f7eade1066dd12c90f6cef101b74c5e974bfa/brats/functional.py#L135

    assert prediction.device == target.device
    assert (
        prediction.shape == target.shape
    ), "Prediction and target must have the same dimensions."
    assert (
        prediction.dim() == 2
        or prediction.dim() == 3
        or prediction.dim() == 4  # ToDo: Only input one channel.
    ), "Prediction and target must have either two or three dimensions."
    assert _is_binary(prediction), "Predictions must be binary."
    assert _is_binary(target), "Target must be binary."

    prediction = prediction.cpu().detach().numpy().astype(np.bool)
    target = target.cpu().detach().numpy().astype(np.bool)

    if np.count_nonzero(prediction) == 0 or np.count_nonzero(target) == 0:
        return torch.as_tensor(float("nan"))

    distances_to_target = _distances_to_surface(prediction, target)
    # pylint: disable=arguments-out-of-order
    distances_to_prediction = _distances_to_surface(target, prediction)

    distances = np.hstack((distances_to_target, distances_to_prediction))

    return torch.quantile(
        torch.from_numpy(distances), q=percentile, keepdim=False
    ).float()


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
        percentile (float, optional): Percentile for which the Hausdorff distance is to be calculated, must be in
            :math:`\[0, 1\]`.
        dim (int, optional): The dimensionality of the input. Must be either 2 or 3. Defaults to 2.
        slices_per_image (int, optional): Number of slices per 3d image. Must be specified if `dim` is 2.
    """

    def __init__(self,
        percentile: float = 0.95,
        dim: int = 2,
        slices_per_image: Optional[int] = None
        ):
        super().__init__()
        self.percentile = percentile
        self.predictions = []
        self.targets = []
        self.hausdorff_distance = torch.tensor(0.0)
        self.add_state("predictions", [])
        self.add_state("targets", [])
        self.add_state("hausdorff_distance", torch.tensor(0.0))
        self.all_image_locations = None
        self.hausdorff_distance_cached = False

        if dim not in [2, 3]:
            raise ValueError(
                f"Dimensionality must be either 2 or 3, but is {dim} instead."
            )

        if dim == 2 and slices_per_image is None:
            raise ValueError(
                "For 2d inputs the `slices_per_image` parameter needs to be specified."
            )

        self.dim = dim
        self.slices_per_image = slices_per_image

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.hausdorff_distance_cached = False

        # we just collect all slices of the image since, for 3d images, the Hausdorff distance needs to be computed over
        # all slices
        # note that this will be memory-intensive if this is done for multiple 3d images in parallel
        self.predictions.append(prediction)
        self.targets.append(target)

        if self.dim == 3 or len(self.predictions) == self.slices_per_image:
            self.compute()

    def compute(self) -> torch.Tensor:
        """
        Computes the Hausdorf distance over all slices that were registered using the `update` method as the maximum per
         slice-distance.

        Returns:
            Tensor: Hausdorff distance.
        """

        if self.hausdorff_distance_cached:
            return self.hausdorff_distance

        if self.dim == 2:
            predictions = torch.stack(self.predictions)
            targets = torch.stack(self.targets)
        else:
            predictions = torch.cat(self.predictions, dim=0)
            targets = torch.cat(self.targets, dim=0)

        hausdorff_dist = hausdorff_distance(
            predictions,
            targets,
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
        self.all_image_locations = None

        return hausdorff_dist