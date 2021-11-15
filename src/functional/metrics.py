"""
Module containing model evaluation metrics.

The metric implementations are based on the TorchMetrics framework. For instructions on how to implement custom metrics
with this framework, see https://torchmetrics.readthedocs.io/en/latest/pages/implement.html.
"""
import torch
import torchmetrics


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

    assert prediction.shape == target.shape

    flattened_prediction = prediction.view(-1).float()
    flattened_target = target.view(-1).float()

    ones = torch.ones(flattened_prediction.shape)

    true_negatives = ((ones - flattened_prediction) * (ones - flattened_target)).sum()
    true_negatives_false_positives = (ones - flattened_target).sum()
    return (true_negatives + smoothing) / (true_negatives_false_positives + smoothing)


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
        assert prediction.shape == target.shape

        flattened_prediction = prediction.view(-1).float()
        flattened_target = target.view(-1).float()

        ones = torch.ones(flattened_prediction.shape)

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
