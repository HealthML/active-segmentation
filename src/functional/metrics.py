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


def recall(
    prediction: torch.Tensor, target: torch.Tensor, smoothing: float = 0
) -> torch.Tensor:
    r"""
    Computes the recall from a predicted segmentation mask and the target mask:

        :math:`Recall = \frac{TP}{TP + FN}`

    Using the `smoothing` parameter, Laplacian smoothing can be applied:

        :math:`DSC = \frac{TP + \lambda}{TP + FN + \lambda}`

    Note:
        In this method, the `prediction` tensor is considered as a segmentation mask of a single 2D or 3D image for a
        given class and thus it aggregates over :math:`TP`, and :math:`FN` over all channels and dimensions.

    Args:
        prediction (Tensor): The predicted segmentation mask, where each value is in :math:`[0, 1]`.
        target (Tensor): The target segmentation mask, where each value is in :math:`\{0, 1\}`.
        smoothing (float, optional): Laplacian smoothing factor.

    Returns:
        Tensor: Recall.

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


class Recall(torchmetrics.Metric):
    """
    Computes the Recall. Can be used for 3D images whose slices are scattered over multiple batches.

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
        Computes the recall  over all slices that were registered using the `update` method.

        Returns:
            Tensor: Recall.
        """

        return (self.true_positives + self.smoothing) / (
            self.true_positives_false_negatives + self.smoothing
        )
