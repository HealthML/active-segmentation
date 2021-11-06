import torch


class SegmentationLoss(torch.nn.Module):
    """
    Base class for implementation of segmentation losses.

    Args:
        smoothing (int, optional): Laplacian smoothing factor.
        reduction (str, optional): Reduction function that is to be used to aggregate the loss values of the images of
            one batch, must be either "mean", "sum" or "none".
    """

    def __init__(self, smoothing: int = 1, reduction: str = "mean"):

        super(SegmentationLoss, self).__init__()
        self.smoothing = smoothing
        if reduction and reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method.")
        self.reduction = reduction

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Aggregates the loss values of the different classes of an image as well as the different images of a batch.

        Args:
            loss (Tensor): Loss to be aggregated.

        Returns:
            Aggregated loss value.

        Shape:
            - Loss: :math:`(N, C)`, where `N = batch size`, and `C = number of classes`, or
             `(N)` for binary segmentation tasks.
            - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N, C)`. Otherwise, scalar.
        """

        assert loss.dim() == 1 or loss.dim() == 2

        # aggregate loss values for all channels and the entire batch
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(SegmentationLoss):
    """
    Implementation of Dice loss for segmentation tasks. The Dice loss originally was formulated in:

        Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for
        volumetric medical image segmentation, 2016.

    In this implementation an adapted version of the Dice loss is used that includes Laplacian smoothing.
    For a discussion on different dice loss implementations, see https://github.com/pytorch/pytorch/issues/1249.

    Args:
        smoothing (int, optional): Laplacian smoothing factor.
        reduction (str, optional): Reduction function that is to be used to aggregate the loss values of the images of
            one batch, must be either "mean", "sum" or "none".
    """

    def __init__(self, smoothing: int = 1, reduction: str = "mean"):
        super(DiceLoss, self).__init__(smoothing, reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask with each channel being either a sharp segmentation mask or
                the output of a sigmoid layer.
            target (Tensor): Target segmentation mask with the same number of channels as the prediction.
        Returns:
            Tensor: Dice loss.

        Shape:
            - Prediction: :math:`(N, C, height, width)`, where `N = batch size`, and `C = number of classes`, or
             `(N, height, width)` for binary segmentation tasks.
            - Target: :math:`(N, C, height, width)`, where each value is in
              :math:`\{0, 1\}`, or `(N, height, width)` for binary segmentation tasks.
            - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N, C)`. Otherwise, scalar.
        """

        assert prediction.shape == target.shape

        flattened_target = target.view(*target.shape[:-2], -1).float()
        flattened_prediction = prediction.view(*prediction.shape[:-2], -1).float()

        intersection = (flattened_prediction * flattened_target).sum(dim=-1)

        # compute loss for each channel
        # since the loss is to be minimized, the loss value is negated
        dice_loss = -1. * (2. * intersection + self.smoothing) / (
                (flattened_target * flattened_target + flattened_prediction * flattened_prediction).sum(dim=-1)
                + self.smoothing)

        return self._reduce_loss(dice_loss)


class FalsePositiveLoss(SegmentationLoss):
    """
    Implementation of false positive loss.

    Args:
        smoothing (int, optional): Laplacian smoothing factor.
        reduction (str, optional): Reduction function that is to be used to aggregate the loss values of the images of
            one batch, must be either "mean", "sum" or "none".
    """

    def __init__(self, smoothing: int = 1, reduction: str = "mean"):
        super(FalsePositiveLoss, self).__init__(smoothing, reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes false positive loss for binary segmentation masks.

        :param prediction: Predicted binary segmentation mask.
        :param target: Target binary segmentation mask.
        :return: Loss value as 1-element tensor.
        """

        assert prediction.shape == target.shape

        flattened_target = target.view(*target.shape[:-2], -1).float()
        flattened_prediction = prediction.view(*prediction.shape[:-2], -1).float()

        false_positives = ((1 - flattened_target) * flattened_prediction).sum(-1)
        positives = flattened_prediction.sum(-1)

        # in contrast to the Dice loss, the smoothing term is only added to the denominator
        # if there are no positives at all, this will yield an optimal loss value of zero
        fp_loss = false_positives / (self.smoothing + positives)

        return self._reduce_loss(fp_loss)
