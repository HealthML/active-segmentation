import torch


class DiceLoss(torch.nn.Module):
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

        super(DiceLoss, self).__init__()
        self.smoothing = smoothing
        if reduction and reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method.")
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask
            target (Tensor): Target segmentation mask
        Returns:
            Tensor: Dice loss.

        Shape:
            - Prediction: :math:`(N, C, height, width)`, where `N = batch size`, and `C = number of classes`, or
             `(N, height, width)` for binary segmentation tasks.
            - Target: :math:`(N, C, height, width)`, where each value is in
              :math:`\{0, 1\}`, or `(N, height, width)` for binary segmentation tasks.
            - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N)`. Otherwise, scalar.
        """

        class_predictions = (prediction > 0.5).float()
        class_predictions.requires_grad = True

        assert prediction.shape == target.shape

        flattened_target = target.view(*target.shape[:-2], -1).float()
        flattened_prediction = class_predictions.view(*class_predictions.shape[:-2], -1).float()

        intersection = (flattened_prediction * flattened_target).sum(dim=-1)

        # compute loss for each channel
        # since the loss is to be minimized, the loss value is negated
        dice_loss = -1. * (2. * intersection + self.smoothing) / (
                (flattened_target * flattened_target + flattened_prediction * flattened_prediction).sum(dim=-1)
                + self.smoothing)

        # compute mean of channel losses
        dice_loss = dice_loss.mean(-1)

        # aggregate loss values for the entire batch
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        return dice_loss
