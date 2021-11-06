import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, smoothing=1, reduction: str = "mean"):
        """
        Implementation of Dice loss for binary segmentation tasks. The Dice loss originally was formulated in:

            Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for
            volumetric medical image segmentation, 2016.

        In this implementation an adapted version of the Dice loss is used that includes Laplacian smoothing.
        For a discussion on different dice loss implementations, see https://github.com/pytorch/pytorch/issues/1249.

        :param smoothing: Laplacian smoothing factor.
        :param reduction: Reduction function that is to be used to aggregate the loss values of the images of one batch,
            must be either "mean", "sum" or "none".
        """

        super(DiceLoss, self).__init__()
        self.smoothing = smoothing
        if reduction and reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method.")
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param prediction: Predicted binary segmentation mask with dimensions [batch_size, img_height, img_width] or
            [batch_size, 1, img_height, img_width]
        :param target: Target binary segmentation mask with dimensions [batch_size, img_height, img_width] or
            [batch_size, 1, img_height, img_width]
        :return: Loss value as a 1-element tensor.
        """

        class_predictions = (prediction > 0.5).float()
        class_predictions.requires_grad = True

        flattened_target = target.view(target.shape[0], -1).float()
        flattened_prediction = class_predictions.view(class_predictions.shape[0], -1).float()

        intersection = (flattened_prediction * flattened_target).sum(dim=1)

        # since the loss is to be minimized, the loss value is negated
        dice_loss = -1. * (2. * intersection + self.smoothing) / (
                (flattened_target * flattened_target + flattened_prediction * flattened_prediction).sum(dim=1)
                + self.smoothing)

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        return dice_loss
