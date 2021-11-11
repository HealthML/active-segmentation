# pylint: disable=all
import torch


def dsc(input: torch.Tensor, target: torch.Tensor, smoothing=1) -> torch.Tensor:
    """
    Computes Dice similarity coefficient (DSC) for two binary segmentation masks.

    :param input: First binary segmentation mask.
    :param target: Second binary segmentation mask.
    :param smoothing: Smoothing factor.
    :return: Dice similarity coefficient as 1-element tensor.
    """

    flattened_target = torch.flatten(target).float()
    flattened_input = torch.flatten(input).float()
    intersection = (flattened_target * flattened_input).sum()
    score = (2.0 * intersection + smoothing) / (
        (flattened_target * flattened_target).sum()
        + (flattened_input * flattened_input).sum()
        + smoothing
    )

    return score


def recall(input: torch.Tensor, target: torch.Tensor, smoothing=1) -> torch.Tensor:
    """
    Computes number of false positive pixels for a predicted binary segmentation mask.

    :param input: Predicted binary segmentation mask.
    :param target: Target binary segmentation mask.
    :param smoothing: Smoothing factor.
    :return: Number of false positives as 1-element tensor.
    """

    flattened_target = torch.flatten(target).float()
    flattened_input = torch.flatten(input).float()

    true_positives = flattened_target.sum()
    intersection = (flattened_target * flattened_input).sum()
    return (intersection + smoothing) / (true_positives + smoothing)


class DiceLoss(torch.nn.Module):
    def __init__(self, smoothing=1):
        super(DiceLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes dice loss for binary segmentation masks.

        :param input: Predicted binary segmentation mask.
        :param target: Target binary segmentation mask.
        :return: Loss value as 1-element tensor.
        """

        loss = 1 - dsc(target, input)
        return loss


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, smoothing=1):
        super(BCEDiceLoss, self).__init__()
        self.smoothing = smoothing
        self.binary_crossentropy_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smoothing)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes a combined loss from dice loss and cross-entropy loss for binary segmentation masks.

        :param input: Predicted binary segmentation mask.
        :param target: Target binary segmentation mask.
        :return: Loss value as 1-element tensor.
        """

        loss = self.binary_crossentropy_loss(input, target) + self.dice_loss(
            input, target
        )
        return loss


class FalsePositiveLoss(torch.nn.Module):
    def __init__(self, smoothing=1):
        super(FalsePositiveLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes false positive loss for binary segmentation masks.

        :param input: Predicted binary segmentation mask.
        :param target: Target binary segmentation mask.
        :return: Loss value as 1-element tensor.
        """

        flattened_target = torch.flatten(target).float()
        flattened_input = torch.flatten(input).float()

        false_positives = ((1 - flattened_target) * flattened_input).sum()
        positives = flattened_input.sum()

        return (smooth + false_positives) / (smooth + positives)


class FalsePositiveDiceLoss(torch.nn.Module):
    def __init__(self, smoothing=1):
        super(FalsePositiveDiceLoss, self).__init__()
        self.smoothing = smoothing
        self.fp_loss = FalsePositiveLoss(smoothing)
        self.dice_loss = DiceLoss(smoothing)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes a combined loss from dice loss and false positive loss for binary segmentation masks.

        :param input: Predicted binary segmentation mask.
        :param target: Target binary segmentation mask.
        :return: Loss value as 1-element tensor.
        """

        return self.fp_loss(input, target) + self.dice_loss(input, target)
